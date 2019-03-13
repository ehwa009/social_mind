#!/usr/bin/env python
#-*- encoding: utf8 -*-

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk import classify, pos_tag, word_tokenize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

from mind_msgs.msg import EntitiesIndex, Reply, ReplyAnalyzed

import rospy
import rospkg
import os
import numpy as np
import pandas as pd
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def load_dataset(filename):
    df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
    # print(df.head())
    intent = df["Intent"]
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])
    return (intent, unique_intent, sentences)

def cleaning(sentences):
    words = []
    for s in sentences:
        # clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        # w = word_tokenize(clean)
        w = word_tokenize(str(s))
        #stemming
        words.append([i.lower() for i in w])
    return words  

def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token

def check_max_length(words):
    return(len(max(words, key = len)))

def encoding_doc(token, words):
    return(token.texts_to_sequences(words))

def padding_doc(encoded_doc, max_length):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

def one_hot(encode):
    o = OneHotEncoder(sparse = False)
    return(o.fit_transform(encode))

def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
    model.add(Bidirectional(LSTM(128)))
    # model.add(LSTM(128))
    # model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation = "softmax"))
    return model

# def predictions(text):
#     clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
#     test_word = word_tokenize(clean)
#     test_word = [w.lower() for w in test_word]
#     test_ls = word_tokenizer.texts_to_sequences(test_word)
#     print(test_word)
#     #Check for unknown words
#     if [] in test_ls:
#         test_ls = list(filter(None, test_ls))

#     test_ls = np.array(test_ls).reshape(1, len(test_ls))

#     x = padding_doc(test_ls, max_length)
#     print(x.shape)

# #     pred = model.predict_proba(x)
#     pred = model.predict_classes(x)


    # return pred 

class TagGenerator:

    def __init__(self):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        # predifined value for the model
        self.MAX_LENGTH = 25
        self.tag_list = ['multiple_choice', 'welcome', 'request', 'greeting', 'inform', 'confirm_answer', 'thanks', 'closing']

        save_path = rospkg.RosPack().get_path('motion_arbiter') + '/config'
        data_path = rospkg.RosPack().get_path('motion_arbiter') + '/data/intent_data.csv'
        try:
            save_path = rospy.get_param('~save_path', default=save_path)
            self.model = load_model(save_path + '/model.h5')
            self.model._make_predict_function()
        except IOError as e:
            rospy.logerr(e)
            exit(-1)
        rospy.loginfo('loaded trained model succeed.')
        

        # call dataset
        try:
            utterance_file = rospy.get_param('~utterance_data', default=data_path)
        except KeyError:
            rospy.logerr('set param utterance_data....')

        utterance_file = os.path.expanduser(utterance_file)
        utterance_file = os.path.abspath(utterance_file)
        
        # load dataset using pandas
        intent, unique_intent, sentences = load_dataset(utterance_file)
        # cleaning and tokenize sentences
        cleaned_words = cleaning(sentences)
        self.word_tokenizer = create_tokenizer(cleaned_words)

        rospy.Subscriber('reply', Reply, self.handle_domain_reply)
        self.pub_reply_analyzed = rospy.Publisher('reply_analyzed', ReplyAnalyzed, queue_size=10)
        rospy.loginfo("\033[93m[%s]\033[0m initialized." % rospy.get_name())

    def handle_domain_reply(self, msg):
        sents = self.sent_detector.tokenize(msg.reply.strip())
        # create ros topic message
        msg = ReplyAnalyzed()
        msg.header.stamp = rospy.Time.now()
        
        for sent in sents:
            # sperate tags and text
            sent_tags = re.findall('(%[^}]+%)', sent)
            sent_text = re.sub('(%[^}]+%)', '', sent).strip()           

            # if task manager select intent we use it, or we use classifier for select intent
            result = ''
            remain_tags = ''
            if not any('sm=' in tag for tag in sent_tags):
                remain_tags = ''

                clean = re.sub(r'[^ a-z A-Z 0-9]', " ", sent_text)
                test_word = word_tokenize(clean)

                test_ls = self.word_tokenizer.texts_to_sequences(test_word)
                if [] in test_ls:
                    test_ls = list(filter(None, test_ls))
                test_ls = np.array(test_ls).reshape(1, len(test_ls))
                x = padding_doc(test_ls, self.MAX_LENGTH)
                
                # predict label
                pred = self.model.predict_classes(x)
                pred_tag = str(self.tag_list[pred[0]])
            else:
                tag_text = sent_tags[0].strip('{}').split('|')
                matching = [s for s in tag_text if "sm=" in s]
                if len(matching) > 1:
                    rospy.logwarn('Only one sm tags allowed...')
                result = matching[0].split('=')[1]
                for s in tag_text:
                    if not "sm=" in s:
                        remain_tags += s + '|'
                if remain_tags != '':
                    remain_tags = '{' + remain_tags.rstrip('|') + '}'

            entity = EntitiesIndex()
            for i in pos_tag(word_tokenize(clean)):
                if(i[1] in ['RB', 'PRP', 'NN', 'PRP$']):
                    entity.entity.append(i[0])
                    entity.entity_index.append(clean.index(i[0]))


            # add contents to the message
            msg.entities.append(entity)
            msg.sents.append(remain_tags + ' ' + sent_text)
            msg.act_type.append(pred_tag + '/%d'%len(clean))

        self.pub_reply_analyzed.publish(msg)

if __name__ == '__main__':
    rospy.init_node('tag_generator', anonymous=False)
    m = TagGenerator()
    rospy.spin()
    
