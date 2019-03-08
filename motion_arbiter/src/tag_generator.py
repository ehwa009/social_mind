#!/usr/bin/env python
#-*- encoding: utf8 -*-

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

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
    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation = "softmax"))
    return model

def train():
    data_path = rospkg.RosPack().get_path('motion_arbiter') + '/data/intent_data.csv'
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

    word_tokenizer = create_tokenizer(cleaned_words)
    vocab_size = len(word_tokenizer.word_index) + 1
    max_length = check_max_length(cleaned_words)
    print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))

    encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
    padded_doc = padding_doc(encoded_doc, max_length)
    print("Shape of padded docs = ",padded_doc.shape)

    #tokenizer with filter changed
    output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
    encoded_output = encoding_doc(output_tokenizer, intent)
    encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
    output_one_hot = one_hot(encoded_output)

    train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, test_size = 0.2)

    print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
    print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

    model = create_model(vocab_size, max_length)
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()

    save_path = rospkg.RosPack().get_path('motion_arbiter') + '/config'
    checkpoint = ModelCheckpoint(save_path + "/model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint, early_stopping])
    rospy.loginfo("train done, and save trained model.")

class TagGenerator:

    def __init__(self):
        # predifined value for the model
        self.MAX_LENGTH = 125

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
        intent, self.unique_intent, sentences = load_dataset(utterance_file)
        # cleaning and tokenize sentences
        cleaned_words = cleaning(sentences)
        self.word_tokenizer = create_tokenizer(cleaned_words)

        rospy.Subscriber('reply', Reply, self.handle_domain_reply)
        self.pub_reply_analyzed = rospy.Publisher('reply_analyzed', ReplyAnalyzed, queue_size=10)
        rospy.loginfo("\033[93m[%s]\033[0m initialized." % rospy.get_name())

    def handle_domain_reply(self, msg):
        remain_tags = ''
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", msg.reply)
        test_word = word_tokenize(clean)

        test_ls = self.word_tokenizer.texts_to_sequences(test_word)
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))
        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        x = padding_doc(test_ls, self.MAX_LENGTH)
        pred = self.model.predict_proba(x)

        # get predicted label
        pred_tag = self.get_final_output(pred, self.unique_intent)

        entity = EntitiesIndex()

        # create ros topic message
        msg = ReplyAnalyzed()
        msg.header.stamp = rospy.Time.now()
        # add contents to the message
        msg.entities.append(entity)
        msg.sents.append(remain_tags + ' ' + clean)
        msg.act_type.append(pred_tag + '/%d'%len(clean))

        self.pub_reply_analyzed.publish(msg)

    def get_final_output(self, pred, classes):
        predictions = pred[0]
        classes = np.array(classes) 
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)

        tag_idx = list(predictions).index(max(predictions))
        pred_tag = classes[tag_idx]

        return pred_tag
   

if __name__ == '__main__':
    rospy.init_node('tag_generator', anonymous=False)
    # train()
    m = TagGenerator()
    rospy.spin()
    
