#!/usr/bin/env python
# coding: utf-8

# # Chat bot question answering.
# 
# The goal of your project is to create a chatbot model that provides answers on client questions.   
# Your goal is to divide your dataset on several sections and to make multi-level classifier that will classify the section and later the most reasonable(closest) answer.    
# Take care about text-preprocessing, stop words removal, and sentence vectorization
# 
# This project consists of the following activities :  
# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import string
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support as score

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adadelta, Adam, SGD, RMSprop

from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix, classification_report, confusion_matrix


from bpemb import BPEmb
import xgboost as xgb



brainster_df = pd.read_csv('dataset/dataset_brainster.csv')


# # Provide and prepare data information


questions = brainster_df.questions
other_col = brainster_df.drop(columns='questions', axis=0)

questions_array = questions.to_numpy()


# # Data Preprocessing

lat_to_cyr = {'kj' : 'ќ', 'gj' : 'ѓ', 'zh' : 'ж', 'ch' : 'ч', 'sh' : 'ш', 'dj' : 'ѓ',
              'a' : 'а', 'b' : 'б', 'c' : 'ц', 'd' : 'д', 'e' : 'е', 'f' : 'ф', 'g' : 'г',
              'h' : 'х', 'i' : 'и', 'j' : 'ј', 'k' : 'к', 'l' : 'л', 'm' : 'м', 'n' : 'н',
              'o' : 'о', 'p' : 'п', 'q' : 'љ', 'r' : 'р', 's' : 'с', 't' : 'т', 'u' : 'у',
              'v' : 'в', 'w' : 'њ', 'x' : 'џ', 'y' : 'ѕ', 'z' : 'з'
             }

stop_words_mkd = pd.read_csv('stop_words.txt').to_numpy()
stop_words_list = []
for words in stop_words_mkd:
    for word in words:
        stop_words_list.append(word)
        

# input (text) must be array

def text_preprocessed(text, stop_words):
    remove_punctuation = []
    for questions in text:
        lower_case = questions.lower()
        string_punct = str(lower_case).translate(str.maketrans('', '', string.punctuation))
        tokenization = nltk.word_tokenize(string_punct)
        stop_words_removed = []
        for token in tokenization:
            if token not in stop_words:
                stop_words_removed.append(token)
        final = ' '.join(stop_words_removed)
        remove_punctuation.append(final)
    
    questions = []
    for question in remove_punctuation:
        for key, value in lat_to_cyr.items():
            question = re.sub(key, value, question.lower())
        questions.append(question)
    return questions


questions_translated = text_preprocessed(questions_array, stop_words_list)


print(questions_translated[1:7])
