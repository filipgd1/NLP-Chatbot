#!/usr/bin/env python
# coding: utf-8

# # Chat bot question answering.
# 
# The goal of your project is to create a chatbot model that provides answers on client questions.   
# Your goal is to divide your dataset on several sections and to make multi-level classifier that will classify the section and later the most reasonable(closest) answer.    
# Take care about text-preprocessing, stop words removal, and sentence vectorization
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

#get_ipython().run_line_magic('matplotlib', 'inline')


## Load Data

brainster_df = pd.read_csv('dataset/dataset_brainster.csv')

questions = brainster_df.questions
other_col = brainster_df.drop(columns='questions', axis=0)


questions_array = questions.to_numpy()

# # Data Preprocessing


# # Define Model Architecture

## Count Vectorizer Model

count_vector_model = CountVectorizer(stop_words=stop_words_list, strip_accents='unicode')

count_vector_features = count_vector_model.fit_transform(questions_translated)
questions_df = pd.DataFrame(data = count_vector_features.todense(), 
                            columns=count_vector_model.get_feature_names())


## TF-IDF

tf_idf_model = TfidfVectorizer(stop_words=stop_words_list, strip_accents='unicode')

tf_idf_features = tf_idf_model.fit_transform(questions_translated)
questions_df_tfidf = pd.DataFrame(data = tf_idf_features.todense(), 
                            columns=tf_idf_model.get_feature_names())

## TfidfVectorizer Ngrams

tf_idf_ngram_model = TfidfVectorizer(ngram_range=(1,2), stop_words=stop_words_list, strip_accents='unicode')


tf_idf_ngram_features = tf_idf_ngram_model.fit_transform(questions_translated)
questions_df_tfidf_ngram = pd.DataFrame(data = tf_idf_ngram_features.todense(), 
                            columns=tf_idf_ngram_model.get_feature_names())


## Test Part

client = text_preprocessed(["колку трае академијата за дигитален маркетинг"], stop_words_list)


# CountVectorizer

client_count_feature = count_vector_model.transform(client)


# TfIdfVectorizer

client_tfidf_feature = tf_idf_model.transform(client)


# TfIdfVectorizer Ngram

client_tfidf_ngram_feature = tf_idf_ngram_model.transform(client)


## Example :  
#  Find the closest dataset question based on user defined question

cv_answer, cv_index, cv_cosine = distance_vectors(other_col, count_vector_features, client_count_feature)

# print("Most equal answer :", cv_answer)
# print('Raw number', cv_index)
# print('Cosine coefficient:', cv_cosine)


tfidf_answer, tfidf_index, tfidf_cosine = distance_vectors(other_col, tf_idf_features, client_tfidf_feature)

# print("Most equal answer :", tfidf_answer)
# print('Raw number', tfidf_index)
# print('Cosine coefficient:', tfidf_cosine)


ngram_answer, ngram_index, ngram_cosine = distance_vectors(other_col, tf_idf_ngram_features, client_tfidf_ngram_feature)

# print("Most equal answer :", ngram_answer)
# print('Raw number', ngram_index)
# print('Cosine coefficient:', ngram_cosine)
