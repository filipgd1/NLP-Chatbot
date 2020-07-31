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


def distance_vectors(answer, features, client_feature):
    cosine_function = lambda a, b : round(np.inner(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), 3)
    distances = []
    for vector in features.toarray():
        for clietn_vector in client_feature.toarray():
            cosine = cosine_function(vector, clietn_vector)
            distances.append(cosine)
    index = np.argmax(distances)
    max_cosine = max(distances)
    return answer.answers[index], index, max_cosine   

