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


# # Word Embbeding


bpemb_mk = BPEmb(lang="mk", dim=300)


def result_embed(questions, clinet_question, stop_words):
# embeding of question from dataset
    embed_questions = []
    for question in questions:
        embed_questions.append(bpemb_mk.embed(question).mean(axis=0))

# preproccess and ebmeding of client question
    prepared_question = text_preprocessed([clinet_question], stop_words)

    for query in [prepared_question]:
        query_embedding = bpemb_mk.embed(query).mean(axis=0)
        
        # cdist give us how much is the error similarity
        distances = cdist([query_embedding], embed_questions, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        
        print("Прашање:", query)
        print("\n======================\n")
        print("\nTop 5 најдобри резултати:\n")

        for idx, distance in results[0:10]:
            print(other_col.answers[idx].strip(),'\n', "(Score: %.4f)" % (1-distance),'\n', "Index: ", idx)
    

result_embed(questions_translated, 'dali rabotite vo pythorch', stop_words_list)
