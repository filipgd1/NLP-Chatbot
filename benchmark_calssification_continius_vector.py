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


benchmark = pd.read_csv('dataset/chatbot_banchmark.csv')

benchmark_cat = benchmark.drop(columns='questions', axis=0)

benchmark_questions = benchmark.questions

benchmark_questions[1:5]


def benchmark_predict(benchmark, stop_words):
    benchmark_embed = []
    for i in benchmark:
        benchmark_embed.append(bpemb_mk.embed(text_preprocessed([i], stop_words)).mean(axis=0))
    
    benchmark_reshape = []
    for j in benchmark_embed:
        benchmark_reshape.append(j.reshape(1, 300))
    
    benchmark_predict = []
    for k in benchmark_reshape:
            benchmark_predict.append(np.argmax(model.predict(k)))
    
    return benchmark_predict


benchmark_class = benchmark_predict(benchmark_questions, stop_words_list)



precision, recall, relu_fscore, support = score(benchmark_cat.category_id, benchmark_class, average='macro')
#print(np.round(relu_accuracy, 3), np.round(relu_fscore, 3))


labels = ['G', 'M', 'GD', 'FEP', 'FSP', 'DS', 'QA', 'UX/UI']

becnhmark_report = classification_report(benchmark_cat.category_id, benchmark_class)
#print('Classification Report:\n', becnhmark_report)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(benchmark_cat.category_id, benchmark_class),
                              display_labels=labels, )

disp = disp.plot(values_format='3')
plt.show()

df = pd.DataFrame({'questions':questions_translated})

df = pd.concat([df, other_col.category_id], axis=1)

predicted_class_datasets = []
for i in benchmark_class:
    predicted_class_datasets.append(df.questions[df.category_id == i])
    
#print(predicted_class_datasets[1:3])    


def answer_embed(dataset, benchmarks, stop_words):
    # embeding of question from dataset
    embed_questions = []
    for question in dataset:
        embed_questions.append(bpemb_mk.embed(question).mean(axis=0))

    # preproccess and ebmeding of client question
    prepared_banchmarks = text_preprocessed(benchmarks, stop_words)
    banchmark_embedding =  []
    for banchmark in prepared_banchmarks:
        banchmark_embedding.append(bpemb_mk.embed(banchmark).mean(axis=0))
        
    # cdist give us how much is the error similarity
    benchmark_distences = []
    for i in banchmark_embedding:
        benchmark_distences.append(cdist([i], embed_questions, "cosine")[0])
    
    results = []
    for i in benchmark_distences:
        results.append(zip(range(len(i)), i))
    
    sorted_results = []
    for i in results:
        sorted_results.append(sorted(i, key=lambda x: x[1]))
    
    best_scores = []
    for j in sorted_results:
        best_scores.append(1-j[0][1])

    return best_scores    

answer_result = answer_embed(questions_translated, benchmark_questions, stop_words_list)

#print("Benchmark overal result without classification: ", np.array(answer_result).mean())




