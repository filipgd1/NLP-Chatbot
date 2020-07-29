# Brainster Chatbot - Rubik

## Description
Brainster Chatbot is a machine-learning based conversational dialog engine build in Python and its libraries which makes it possible to generate responses based on collections of known conversations. The main purpose of the Brainster Chatbot is asnwering questions to visitors of the [website](https://brainster.co/) and [Facebook business page](https://www.facebook.com/brainster.co). Questions may be related with all academies, courses and bootcapms organized by Brainster. Moreover, you can get more familiar with all departments that one day you will be part of it.

	* Видео со проба од ботот (screen capture)

## How does it work?
Once active, the Chatbot starts to communicate. Each time a user enters a query, the bot provides an appropriate response based on its training. To achieve this the input query is taken through several stages.

## Project/Chat Architecture
![Chatbot architecture](images/chatbot_flow.png)



1. Any latin characters of the input query are converted to cyrillic characters. This is in line with the training of the Chatbot, as all training questions have been in Macedonian and cyrillic.
2. Punctuation and stop-words are removed from the input query.
3. The input query is vectorized.
4. Based on this vector, the original input query is classified in one of eight categories, refering either to one of seven academies offered by Brainster, or as a general question (not academy-speciffic).
5. Using the classification from the previous step, cosine similarity is used to determine what question in the appropriate class is closest to the input query (this is achieved by considering the vectorized forms of both strings).
6. Finally, the answer to the question identified in the prevous step is produced as a response to the original query.



## NLP Algorithms used
[x] BERT
[x] Word Embedding
[x] TF-IDF
[x] TF-IDF ngrams
[x] CountVectorizer

## Clasification Algorithms used
[x] Neural Networks
[x] XGBoost
[x] RandomForest
[x] NaiveBayes
[x] KNN

## Summary of results & benchmark
* Precision, recall, TP, FP, TN, FN...


## Team members
[Contribution guidelines for this project](CONTRIBUTING.md)

* [Martina Nestorovska](https://www.linkedin.com/in/martina-nestorovska-b367ba8/)
* [Gabriela Bonkova](https://www.linkedin.com/in/gabriela-bonkova-a25607194/)
* [Filip Nikolovski](https://www.linkedin.com/in/filip-nikolovski-a26559ab/)
* [Aleksandar Gjurcinoski](https://www.linkedin.com/in/aleksandar-gjurcinoski-7594a242/)


## Requirements & Installation Instructions
* Libraires used for this project: pandas, NumPy, ScyPy, matplotlib, seaborn, nltk, sci-kit Learn, regex, string, keras, bpemb, BERT, xgboost
* Modules used for this project: 

## Special Thanks to Kiril Cvetkov :)




Да се проверат прашањата
Да се направи chit-chat

