# Brainster Chatbot - Rubik

## Description
Brainster Chatbot is a machine-learning based conversational dialog engine build in Python and its libraries which makes it possible to generate responses based on collections of known conversations. The main purpose of the Brainster Chatbot is asnwering questions to visitors of the [website](https://brainster.co/) and [Facebook business page](https://www.facebook.com/brainster.co). Questions may be related with all academies, courses and bootcapms organized by Brainster. Moreover, you can get more familiar with all departments that one day you will be part of it.

	* Видео со проба од ботот (screen capture)



## How does it work?
Once active, the Chatbot starts to communicate. Each time a user enters a query, the bot provides an appropriate response based on its training. To achieve this the input query is taken through several stages.




## Project/Chat Architecture

### Data and Dataset Generation
Dataset generation started with an set of 300+ questions Brainster received via email or social media. Generated questions were classified into 8 classes, 7 of which related to an Academy offered by Brainster (Digital Marketing, Graphical Design, Data Science, Front-end Programming, Full-stack Programming, Software  Testing, UX/UI), and one class for general questions. The initial set of questions was expanded by more than tenfold (to 3100+ questions), by writing new, or by rewriting existing questions with slightly modified wording in order to capture the nuances (question diversification).

### Dataset Preprocessing
The questions in the dataset were individually processed as described in the process outlined further.

1. Any latin characters in the question are converted to cyrillic characters.
2. Punctuation and stop-words are removed from the question. Finally, the question is tokenized.
3. The question is vectorized using [word-embedding](https://nlp.h-its.org/bpemb/). The output vector is of dimesion 300.

The final outcome is a dataset of 300-by-1 vectors paired with their resprective class. A classification model is then trained on this set.

### Classification Model Traning
Several classification models were trained and tested before deciding which one to use. Early on during the testing it became evident that Random Forest classifier, XGBoost classifier, and a NN-based classifier performed best (no worse than low 90% on any validation accuracy), while the other classifiers performed somewhat worse (Naive Bayes, k-Nearest Neighbors, Gradient Boost, ADA Boost; validation accuracy in the high 80%). The final decision was to use the classifier based on neural networks which has been performing at validation accuracy of 99.21%.

### Responding to Questions
User input queries are processed in the same manner as described in Dataset Processing. Once the query is transformed in the required form, the following process takes place.

1. The query is classified into one of the 8 classes outlined above. 
2. Using the classification from the previous step, cosine similarity is used to determine what question in the appropriate class is closest to the input query.
3. Finally, the answer to the question identified in the prevous step is produced as a response to the user query.

The user interface of the method is implemented in [Telegram](https://telegram.org/)

![Chatbot architecture](images/chatbot_flow.png)

## NLP Algorithms used
[x]BERT
[x]Word Embedding
[x]TF-IDF
[x]TF-IDF ngrams
[x]CountVectorizer

## Clasification Algorithms used
[x]Neural Networks
[x]XGBoost
[x]RandomForest
[x]NaiveBayes
[x]KNN

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

