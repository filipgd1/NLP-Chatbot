# Brainster Chatbot - Rubik

# Contents
- [Description](#description)
- [How does it work?](#howdoesitwork)
- [Project/Chat Architecture](#architecture)
	- [Data and Dataset Generation](#dataset)
	- [Dataset Preprocessing](#preprocessing)
	- [Classification Model Traning](#training)
	- [Responding to Questions](#responding)
	- [NLP Techniques Used](#nlpused)
	- [Clasification Algorithms Used](#classused)
- [Summary of results & benchmark](#benchmark)
- [Requirements](#requirements)
- [Team members](#team)
- [Special Thanks](#thanks)


## Description {#description}
Brainster's Chatbot Rubik is a machine learning based conversational dialog engine build in Python and its libraries which makes it possible to generate responses based on collections of known conversations. The main purpose of the Brainster Chatbot is asnwering questions to visitors of the [website](https://brainster.co/) and [Facebook business page](https://www.facebook.com/brainster.co). Questions may be related with all academies, courses and bootcapms organized by Brainster. Moreover, you can get more familiar with all departments that one day you will be part of it.

	* Видео со проба од ботот (screen capture)



## How does it work? {#howdoesitwork}
Once active, the Chatbot starts to communicate. Each time a user enters a query, the bot provides an appropriate response based on its training. The bot accepts queries in Macedonian, and responds in Macedonian. Input shoud preferably be in cyrillic in order to maximize the quality of the answers. Latin inputs are automatically converted into cyrillic, but such a case some characters/combinations of characters could be transliterated wrongly.




## Project/Chat Architecture {#architecture}

### Data and Dataset Generation {#dataset}
Dataset generation started with an set of 300+ questions asked by people which Brainster had received via email or social media. All questions were classified into 8 classes, 7 of which related to an Academy offered by Brainster (Digital Marketing, Graphical Design, Data Science, Front-end Programming, Full-stack Programming, Software  Testing, UX/UI), and one class for general questions. The initial set of questions was expanded by more than tenfold (to 3100+ questions), by writing new, or by rewriting existing questions with slightly modified wording in order to capture the nuances (question diversification).

### Dataset Preprocessing {#preprocessing}
The questions in the dataset were individually processed as described in the process outlined further.

1. Any latin characters in the question were converted to cyrillic characters.
2. Punctuation and stop-words were removed from the question, and the question was then tokenized.
3. The question was vectorized using [word-embedding](https://nlp.h-its.org/bpemb/). The output vector has of dimesion 300.

The final outcome is a dataset of 300-by-1 vectors paired with their resprective class. A classification model is then trained on this set.

### Classification Model Traning {#training}
Several classification models were trained and tested before deciding which one to use. Early on during the testing it became evident that Random Forest classifier, XGBoost classifier, and a neural network based classifier performed best (no worse than low 90% on any validation accuracy), while the other classifiers performed somewhat worse (Naive Bayes, k-Nearest Neighbors, Gradient Boost, ADA Boost; validation accuracy in the high 80%). The final decision was to use the classifier based on neural networks which has been performing at validation accuracy of 99.21%.

Apart from these models, we tested the performance of the [BERT pre-trained language model](https://github.com/google-research/bert). Its performance was only slightly worse than our own approach.

### Responding to Questions {#responding}
User input queries are processed in the same manner as described in Dataset Preprocessing. Once the query is transformed in the required input form, the following process takes place.

1. The query is classified into one of the 8 classes outlined above. 
2. Using the classification from the previous step, cosine similarity is used to determine what question in the appropriate class is closest to the input query.
3. Finally, the answer to the question identified in the prevous step is produced as a response to the user query.

The user interface of the method is implemented in [Telegram](https://telegram.org/).

![Chatbot architecture](images/chatbot_flow.png)

### NLP Techniques Used {#nlpused}
The following natural language processing techniques were tried and tested:

- [ ] CountVectorizer
- [ ] TF-IDF
- [ ] TF-IDF ngrams
- [x] Word Embedding
- [ ] BERT

### Clasification Algorithms Used {#classused}
The following classification methods were tried and tested:

- [ ] RandomForest
- [ ] XGBoost
- [ ] NaiveBayes
- [ ] KNN
- [x] Neural Networks


## Summary of results & benchmark {#benchmark}
* Precision, recall, TP, FP, TN, FN...

## Requirements {#requirements}
Check [this list](requirements.txt) for details about modules and versions used in this implementation.


## Team members {#team}
[Contribution guidelines for this project](CONTRIBUTING.md)

* [Martina Nestorovska](https://www.linkedin.com/in/martina-nestorovska-b367ba8/)
* [Gabriela Bonkova](https://www.linkedin.com/in/gabriela-bonkova-a25607194/)
* [Filip Nikolovski](https://www.linkedin.com/in/filip-nikolovski-a26559ab/)
* [Aleksandar Gjurcinoski](https://www.linkedin.com/in/aleksandar-gjurcinoski-7594a242/)


## Special Thanks to Kiril Cvetkov :) {#thanks}




Да се проверат прашањата
Да се направи chit-chat

