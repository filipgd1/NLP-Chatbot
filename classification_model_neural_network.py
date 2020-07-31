#!/usr/bin/env python
# coding: utf-8

# # Chat bot question answering.
# 
# The goal of your project is to create a chatbot model that provides answers on client questions.   
# Your goal is to divide your dataset on several sections and to make multi-level classifier that will classify the section and later the most reasonable(closest) answer.    
# Take care about text-preprocessing, stop words removal, and sentence vectorization
# 

# # Word Embbeding

bpemb_mk = BPEmb(lang="mk", dim=300)

# # Classification Model

def dataset_preprocess(X_part, labels):
    embed_questions = []
    for question in X_part:
        embed_questions.append(bpemb_mk.embed(question).mean(axis=0))
    questions_embed = np.array(embed_questions)    
    target = to_categorical(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(questions_embed, target, test_size=0.20, random_state=10)
    
    return X_train, X_test, y_train, y_test
    


X_train, X_test, y_train, y_test = dataset_preprocess(questions_translated, other_col.category_id)


model = Sequential()

model.add(Dense(256, input_dim=300))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))



adam = Adam(learning_rate=0.001)
sgd = SGD(momentum=0.9)
adadelta = Adadelta(learning_rate=0.1)



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


my_callback1 = ModelCheckpoint('best_model.pt', verbose=1, save_best_only=True, mode='max', monitor='val_accuracy')
#my_callback2 = EarlyStopping(patience=7)

my_callbacks = [my_callback1]

model.fit(X_train, y_train, batch_size=None,
    epochs=50,
    verbose=1,
    callbacks=my_callbacks,
    validation_split=0.0,
    validation_data=(X_test, y_test),
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1)


model.load_weights('best_model.pt')


predict_question = model.predict(X_test)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose = 1, callbacks=my_callbacks)
# print("Accuracy relu activation: %.2f%%\n" % (scores[1]*100))


# Convert predictions to 0/1 vectors
y_pred_relu = np.array([int(np.argmax(predict_question[i])) for i in range(len(predict_question))])
y_test_array = [np.argmax(y_test[i]) for i in range(len(y_test))]

relu_accuracy = (y_pred_relu == y_test_array).mean()


precision, recall, relu_fscore, support = score(y_test_array, y_pred_relu, average='macro')
#print(np.round(relu_accuracy, 3), np.round(relu_fscore, 3))


labels = ['G', 'M', 'GD', 'FEP', 'FSP', 'DS', 'QA', 'UX/UI']

cr_we_relu = classification_report(y_test_array, y_pred_relu)
#print('Classification Report: relu \n', cr_we_relu)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_array, y_pred_relu),
                              display_labels=labels, )

disp = disp.plot(values_format='3')

plt.show()


def class_predict(client_question, stop_words):
    client_question_embed = bpemb_mk.embed(text_preprocessed([client_question], stop_words)).mean(axis=0)
    question_reshape = client_question_embed.reshape(1, 300)
    class_predict = np.argmax(model.predict(question_reshape))
    print(labels[class_predict])


prasanje = 'што се учи на академијата за дизајн'

class_predict(prasanje, stop_words_list)

