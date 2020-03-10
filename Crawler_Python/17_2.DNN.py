# -*- coding: utf-8 -*-
"""
Created on WEN Feb 26 10:21:04 2020

@author: Jane

Based on DNN to classify Scams and Malicious

"""
import tensorflow as tf
import itertools
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import seaborn as sns
import sklearn
from keras import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import time
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)  # cm = number
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def DNN_model(X, y, class_weights):
    model = Sequential(layers=None, name=None)
    model.add(Dense(64, input_dim=44,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3, noise_shape=None, seed=21))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
                epsilon=None, decay=0., amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_his = model.fit(
        X, y, epochs=258, validation_split=0.3, batch_size=32, shuffle=True, verbose=1, class_weight={0: 1, 1: 2.4})  # class_weight='auto',{0: 1, 1: 100} # epochs = training sample/batch_size
    return (model, train_his)
    # 8263


if __name__ == '__main__':
    shuffle_train_Malicious = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/shuffle_train_Malicious_0.97.csv", delimiter=",")
    shuffle_train_Scams = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/shuffle_train_Scams_0.97.csv", delimiter=",")
    #=====merge csv to train=====
    #train = shuffle_train_Malicious + shuffle_train_Scams
    train = np.concatenate((shuffle_train_Malicious, shuffle_train_Scams))
    # train = loadtxt(
    #     "C:/Users/Jane/Desktop/NTU/108@Super/Code/1.4w(28040).csv", delimiter=",")
    np.random.shuffle(train)
    X = train[:, 0:44]  # 1:2500
    y = train[:, 44]
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    # testing data
    test = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/0225_testing_balanced.csv", delimiter=",")
    # np.random.shuffle(train)
    testX = test[:, 0:44]  # 1:2500
    y_test_data = test[:, 44]
    class_names = ['Malicious', 'Scam']
    # # Reshape and normalize training data
    # trainX = X[:, :].reshape(
    #     X.shape[0], 44, 1, 1).astype('float32')
    # X_train = trainX
    # y_train = np_utils.to_categorical(y)
    # # Reshape and normalize test data
    # testX = testX[:, :].reshape(testX.shape[0], 44, 1, 1).astype('float32')
    # X_test = testX
    # y_test = y_test_data
    # y_test = np_utils.to_categorical(y_test)  # num_classes label約有幾個
    # calculate the class weight
    #=================train=====================
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(y), y)
    print(class_weights)
    model, train_history = DNN_model(X, y, class_weight)
    # plot_CM(train_history, 'acc', 'val_acc')
    # plot_CM(train_history, 'loss', 'val_loss')
    scores = model.evaluate(
        X, y, verbose=0)
    print("")
    print('LOSS= ', scores[0])
    print('Accuracy=', scores[1])
    # print('F1-Score= ', f1_score)
    # print('Precision=', precision)
    # print('Recall=', recall)
    model.save('DNN_model.h5')
    print("=========Test===========")
    test_model = tf.contrib.keras.models.load_model('DNN_model.h5')
    scores2 = test_model.evaluate(X, y)
    print("")
    print('LOSS= ', scores2[0])
    print('Accuracy=', scores2[1])
    scores_test = test_model.evaluate(testX, y_test_data)
    print("")
    print('Test LOSS= ', scores_test[0])
    print('Test Accuracy=', scores_test[1])
    ST = time.time()
    y_pred = test_model.predict(testX).ravel()
    threshold = 0.5
    y_pred_binarized = np.where(y_pred > threshold, 1, 0)
    ET = time.time()
    print('TIME:', ET - ST)
    print('Accuracy:', accuracy_score(y_test_data, y_pred_binarized))
    print('Precision:', precision_score(y_test_data, y_pred_binarized))
    print('Recall:', recall_score(y_test_data, y_pred_binarized))
    AP = average_precision_score(y_test_data, y_pred_binarized)
    print('Average precision-recall score:', AP)
    print('F1_score:', f1_score(y_test_data, y_pred_binarized))
    print('AUC_score:', roc_auc_score(y_test_data, y_pred_binarized))
    cm = confusion_matrix(y_test_data, y_pred_binarized)
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(DNN)')
    plt.show()
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(DNN)')
    plt.show()
    fpr, tpr, threshold = roc_curve(y_test_data, y_pred_binarized)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    precision, recall, thresholds = precision_recall_curve(
        y_test_data, y_pred_binarized)
    plt.figure("P-R Curve")
    plt.plot(recall, precision, color='red', linewidth=1.0,
             linestyle='--', label='P-R Curve (AP = %0.2f)' % AP)
    plt.title('Precision/Recall Curve')
    plt.legend(loc="lower right")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    # # predict probabilities for test set
    # yhat_probs = test_model.predict(testX, verbose=0)
    # # predict crisp classes for test set
    # yhat_classes = test_model.predict_classes(testX, verbose=0)
    # # reduce to 1d array
    # yhat_probs = yhat_probs[:, 0]
    # yhat_classes = yhat_classes[:, 0]

    # # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(testy, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # # precision tp / (tp + fp)
    # precision = precision_score(testy, yhat_classes)
    # print('Precision: %f' % precision)
    # # recall: tp / (tp + fn)
    # recall = recall_score(testy, yhat_classes)
    # print('Recall: %f' % recall)
    # # f1: 2 tp / (2 tp + fp + fn)
    # f1 = f1_score(testy, yhat_classes)
    # print('F1 score: %f' % f1)

    # # kappa
    # kappa = cohen_kappa_score(testy, yhat_classes)
    # print('Cohens kappa: %f' % kappa)
    # # ROC AUC
    # auc = roc_auc_score(testy, yhat_probs)
    # print('ROC AUC: %f' % auc)
    # # confusion matrix
    # matrix = confusion_matrix(testy, yhat_classes)
    # print(matrix)


# Accuracy= 0.9943260793636085
