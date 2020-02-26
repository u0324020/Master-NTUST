# -*- coding: utf-8 -*-
"""
Created on WEN Feb 26 10:21:04 2020

@author: Jane

特徵重要度前十名跑迴圈 看Acc, Pre, Auc + plot

"""
from imblearn.over_sampling import SMOTE
from collections import Counter
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import warnings
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from matplotlib import pyplot as plt
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import itertools
from sklearn.metrics import roc_auc_score
import time
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn import preprocessing
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier


def lightgbm_10fold(X, y):
    print("-------Lightgbm_10fold---------")
    class_names = ['Malicious', 'Scam']
    start = time.time()
    gbm = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.3, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.,
                             min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
    y_pred = cross_val_predict(gbm, X, y, cv=10)
    scores = cross_val_score(gbm, X, y, cv=10, scoring='accuracy')
    print('標準差:', np.std(scores))
    print('10次', scores)
    print('平均', scores.mean())
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y, y_pred))
    print('Precision:', precision_score(y, y_pred))
    print('Recall:', recall_score(y, y_pred))
    print('F1_score:', f1_score(y, y_pred))
    print('AUC_score:', roc_auc_score(y, y_pred))
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)

    return(precision, auc, acc)
    fpr, tpr, threshold = roc_curve(y, y_pred)
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
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix, without normalization')
    plt.savefig('Xgboost(381).png', dpi=150)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('Xgboost-1(381).png', dpi=150)
    plt.show()


def lightgbm(X, y):
    # Test_scams.csv

    print("-------Lightgbm---------")
    class_names = ['Malicious', 'Benign']
    start = time.time()
    seed = 7
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Initialize CatBoostClassifier
    gbm = lgb.LGBMRegressor(objective='regression',
                            learning_rate=0.3, n_estimators=50, num_threads=8)
    # Fit model
    gbm.fit(X_train, y_train)
    print(gbm)
    y_pred = gbm.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print('F1_score:', f1_score(y_test, predictions))
    print('AUC_score:', roc_auc_score(y_test, predictions))
    # fpr, tpr, threshold = roc_curve(y_test, predictions)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # # plt.style.use('ggplot')
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('Catboost-Only-ROC.png', dpi=150)
    # cnf_matrix = confusion_matrix(y_test, predictions)
    # np.set_printoptions(precision=2)
    # # print(cnf_matrix)
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
    #                       title='Confusion matrix(Catboost), without normalization')
    # plt.savefig('Catboost-Only(381).png', dpi=150)
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix(Catboost)')
    # plt.savefig('Catboost-Only-1(381).png', dpi=150)
    # plt.show()


def lightgbm_importance(X, y):
    print(Counter(y))
    gbm = lgb.LGBMRegressor(objective='regression',
                            learning_rate=0.3, n_estimators=50, num_threads=8)
    # Fit model
    gbm.fit(X, y)
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(gbm, max_num_features=10)
    plt.title("Featurertances")
    plt.show()


if __name__ == '__main__':
    # train = loadtxt(
    #    "C:/Users/Jane/Desktop/NTU/Scam/Code/0126_Balanced_with_Benign.csv", delimiter=",")

    train = loadtxt(
        "Scams_Feature_3D.csv", delimiter=",")
    np.random.shuffle(train)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_X = train[:, 0:10]
    y = train[:, 10]
    X = min_max_scaler.fit_transform(train_X)  # normalize
    result_P = []
    result_AUC = []
    result_acc = []
    for i in range(1, 11):
        print("==================", i)
        P, AUC, acc = lightgbm_10fold(X[:, 0:i], y)
        result_P.append(P)
        result_AUC.append(AUC)
        result_acc.append(acc)
    print("========Precision==========")
    for k in result_P:
        print(k)
    print("========AUC==========")
    for j in result_AUC:
        print(j)
    print("========Acc==========")
    for n in result_acc:
        print(n)
    plt.figure("Loop precision")
    plt.xticks(range(0, 10))
    plt.plot(result_P, linewidth=1.0, label='Precision')
    plt.title('Precision Curve')
    plt.legend(loc="lower right")
    plt.xlabel('Dimension')
    plt.ylabel('Precision')
    plt.show()

    plt.figure("Loop AUC")
    plt.xticks(range(0, 10))
    plt.plot(result_AUC, linewidth=1.0, label='AUC')
    plt.legend(loc="lower right")
    plt.xlabel('Dimension')
    plt.ylabel('AUC Score')
    plt.title('AUC Score Curve')
    plt.show()

    plt.figure("Loop Accuracy")
    plt.xticks(range(0, 10))
    plt.plot(result_P, linewidth=1.0, label='Accuracy')
    plt.title('Accuracy Curve')
    plt.legend(loc="lower right")
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    plt.show()
