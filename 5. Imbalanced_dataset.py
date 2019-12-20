# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:13:04 2019

@author: Jane

做調參數跟baggaing的imbalance實驗

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
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import BaggingClassifier


def Xgboost(X, y, Weight):
    # class_names = ['Malicious', 'Scam']
    class_names = ['Malicious', 'Scam']

    start = time.time()
    seed = 21
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    XG = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="binary:logistic", booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                       max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
    XG.fit(X_train, y_train, eval_metric=[
           "error", "logloss"], eval_set=eval_set, verbose=True)
    print(XG)
    y_pred = XG.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    w = 1 - (Weight * 0.001)
    print('Precision:', precision_score(y_test, predictions))
    print('Recall = TPR :', recall_score(y_test, predictions))
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((math.pow(w, 2) + 1) * (precision * recall)) /
          ((math.pow(w, 2) * (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def Tree(X, y, Weight):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=21)
    XG = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None, presort=False)
    XG.fit(X_train, y_train)
    print(XG)
    y_pred = XG.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    w = 1 - (Weight * 0.001)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((math.pow(w, 2) + 1) * (precision * recall)) /
          ((math.pow(w, 2) * (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def Bagging(X, y, Weight):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=21)
    bag = BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
    bag.fit(X_train, y_train)
    y_pred = bag.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    w = 1 - (Weight * 0.001)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((math.pow(w, 2) + 1) * (precision * recall)) /
          ((math.pow(w, 2) * (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


if __name__ == '__main__':
    train = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/1219-Imbalanced-testing.csv", delimiter=",")
    np.random.shuffle(train)
    train_X = train[:, 0:44]
    train_y = train[:, 44]
    counter_y = Counter(train_y)
    Weight = round((counter_y[0.0]) / (counter_y[1.0]))  # int
    #Xgboost(train_X, train_y, Weight)
    #Tree(train_X, train_y, Weight)
    Bagging(train_X, train_y, Weight)
