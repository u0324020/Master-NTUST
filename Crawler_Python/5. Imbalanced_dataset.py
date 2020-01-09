# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:13:04 2019

@author: Jane

做10Fold調參數跟baggaing的imbalance實驗

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
from sklearn.svm import SVC
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
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib


def Smote_upsampling(train_X, train_y):
    print(Counter(train_y))
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(train_X, train_y)
    print(Counter(y_smo))
    return (X_smo, y_smo)


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
    w = 1 - (Weight * 0.0001)
    print('Precision:', precision_score(y_test, predictions))
    print('Recall = TPR :', recall_score(y_test, predictions))
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((w + 1) * ((precision * recall) / (precision + recall))))
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
    w = 1 - (Weight * 0.0001)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((w + 1) * ((precision * recall) / (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def SVM(X, y, Weight):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=21)
    XG = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=1e-3,
             cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
    #class_weight={0: 1, 1: 88}
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="sklearn", lineno=196)
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
    w = 1 - (Weight * 0.0001)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((w + 1) * ((precision * recall) / (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def Lightgbm(X, y, Weight):
    # class_names = ['Malicious', 'Scam']
    class_names = ['Malicious', 'Scam']
    start = time.time()
    seed = 21
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    # Initialize CatBoostClassifier
    XG = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight={0: 1, 1: 88}, min_split_gain=0.,
                           min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
    XG.fit(X_train, y_train, eval_metric=[
           "error", "logloss"], eval_set=eval_set, verbose=True)
    print(XG)
    y_pred = XG.predict(X_test)
    print(XG.predict(X[0:1]))
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')
    w = 1 - (Weight * 0.0001)
    print('Precision:', precision_score(y_test, predictions, average='micro'))
    print('Recall = TPR :', recall_score(y_test, predictions, average='micro'))
    print('F1_score:', f1_score(y_test, predictions, average='micro'))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((w + 1) * ((precision * recall) / (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))
    return XG


def Bagging(X, y, Weight):  # when model overfitting # 不推薦，不如RandomForest # https://zhuanlan.zhihu.com/p/26683576
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
    w = 1 - (Weight * 0.0001)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((w + 1) * ((precision * recall) / (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def Voting(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=21)
    bag = VotingClassifier(estimators=[('XG_clf', XGBClassifier()), ('svm_clf', SVC()), ('dt_clf', DecisionTreeClassifier(
        random_state=666))], voting='hard', weights=None, n_jobs=None, flatten_transform=None)
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
    w = 1 - (Weight * 0.0001)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('Based on the Weight,We Tuning the parameter to:', w)
    print('F-Weight_score:', ((w + 1) * ((precision * recall) / (precision + recall))))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def Save_Model(model, path):
    path = path
    joblib.dump(model, path)
    print("===========Saving Model===========")


def Load_Model(X, y, path):
    path = path
    print("===========Load Model===========")
    model = joblib.load(path)
    print(model.predict(X[0:1]))


def For_LightGBM(X, y, Weight):
    # class_names = ['Malicious', 'Scam']
    class_names = ['Malicious', 'Scam']
    start = time.time()
    seed = 21
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    k_range = range(70, 100)
    L_scores = []
    for k in k_range:
        weight = "{0: 1, 1: " + str(k) + "}"
        print(type(weight))
        XG = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=weight, min_split_gain=0.,
                               min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
        XG.fit(X_train, y_train, eval_metric=[
               "error", "logloss"], eval_set=eval_set, verbose=True)
        y_pred = XG.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        L_scores.append(accuracy.mean())
        #scores = cross_val_score(XG, X, y, cv=10, scoring='accuracy')
        # L_scores.append(scores.mean())
    plt.plot(k_range, L_scores)
    plt.xlabel('Value of Weight for LightGBM')
    plt.ylabel('Cross Validated Accuracy')
    plt.show()


def For_Xgboost(X, y, Weight):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    seed = 21
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    k_range = range(1, 100)
    L_scores = []
    for k in k_range:
        print("=====================" + str(k) + "=====================")
        weight = "{0: 1, 1: " + str(k) + "}"
        print(type(weight))
        XG = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="binary:logistic", booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                           max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=k, base_score=0.5, random_state=0, seed=None, missing=None)
        # XG.fit(X_train, y_train, eval_metric=[
        #        "error", "logloss"], eval_set=eval_set, verbose=True)
        # y_pred = XG.predict(X_test)
        # predictions = [round(value) for value in y_pred]
        # accuracy = accuracy_score(y_test, predictions)
        # loss
        #loss = -cross_val_score(XG, X, y, cv=10, scoring='')
        # acc
        accuracy = cross_val_score(XG, X, y, cv=10, scoring='accuracy')
        L_scores.append(accuracy.mean())

        #scores = cross_val_score(XG, X, y, cv=10, scoring='accuracy')
        # L_scores.append(scores.mean())
    plt.plot(k_range, L_scores)
    plt.xlabel('Value of Weight for XGBoost')
    plt.ylabel('Cross Validated Accuracy')
    plt.savefig('/image/10Fold_Weight_XGBoost.png')
    plt.show()


if __name__ == '__main__':
    train = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/1219-Imbalanced-testing.csv", delimiter=",")
    min_max_scaler = preprocessing.MinMaxScaler()
    # np.random.shuffle(train)
    train_X = train[:, 0:44]
    train_y = train[:, 44]
    train_X = min_max_scaler.fit_transform(train_X)  # normalize
    counter_y = Counter(train_y)
    Weight = round((counter_y[0.0]) / (counter_y[1.0]))  # int
    # SMOTE
    #train_X, train_y = Smote_upsampling(train_X, train_y)
    #
    #For_LightGBM(train_X, train_y, Weight)
    For_Xgboost(train_X, train_y, Weight)
    #Xgboost(train_X, train_y, Weight)
    #Tree(train_X, train_y, Weight)
    #Bagging(train_X, train_y, Weight)
    #model = Lightgbm(train_X, train_y, Weight)
    #path = "C:/Users/Jane/Desktop/NTU/Scam/Code/Model/LightGBM.pkl"
    #Save_Model(model, path)
    #Load_Model(train_X, train_y, path)
    #SVM(train_X, train_y, Weight)
    #Voting(train_X, train_y)
