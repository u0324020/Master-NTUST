# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 00:13:04 2019

@author: Jane

特徵萃取 選擇 壓縮
http://www.ipshop.xyz/13707.html
"""
import pandas as pd
import pickle
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from numpy import loadtxt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import lightgbm as lgb


def Smote_upsampling(train_X, train_y):
    print(Counter(train_y))
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(train_X, train_y)
    print(Counter(y_smo))
    return (X_smo, y_smo)


def XGBoost_importance(X, y):
    print(Counter(y))
    model = XGBClassifier()
    model.fit(X, y)
    print("XGBoost Importance : ")
    plot_importance(model)
    pyplot.show()


def lightgbm_importance(X, y):
    print(Counter(y))
    gbm = lgb.LGBMRegressor(objective='regression',
                            learning_rate=0.3, n_estimators=50, num_threads=8)
    # Fit model
    gbm.fit(X, y)
    print("lightgbm Importance : ")
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(gbm, max_num_features=10)
    plt.title("Featurertances")
    plt.show()


def SelectK(X, y):
    # https://zhuanlan.zhihu.com/p/33199547
    selector = SelectKBest(score_func=f_classif, k=3)
    selector.fit(X, y)
    GetSupport = selector.get_support(True)
    TransX = selector.transform(X)
    print("ANOVA Importance : ")
    print(GetSupport)


def PrincipleComponentAnalysis(X, y):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    pca = decomposition.PCA(n_components=3)
    X = X[:2472]
    y = y[:2472]
    pca.fit(X)
    X = pca.transform(X)
    X = min_max_scaler.fit_transform(X)  # normalize

    for name, label in [('1', 0), ('2', 1), ('3', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    # ax.patch.set_facecolor('black')
    # y3 = np.arctan2(X[:, 0], X[:, 1], X[:, 2])  # ranbow漸層
    ax.scatter(X[1236:, 0], X[1236:, 1], X[1236:, 2], c='blue', cmap="brg", s=80,
               edgecolor='k', marker='o', label='Malicious')  # overlap
    ax.scatter(X[:1236, 0], X[:1236, 1], X[:1236, 2], c='red', cmap="brg", s=80, alpha=0.5,
               edgecolor='k', marker='x', label='Scam')
    # ax.view_init(elev=0, azim=0)  # 改變視角,azim沿着z軸旋轉，elev沿着y軸
    ax.set_xlabel('PCA-1')
    ax.set_ylabel('PCA-2')
    ax.set_zlabel('PCA-3')
    plt.legend()
    plt.show()


def plot_3D_3Features(X, y):
    X = min_max_scaler.fit_transform(X)
    X = X[:2472]
    X1 = X[:1236, 43]
    X2 = X[:1236:, 16]
    X3 = X[:1236:, 2]
    X4 = X[1236:, 43]
    X5 = X[1236:, 16]
    X6 = X[1236:, 2]
    y = y[:2472]
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    for name, label in [('1', 0), ('2', 1), ('3', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    # ax.patch.set_facecolor('black')
    # y3 = np.arctan2(X[:, 0], X[:, 1], X[:, 2])  # ranbow漸層
    ax.scatter(X1, X2, X3, c='red', cmap="brg", s=80,
               edgecolor='k', marker='x', label='Scam')  # overlap
    ax.scatter(X4, X5, X6, c='blue', cmap="brg", s=80, alpha=0.5,
               edgecolor='k', marker='o', label='Malicious')

    # ax.view_init(elev=0, azim=0)  # 改變視角,azim沿着z軸旋轉，elev沿着y軸
    ax.set_xlabel('Feature-43')
    ax.set_ylabel('Feature-16')
    ax.set_zlabel('Feature-2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/1219-Imbalanced-testing.csv", delimiter=",")
    min_max_scaler = preprocessing.MinMaxScaler()
    np.random.shuffle(train)
    train_X = train[:, 0:44]
    train_y = train[:, 44]
    train_X = min_max_scaler.fit_transform(train_X)  # normalize
    counter_y = Counter(train_y)
    Weight = round((counter_y[0.0]) / (counter_y[1.0]))  # int
    print(train_X.shape)
    # SMOTE
    # Smote_upsampling(train_X, train_y)
    # print(train_X.shape)
    # PCA(3D)
    # PrincipleComponentAnalysis(train_X, train_y)
    # Feature importance
    # XGBoost_importance(train_X, train_y)
    # lightgbm_importance(train_X, train_y)
    # SelectK(train_X, train_y)
    plot_3D_3Features(train_X, train_y)
