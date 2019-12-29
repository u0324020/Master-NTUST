# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:13:04 2019

@author: Jane

兩個類別的每個特徵平均值算出來 以重疊的方式話長條圖
把二分類的每個特徵之資料分布圖畫成長條圖
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
from sklearn import preprocessing
import seaborn as sns

train = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/1219-Imbalanced-testing.csv", delimiter=",")
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)  # normalize
X1 = train[:1236, 0:44]  # scam data 1236
X0 = train[1236:, 0:44]  # malicious data

feature_names = [i for i in range(44)]
X1_feature_Mean = []
X0_feature_Mean = []

for i in range(44):  # Feature number
    X1_feature_Mean.append(np.mean(X1[:, i]))  # 算平均值
    X0_feature_Mean.append(np.mean(X0[:, i]))
# PLOT
plt.figure()
n = 44
X = np.arange(n) + 1
plt.bar(X, X0_feature_Mean, color="orange", label="Malicious")
plt.bar(X, X1_feature_Mean, color="dodgerblue", alpha=0.7, label="Scams")
plt.legend(loc="upper right")
plt.title('44 Feature Mean', fontsize=22)
plt.xlabel("Feature", fontdict=None, labelpad=None)
plt.ylabel("Mean", fontdict=None, labelpad=None)
plt.show()
