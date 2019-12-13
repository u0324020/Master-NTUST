# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:13:04 2019

@author: Jane

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

train = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/1206-importance.csv", delimiter=",")
# np.random.shuffle(train)
X = train[:, 0:44]  # data
y = train[:, 44]  # label
# print(X.shape)
labels = [1, 0]  # label values
feature_names = [i for i in range(44)]

for i in range(44):  # Feature number
    TSS_count = pd.value_counts(X[:924][i])
    TSS_count = TSS_count.sort_index()
    print("TSS Feature %s -->" % str(i + 1))
    print(TSS_count)

    malicious_count = pd.value_counts(X[925:][i])
    malicious_count = malicious_count.sort_index()
    print("Malicious Feature %s -->" % str(i + 1))
    print(malicious_count)

    fig = plt.figure(num=i, figsize=(12, 9))
    plt.xlabel(str(i + 1) + "Feature Value")
    plt.ylabel("Occurrences Times")
    plt.title(str(i + 1) + " Feature Distribution",
              fontdict=None, loc='center', pad=None)
    bins = [j for j in range(100)]
    TSS_y = []
    malicious_y = []

    for k in range(100):
        if k in list(TSS_count.index):
            TSS_y.append(TSS_count.get(k))
        else:
            TSS_y.append(0)

    for n in range(100):
        if n in list(malicious_count.index):
            malicious_y.append(malicious_count.get(n))
        else:
            malicious_y.append(0)

    plt.bar(bins, TSS_y, width=0.5, alpha=1,
            color="orange", label="TSS")
    plt.bar(bins, malicious_y, width=0.5, alpha=0.5,
            color="blue", label="malicious")
    plt.legend(loc="upper right")
    # plt.show()
    path = 'C:/Users/Jane/Desktop/NTU/Scam/Code/image/FeatureDistribution'
    filename = 'Analysis_' + str(i + 1) + 'Feature.png'
    plt.savefig(path + filename)
