# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:13:04 2019

@author: Jane

把每個特徵的平均值算出來 再分為1,0兩類算平均 最後以長條圖或趨勢圖畫出分布
先篩選幾個特徵再畫圖

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
Full_Avg = []
TSS_Avg = []
Malicious_Avg = []
train = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/1206-importance.csv", delimiter=",")
# np.random.shuffle(train)
X = train[:, 0:44]  # data
y = train[:, 44]  # label
shape = X.shape
shape = shape[0]
labels = [1, 0]  # label values
# print(X.mean())# average

for i in range(44):
    Full_Avg.append(round((X[i].mean()), 2))
    TSS_Avg.append(round((X[:924][i].mean()), 2))
    Malicious_Avg.append(round((X[925:][i].mean()), 2))
    print(str(i) + "---->")
    print(Full_Avg[i])
