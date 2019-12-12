# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:24:04 2019

@author: Jane

將RawData分三類再畫CM:Xgboost

"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import scipy as sp
from sklearn.preprocessing import StandardScaler
from pylab import *
import xgboost as xgb
from sklearn.model_selection import train_test_split

tStart = time.time()


sc = StandardScaler()
train = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/1212_3kind_cla.csv", delimiter=",")
np.random.shuffle(train)
X = train[:, 0:44]
Input_data1 = train[:, 44]
Input_data = sc.fit_transform(X)


def data():
    global X_test, Y_test, X_train, Y_train, Input_data1_predict

    X_train, X_test, Y_train, Y_test = train_test_split(
        Input_data, Input_data1, test_size=0.3, random_state=16)

    clf = xgb.XGBClassifier(max_depth=1, learning_rate=0.01, n_estimators=10,
                            silent=True, objective='binary:logistic', n_jobs=6)
    clf.fit(X_train, Y_train)
    Input_data1_predict = clf.predict(X_test)

    ss = np.array(X_train)
    print(ss.shape)


def confutionmatrix():
    global labels, confmat, accuracy

    labels = [0, 2, 1]
    confmat = np.zeros(shape=(len(labels), len(labels)))
    confmat = confusion_matrix(Y_test, Input_data1_predict, labels)
    print(confmat)

    accuracy = accuracy_score(Y_test, Input_data1_predict)

    print('accuracy:', accuracy, end='   ')


def plot1():

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=5)

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    x_title = arange(len(labels))
    y_title = arange(len(labels))
    labels_name = ["TSS", "Benign", "Malicious"]
    plt.xticks(x_title, labels_name)
    plt.yticks(y_title, labels_name)
    plt.ylabel("Actual", fontsize=14)
    plt.title('Predict', fontsize=14)
    plt.savefig('accuracy.png', dpi=400)
    plt.show()


if __name__ == "__main__":
    data()
    confutionmatrix()
    plot1()

tEnd = time.time()
runtime = tEnd - tStart
print("Time taken: ", runtime, "seconds.")
