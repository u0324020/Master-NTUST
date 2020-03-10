# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:24:04 2019

@author: Jane

1.將RawData分三類再畫CM:Xgboost
Refer : https://kknews.cc/code/4ogqqr3.html
2.Multi-class classification should using multi-class setting
Refer to scikit learn

ps. Multi label classification Refer : https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import scipy as sp
from sklearn.preprocessing import StandardScaler
from pylab import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import class_weight
from sklearn.utils import compute_sample_weight


sc = StandardScaler()
train = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/0304_3_Classify.csv", delimiter=",")
np.random.shuffle(train)
X = train[:, 0:44]
Input_data_y = train[:, 44]
Input_data_X = sc.fit_transform(X)
test = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/0304_3Class_testing_balanced.csv", delimiter=",")
X = test[:, 0:44]
test_y = test[:, 44]
test_X = sc.fit_transform(X)
sample_weight = compute_sample_weight('balanced', Input_data_y)
print("Weight = ", sample_weight)


def data():
    global X_test, Y_test, X_train, Y_train, Input_data1_predict

    X_train = Input_data_X
    Y_train = Input_data_y
    # weight = "{0: 10, 1: 1, 2: 1}" notwork
    X_test = test_X
    Y_test = test_y
    # clf = lgb.LGBMClassifier(boosting='gbdt', bagging_fraction=0.8, bagging_freq=5, num_leaves=300, max_depth=10, learning_rate=0.01, n_estimators=100, subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.,
    #                          class_weight=weight,min_child_weight=1e-3, min_child_samples=20, verbose=-1, min_data_in_leaf=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
    # clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective='binary', class_weight='balanced', min_split_gain=0.,
    #                          min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
    clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="binary:logistic", booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                            class_weight=None, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
    # clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.,
    #                          min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
    # clf = RandomForestClassifier(n_estimators='warn', criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
    tStart = time.time()
    clf.fit(X_train, Y_train)
    # Input_data1_predict = cross_val_predict(clf, X_train, Y_train, cv=10)
    tEnd = time.time()
    runtime = tEnd - tStart
    print("Time taken: ", runtime, "seconds.")
    Input_data1_predict = clf.predict(X_test)

    ss = np.array(X_train)
    print(ss.shape)


def confutionmatrix():
    global labels, confmat, accuracy

    labels = [1, 0, 2]
    confmat = np.zeros(shape=(len(labels), len(labels)))
    confmat = confusion_matrix(Y_test, Input_data1_predict, labels)
    print(confmat)

    accuracy = accuracy_score(Y_test, Input_data1_predict)

    print('accuracy:', accuracy, end='   ')
    print('Precision:', precision_score(
        Y_test, Input_data1_predict, average='macro'))
    print('Recall:', recall_score(Y_test, Input_data1_predict, average='macro'))
    print('F1_score:', f1_score(Y_test, Input_data1_predict, average='macro'))


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
    labels_name = ["TSS", "Malicious", "Benign"]
    plt.xticks(x_title, labels_name)
    plt.yticks(y_title, labels_name)
    plt.ylabel("Actual", fontsize=14)
    plt.title('Predict', fontsize=14)
    plt.show()


if __name__ == "__main__":
    data()
    confutionmatrix()
    plot1()
    # Acc = accuracy_score(X_train, Input_data1_predict_val)
    # f1 = f1_score(X_train, Input_data1_predict_val, average='macro')
    # p = precision_score(X_train, Input_data1_predict_val, average='macro')
    # r = recall_score(X_train, Input_data1_predict_val, average='macro')
    # print('Val_accuracy:', Acc)
    # print("Val_F1_score = ", f1)
    # print("Val_Precision = ", p)
    # print("Val_Recall = ", r)
