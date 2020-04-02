# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:13:04 2019

@author: Jane

1. 所有基礎實驗
2. SMOTE比較

Note: SMOTE後的Pression算不出來
##5833:2430
"""
from sklearn.externals import joblib
import pandas as pd
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
from sklearn.metrics import mean_squared_error
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
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import learning_curve


def Smote_upsampling(train_X, train_y):
    print("------SMOTE------")
    print(Counter(train_y))
    Stime = time.time()
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(train_X, train_y)
    Etime = time.time()
    print(Counter(y_smo))
    print('TIME:', Etime - Stime)
    return (X_smo, y_smo)


def SMOTEENN_upsampling(train_X, train_y):
    print("------SMOTEENN------")
    print(Counter(train_y))
    Stime = time.time()
    smo = SMOTEENN(random_state=42)
    X_smo, y_smo = smo.fit_resample(train_X, train_y)
    Etime = time.time()
    print(Counter(y_smo))
    print('TIME:', Etime - Stime)
    return (X_smo, y_smo)


def SMOTETomek_upsampling(train_X, train_y):
    print("------SMOTETomek------")
    print(Counter(train_y))
    Stime = time.time()
    smo = SMOTETomek(random_state=42)
    X_smo, y_smo = smo.fit_resample(train_X, train_y)
    Etime = time.time()
    print(Counter(y_smo))
    print('TIME:', Etime - Stime)
    return (X_smo, y_smo)

# def Smote_upsampling(train_X, train_y):
#     print(Counter(train_y))
#     # for i in range(1, 43):
#     #     plt_1 = plt.scatter(train_X[:10556, i - 1], train_X[:10556, i], c='b',
#     #                         marker='o', s=40, cmap=plt.cm.Spectral)  # x,y 特徵
#     #     plt_2 = plt.scatter(train_X[10556:, i - 1], train_X[10556:, i], c='r',
#     #                         marker='x', s=50, cmap=plt.cm.Spectral)
#     #     plt.legend(handles=[plt_1, plt_2], loc='upper left')
#     #     plt.savefig('C:/Users/Jane/Desktop/NTU/Scam/Code/image/2Befor_Smote_F' +
#     #                 str(i) + '.png', dpi=150)
#     # 顏色: m, c, r, b
#     # 點: o, x, +, *, v
#     # 透明度 alpha(0~1)

#     smo = SMOTE(random_state=42)
#     X_smo, y_smo = smo.fit_sample(train_X, train_y)
#     print(Counter(y_smo))
#     # for i in range(1, 43):
#     #     plt_1 = plt.scatter(X_smo[:10556, i - 1], X_smo[:10556, i], c='b',
#     #                         marker='o', s=40, cmap=plt.cm.Spectral)  # x,y 特徵
#     #     plt_2 = plt.scatter(X_smo[10556:, i - 1], X_smo[10556:, i], c='r',
#     #                         marker='x', s=50, cmap=plt.cm.Spectral)
#     #     plt.legend(handles=[plt_1, plt_2], loc='upper left')
#     #     plt.savefig('C:/Users/Jane/Desktop/NTU/Scam/Code/image/2After_Smote_F' +
#     #                 str(i) + '.png', dpi=150)
#     return (X_smo, y_smo)
def SVM(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed)
    gbm = SVC(C=1.0, kernel='linear', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=1e-3,
             cache_size=200, class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
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
    AP =average_precision_score(y, y_pred)
    print('Average precision-recall score:',AP)
    print('F1_score:', f1_score(y, y_pred))
    print('AUC_score:', roc_auc_score(y, y_pred))

    return gbm

def RandomForest(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=7)
    XG = RandomForestClassifier(n_estimators='warn', criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto",
                                max_leaf_nodes=20, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
    XG.fit(X_train, y_train)
    print(XG)
    y_pred = XG.predict(X_test)
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
    # plt.savefig('Xgboost-Only-ROC.png', dpi=150)
    # cnf_matrix = confusion_matrix(y_test, predictions)
    # np.set_printoptions(precision=2)
    # # print(cnf_matrix)
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
    #                       title='Confusion matrix(Xgboost), without normalization')
    # plt.savefig('Xgboost-Only(381).png', dpi=150)
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix(Xgboost)')
    # plt.savefig('Xgboost-Only-1(381).png', dpi=150)
    # plt.show()
    A= time.time()
    train_sizes, train_loss, test_loss = learning_curve(
    XG, X, y, cv=10, scoring='brier_score_loss', #https://www.studyai.cn/modules/model_evaluation.html
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]) #做5次的10Flod
    
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    B = time.time()
    print('TIME:', A - B)
    plt.plot(train_sizes, train_loss_mean,color="royalblue",
         label="Training")
    plt.plot(train_sizes, test_loss_mean,color="orange",
            label="Cross-validation")

    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
    return XG


def Xgboost(X, y):
    #class_names = ['Malicious', 'Scam']
    class_names = ['Malicious', 'Scam']

    start = time.time()
    seed = 7
    test_size = 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    XG = XGBClassifier()
    XG.fit(X_train, y_train, eval_metric=[
           "auc", "logloss"], eval_set=eval_set, verbose=True)
    print(XG)
    y_pred = XG.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    results = XG.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.xlabel('Epochs')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()

    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Auc')
    pyplot.xlabel('Epochs')
    pyplot.title('XGBoost Classification Auc')
    pyplot.show()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print('F1_score:', f1_score(y_test, predictions))
    print('AUC_score:', roc_auc_score(y_test, predictions))
    # fpr, tpr, threshold = roc_curve(y_test, predictions)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
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
    # plt.savefig('Xgboost-Only-ROC.png', dpi=150)
    # cnf_matrix = confusion_matrix(y_test, predictions)
    # np.set_printoptions(precision=2)
    # # print(cnf_matrix)
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
    #                       title='Confusion matrix(Xgboost), without normalization')
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix(Xgboost)')
    # plt.show()
    A = time.time()
    train_sizes, train_loss, test_loss = learning_curve(
    XG, X, y, cv=10, scoring='f1', #https://www.studyai.cn/modules/model_evaluation.html
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]) #做5次的10Flod
    
    train_loss_mean = np.mean(train_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)
    B = time.time()
    print('TIME:', A - B)
    plt.plot(train_sizes, train_loss_mean,color="royalblue",
         label="Training")
    plt.plot(train_sizes, test_loss_mean,color="orange",
            label="Cross-validation")

    plt.xlabel("Training examples")
    plt.ylabel("F1_score")
    plt.legend(loc="best")
    plt.show()
    return XG


def Xgboost_10fold(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    XG = XGBClassifier()
    y_pred = cross_val_predict(XG, X, y, cv=10)
    scores = cross_val_score(XG, X, y, cv=10, scoring='accuracy')
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
    # fpr, tpr, threshold = roc_curve(y, y_pred)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
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
    # plt.savefig('Xgboost-ROC.png', dpi=150)
    # cnf_matrix = confusion_matrix(y, y_pred)
    # np.set_printoptions(precision=2)
    # # print(cnf_matrix)
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
    #                       title='Confusion matrix(Xgboost), without normalization')
    # plt.savefig('Xgboost(381).png', dpi=150)
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix(Xgboost)')
    # plt.savefig('Xgboost-1(381).png', dpi=150)
    # plt.show()


def Tree(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=7)
    XG = DecisionTreeClassifier(criterion='gini', max_depth=10)
    XG.fit(X_train, y_train)
    print(XG)
    y_pred = XG.predict(X_test)
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
    fpr, tpr, threshold = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
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
    plt.savefig('Xgboost-Only-ROC.png', dpi=150)
    cnf_matrix = confusion_matrix(y_test, predictions)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(Xgboost), without normalization')
    plt.savefig('Xgboost-Only(381).png', dpi=150)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(Xgboost)')
    plt.savefig('Xgboost-Only-1(381).png', dpi=150)
    plt.show()


def Tree_10flod(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    clf = RandomForestClassifier(n_estimators='warn', criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto",
                                 max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
    y_pred = cross_val_predict(clf, X, y, cv=10)
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
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
    plt.savefig('Tree-ROC.png', dpi=150)
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(Tree), without normalization')
    plt.savefig('Tree(381).png', dpi=150)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(Tree)')
    plt.savefig('Tree-1(381).png', dpi=150)
    # plt.colorbar()
    plt.show()
    return clf


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


def XGBoost_importance(X, y):
    print(Counter(y))
    model = XGBClassifier()
    model.fit(X, y)
    plot_importance(model, max_num_features=42)
    #pyplot.savefig('feature_importance_Xgboost(992).png', dpi=150)
    pyplot.show()


def ANOVA(X, y):
    sc = StandardScaler()
    Input_data_nosc = X
    Input_data1 = y
    Input_data = sc.fit(Input_data_nosc)

    b = np.arange(1, Input_data.shape[1] + 1)

    f_classif_, pval = f_classif(Input_data, Input_data1)
    print(f_classif_)
    print(Input_data.shape)
    plt.bar(b, f_classif_)
    plt.title('Anova_100', fontsize=18)
    plt.savefig('Anova.png', dpi=150)
    plt.show()


def Catboost_10fold(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    model = CatBoostClassifier(
        iterations=2,
        learning_rate=1,
        depth=2, loss_function='Logloss')
    y_pred = cross_val_predict(model, X, y, cv=10)
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
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
    plt.savefig('Xgboost-ROC.png', dpi=150)
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(Xgboost), without normalization')
    plt.savefig('Xgboost(381).png', dpi=150)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(Xgboost)')
    plt.savefig('Xgboost-1(381).png', dpi=150)
    plt.show()


def Catboost(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    seed = 7
    test_size = 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(
        iterations=2,
        learning_rate=1,
        depth=2, loss_function='Logloss')
    # Fit model
    model.fit(X_train, y_train)
    print(model)
    y_pred = model.predict(X_test)
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
    fpr, tpr, threshold = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
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
    plt.savefig('Catboost-Only-ROC.png', dpi=150)
    cnf_matrix = confusion_matrix(y_test, predictions)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(Catboost), without normalization')
    plt.savefig('Catboost-Only(381).png', dpi=150)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(Catboost)')
    plt.savefig('Catboost-Only-1(381).png', dpi=150)
    plt.show()


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
    plt.savefig('Xgboost-ROC.png', dpi=150)
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(Xgboost), without normalization')
    plt.savefig('Xgboost(381).png', dpi=150)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(Xgboost)')
    plt.savefig('Xgboost-1(381).png', dpi=150)
    plt.show()


def lightgbm(X_train, y_train):
    # Test_scams.csv
    train = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/Test_scams.csv", delimiter=",")
    np.random.shuffle(train)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_test = train[:, 0:44]
    y_test = train[:, 44]
    X_test = min_max_scaler.fit_transform(X_test)

    print("-------Lightgbm---------")
    class_names = ['Malicious', 'Scam']
    start = time.time()
    # seed = 7
    # test_size = 0.3
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed)
    # eval_set = [(X_train, y_train), (X_test, y_test)]

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
    # lrarning curve

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
    gbm = lgb.LGBMClassifier(objective= 'binary',learning_rate=0.1)
    #gbm = lgb.LGBMRegressor(objective='regression',
    #                        learning_rate=0.3, n_estimators=50, num_threads=8)
    # Fit model
    gbm.fit(X, y)
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(gbm, max_num_features=42)
    plt.title("LightGBM Featurer Importance")
    plt.show()


def AdaBoost(X, y):  # 模型太弱就用boost
    class_names = ['Malicious', 'Scam']
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=21)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20,
                                                    min_samples_leaf=5), algorithm="SAMME", n_estimators=200, learning_rate=0.8)
    bdt.fit(X_train, y_train)
    y_pred = bdt.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print('Precision:', precision)
    print('Recall = TPR:', recall)
    print('F1_score:', f1_score(y_test, predictions))
    print('AUC_score:', roc_auc_score(y_test, predictions))


def Weight_LightGBM(X, y, W):
    seed = 21
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        train_X, train_y, test_size=test_size, random_state=seed)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # print("=======Org=======")
    # gbm1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0.,
    #                           min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split')
    # # Fit model

    # gbm1.fit(X_train, y_train)
    # print(gbm1)
    # start = time.time()
    # y_pred = gbm1.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    # end = time.time()
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # print('TIME:', end - start)
    # print('Accuracy:', accuracy_score(y_test, predictions))
    # print('Precision:', precision_score(y_test, predictions))
    # print('Recall:', recall_score(y_test, predictions))
    # print('Average precision-recall score:',
    #       average_precision_score(y_test, predictions))
    # print('F1_score:', f1_score(y_test, predictions))
    # print('AUC_score:', roc_auc_score(y_test, predictions))
    # precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    # plt.figure("P-R Curve")
    # plt.title('Precision/Recall Curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.plot(recall, precision)
    # plt.show()

    print("=======ClassWeight=======")
    #print(W)
    gbm = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=30,max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0., min_child_weight=1e-3,
                             min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split', is_unbalance=True)
    # Fit model
    gbm.fit(X_train, y_train, eval_set=eval_set)
    start = time.time()
    y_pred = gbm.predict(X_test)
    predictions = [round(value) for value in y_pred]
    end = time.time()
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print('Average precision-recall score:',
          average_precision_score(y_test, predictions))
    print('F1_score:', f1_score(y_test, predictions))
    print('AUC_score:', roc_auc_score(y_test, predictions))
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.show()
    
    A = time.time()
    train_sizes, train_loss, test_loss = learning_curve(
    gbm, X, y, cv=10, scoring='brier_score_loss', #https://www.studyai.cn/modules/model_evaluation.html
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]) #做5次的10Flod
    
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    B = time.time()
    print('TIME:', A - B)
    plt.plot(train_sizes, train_loss_mean,color="royalblue",
         label="Training")
    plt.plot(train_sizes, test_loss_mean,color="orange",
            label="Cross-validation")

    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
    print('mean of loss:'+str(np.mean(test_loss_mean)))
    #acc
    A = time.time()
    train_sizes, train_loss, test_loss = learning_curve(
    gbm, X, y, cv=10, scoring='accuracy', #https://www.studyai.cn/modules/model_evaluation.html
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]) #做5次的10Flod
    
    train_loss_mean = np.mean(train_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)
    B = time.time()
    print('TIME:', A - B)
    plt.plot(train_sizes, train_loss_mean,color="royalblue",
         label="Training")
    plt.plot(train_sizes, test_loss_mean,color="orange",
            label="Cross-validation")

    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()
    print('mean of Acc:'+str(np.mean(test_loss_mean)))
    return gbm


def Weight_lightgbm_10fold(X, y, W):
    print("-------Lightgbm_10fold(RF_0.01)---------")
    class_names = ['Malicious','Scam']
    start = time.time()
    gbm = lgb.LGBMClassifier(boosting='rf',bagging_fraction = 0.8,bagging_freq=5, num_leaves=300, max_depth=10, learning_rate=0.01, n_estimators=100, subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0., min_child_weight=1e-3,
                             min_child_samples=20,verbose=-1,min_data_in_leaf=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split', scale_pos_weight=W)
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
    AP =average_precision_score(y, y_pred)
    print('Average precision-recall score:',AP)
    print('F1_score:', f1_score(y, y_pred))
    print('AUC_score:', roc_auc_score(y, y_pred))
    #=====plot===
    # fpr, tpr, threshold = roc_curve(y, y_pred)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
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
    # cnf_matrix = confusion_matrix(y, y_pred)
    # np.set_printoptions(precision=2)
    # # print(cnf_matrix)
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
    #                       title='Confusion matrix(LightGBM(gbdt)_10Fold)')
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix(LightGBM(gbdt)_10Fold)')
    # # plt.savefig('Xgboost-1(381).png', dpi=150)
    # plt.show()
    # # PR curve
    # precision, recall, thresholds = precision_recall_curve(y, y_pred)
    # plt.figure("P-R Curve")
    # plt.plot(recall, precision, color='red', linewidth=1.0,
    #          linestyle='--', label='P-R Curve (AP = %0.2f)' % AP)
    # plt.title('Precision/Recall Curve')
    # plt.legend(loc="lower right")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.show()

    return gbm

def Test_light_rf(clf,X_test,y_test,X_train, y_train):
    class_names = ['Malicious','Scam']
    print("Test....")
    XG = clf
    XG.fit(X_train, y_train)
    start = time.time()
    y_pred = XG.predict(X_test)
    print(XG.predict(X_train[0:1]))
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    print('TIME:', end - start)
    print('Accuracy:', accuracy_score(y_test, predictions))
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')
    AP =average_precision_score(y_test, predictions, average='micro')
    print('Precision:', precision)
    print('Recall = TPR :', recall)
    print('F1_score:', f1_score(y_test, predictions, average='micro'))
    print('AUC_score:', roc_auc_score(y_test, predictions))
    #=========PLOT==========
    fpr, tpr, threshold = roc_curve(y_test, predictions)
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
    cnf_matrix = confusion_matrix(y_test, predictions)
    np.set_printoptions(precision=2)
    # print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Oranges,
                          title='Confusion matrix(LightGBM_10Fold)')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix(LightGBM_10Fold)')
    # plt.savefig('Xgboost-1(381).png', dpi=150)
    plt.show()
    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    plt.figure("P-R Curve")
    plt.plot(recall, precision, color='red', linewidth=1.0,
             linestyle='--', label='P-R Curve (AP = %0.2f)' % AP)
    plt.title('Precision/Recall Curve')
    plt.legend(loc="lower right")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

# boosting_type='gbdt', num_leaves=31, metric={'mean_absolute_error'},max_depth=-1,
#  learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective='binary', class_weight=None, min_split_gain=0., min_child_weight=1e-3,
# min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., 
# reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split', is_unbalance=True
    

def plt_lightGBM(X, y,test_X,test_y):
    X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.3, random_state=12345)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # to record eval results for plotting
    evals_result = {}
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc','binary_logloss','binary_error'},#binary_logloss #binary_error auc
    'num_leaves': 30,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf':4,
    'learning_rate':0.1,
    #'n_estimators':1000,
     #'min_sum_hessian_in_leaf': 5,
    'is_unbalance':True,
    #'verbose':10
    }

    print('Start training...')

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=[lgb_train, lgb_valid],
                    evals_result=evals_result,
                    verbose_eval=10,
                    early_stopping_rounds=50)
    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    # eval rmse
    print('\nThe rmse of prediction is:', mean_squared_error(y_valid, y_pred) ** 0.5)
    print('Plot metrics during training...')
    ax = lgb.plot_metric(evals_result, metric='binary_error')###
    plt.show()
    print('Feature重要性排序...')
    ax = lgb.plot_importance(gbm)
    plt.show()

    # print('\nPredicting test set...')
    # y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration)
    
    # # y_pred = model.predict(dtest)
    # print("The rmse of loaded model's prediction is:",mean_squared_error(test_y, y_pred) ** 0.5)

    # print("Finished.")

    # from sklearn.model_selection import GridSearchCV
    # estimator = lgb.LGBMRegressor()

    # # get possible parameters
    # estimator.get_params().keys()

    # # fill parameters ad libitum
    # param_grid = {
    # 'num_leaves': [20, 30],    
    # 'learning_rate': [0.01, 0.1],
    # #     'n_estimators': [],
    # #     'colsample_bytree' :[],
    # #     'min_split_gain' :[],
    # #     'subsample_for_bin' :[],
    # #     'max_depth' :[],
    # #     'subsample' :[], 
    # #     'reg_alpha' :[], 
    # #     'max_drop' :[], 
    # #     'gaussian_eta' :[], 
    # #     'drop_rate' :[], 
    # #     'silent' :[], 
    # #     'boosting_type' :[], 
    # #     'min_child_weight' :[], 
    # #     'skip_drop' :[], 
    # #     'learning_rate' :[], 
    # #     'fair_c' :[], 
    # #     'seed' :[], 
    # #     'poisson_max_delta_step' :[], 
    # #     'subsample_freq' :[], 
    # #     'max_bin' :[], 
    # #     'n_estimators' :[], 
    # #     'nthread' :[], 
    # #     'min_child_samples' :[], 
    # #     'huber_delta' :[], 
    # #     'use_missing' :[], 
    # #     'uniform_drop' :[], 
    # #     'reg_lambda' :[], 
    # #     'xgboost_dart_mode' :[], 
    # #     'objective'
    # }


    # gbm = GridSearchCV(estimator, param_grid)

    # gbm.fit(X_train, y_train)

    # # list them
    # print('Best parameters found by grid search are:', gbm.best_params_)
if __name__ == '__main__':
    # train = loadtxt(
    #     "C:/Users/Jane/Desktop/NTU/Scam/Code/0319_imbalance_10w,0.25.csv", delimiter=",")
    #===========Shuffle Training Data=============
    # train_Scam = loadtxt(
    #     "C:/Users/Jane/Desktop/NTU/Scam/Code/0224_Imbalanced_S2500.csv", delimiter=",")
    # train_Malicious = loadtxt(
    #     "C:/Users/Jane/Desktop/NTU/Scam/Code/0224_Imbalanced_M10.csv", delimiter=",")
    # np.random.shuffle(train_Scam)
    # np.random.shuffle(train_Malicious)
    # shuffle_train_Malicious = train_Malicious[:5833,:]
    # np.savetxt('shuffle_train_Malicious_'+str(time.time())+'.csv',shuffle_train_Malicious, delimiter=',')
    # print('save shuffle Malicious csv...')
    # shuffle_train_Scams = train_Scam[:2430,:]
    # print('save shuffle Scams csv...')
    # np.savetxt('shuffle_train_Scams_'+str(time.time())+'.csv',shuffle_train_Scams, delimiter=',')
    # =========Certain Training Dataset========
    shuffle_train_Malicious = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/shuffle_train_Malicious_0.97 - 42.csv", delimiter=",")
    shuffle_train_Scams = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/shuffle_train_Scams_0.97 - 42.csv", delimiter=",")
    #=====merge csv to train=====
    #train = shuffle_train_Malicious + shuffle_train_Scams
    train = np.concatenate((shuffle_train_Malicious, shuffle_train_Scams))
    np.random.shuffle(train)
    #train_X = np.column_stack((train[:, 0:19],train[:, 41]))
    train_X = train[:, 19:42]
    train_y = train[:, 42]
    # Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    #train_X = min_max_scaler.fit_transform(train_X)  # normalize
    counter_y = Counter(train_y)
    Weight = round((counter_y[0.0]) / (counter_y[1.0]))  # int
    #Weight = None
    test = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/0318_testing_balanced - 42.csv", delimiter=",")
    np.random.shuffle(test)
    min_max_scaler = preprocessing.MinMaxScaler()
    test_X = test[:, 19:42]
    #test_X = np.column_stack((test[:, 0:19],test[:, 41]))
    test_y = test[:, 42]
    #test_X = min_max_scaler.fit_transform(test_X)  # normalize
    # Weight_lightgbm_10fold(train_X, train_y, Weight)
    #lightgbm_importance(train_X, train_y)
    #ANOVA(train_X, train_y)
    #model = SVM(train_X, train_y)
    #XGBoost_importance(train_X, train_y)
    # Random
    
    #Xgboost_10fold(train_X, train_y)
    # Smote
    #X, y = Smote_upsampling(train_X, train_y)
    # gbm = RandomForest(train_X, train_y)
    # gbm=Xgboost(train_X, train_y)
    # gbm = Tree_10flod(train_X, train_y)
    #gbm = Weight_lightgbm_10fold(train_X, train_y, Weight)
    plt_lightGBM(train_X, train_y,test_X,test_y)
    # gbm=Weight_LightGBM(train_X, train_y, Weight)
    # joblib.dump(gbm, 'save_model/Lightgbm_10fold_Weight_gbdt_0318.pkl')
    # print("save model...")
    # gbm2 = joblib.load('save_model/Lightgbm_10fold_Weight_gbdt_0318.pkl')
    # Test_light_rf(gbm2, test_X, test_y,train_X, train_y)
    # print(Weight)
    #X1, y1 = SMOTEENN_upsampling(train_X, train_y)
    #X2, y2 = SMOTETomek_upsampling(train_X, train_y)
    #Catboost(train_X, train_y)
    #XGBoost_importance(train_X, train_y)
    #Tree(train_X, train_y)
    #Tree_10flod(train_X, train_y)
    #
    #lightgbm(train_X, train_y)
    # lightgbm(X1, y1)
    # lightgbm(X2, y2)
    # print("---RandomForest----")
    # RandomForest(train_X, train_y)
    # print("---Xgboost----")
    
    #lightgbm(train_X, train_y)
    # print("---Tree_10flod----")
    # Tree_10flod(train_X, train_y)
    # print("---Xgboost_10fold----")
    # Xgboost_10fold(train_X, train_y)
    #lightgbm_10fold(train_X, train_y)

    #lightgbm_importance(train_X, train_y)
    #Catboost_10fold(train_X, train_y)
