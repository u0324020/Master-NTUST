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


def Smote_upsampling(train_X, train_y):
    print(Counter(train_y))
    # for i in range(1, 43):
    #     plt_1 = plt.scatter(train_X[:10556, i - 1], train_X[:10556, i], c='b',
    #                         marker='o', s=40, cmap=plt.cm.Spectral)  # x,y 特徵
    #     plt_2 = plt.scatter(train_X[10556:, i - 1], train_X[10556:, i], c='r',
    #                         marker='x', s=50, cmap=plt.cm.Spectral)
    #     plt.legend(handles=[plt_1, plt_2], loc='upper left')
    #     plt.savefig('C:/Users/Jane/Desktop/NTU/Scam/Code/image/2Befor_Smote_F' +
    #                 str(i) + '.png', dpi=150)
    # 顏色: m, c, r, b
    # 點: o, x, +, *, v
    # 透明度 alpha(0~1)

    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(train_X, train_y)
    print(Counter(y_smo))
    # for i in range(1, 43):
    #     plt_1 = plt.scatter(X_smo[:10556, i - 1], X_smo[:10556, i], c='b',
    #                         marker='o', s=40, cmap=plt.cm.Spectral)  # x,y 特徵
    #     plt_2 = plt.scatter(X_smo[10556:, i - 1], X_smo[10556:, i], c='r',
    #                         marker='x', s=50, cmap=plt.cm.Spectral)
    #     plt.legend(handles=[plt_1, plt_2], loc='upper left')
    #     plt.savefig('C:/Users/Jane/Desktop/NTU/Scam/Code/image/2After_Smote_F' +
    #                 str(i) + '.png', dpi=150)
    return (X_smo, y_smo)


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
           "error", "logloss"], eval_set=eval_set, verbose=True)
    print(XG)
    y_pred = XG.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    end = time.time()
    results = XG.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()

    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()
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
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
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
    plot_importance(model)
    pyplot.savefig('feature_importance_Xgboost(992).png', dpi=150)
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


def lightgbm(X, y):
    class_names = ['Malicious', 'Scam']
    start = time.time()
    seed = 7
    test_size = 0.5
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
    fpr, tpr, threshold = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    # plt.style.use('ggplot')
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


def AdaBoost(X, y):  # 模型太弱就用Ada
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


if __name__ == '__main__':
    train = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/1206-importance.csv", delimiter=",")
    np.random.shuffle(train)
    train_X = train[:, 0:44]
    train_y = train[:, 44]
    #ANOVA(train_X, train_y)
    #XGBoost_importance(train_X, train_y)
    #Tree_10flod(train_X, train_y)
    #Xgboost_10fold(train_X, train_y)
    # Smote
    #X, y = Smote_upsampling(train_X, train_y)
    #Catboost(train_X, train_y)
    #XGBoost_importance(train_X, train_y)
    #Tree(train_X, train_y)
    #Tree_10flod(X, y)
    #Xgboost(train_X, train_y)
    lightgbm(train_X, train_y)
    #Xgboost_10fold(X, y)
    #Catboost_10fold(train_X, train_y)
    #lightgbm_10fold(train_X, train_y)
    #lightgbm_importance(train_X, train_y)
