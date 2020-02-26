# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:08:04 2020

@author: Jane

Based on LightGBM to classify Scams and Malicious
Reference:https://www.cnblogs.com/bjwu/p/9307344.html
"""
from collections import Counter
from sklearn.metrics import mean_squared_error
from numpy import loadtxt
import warnings
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from matplotlib import pyplot as plt
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import itertools
from sklearn.metrics import roc_auc_score
import time
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV


def train_lightGBM(train_X, train_y, W):
    data_train = lgb.Dataset(train_X, train_y, silent=True)
    # lgb_eval = lgb.Dataset(test_X, test_y, reference=data_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #'device': 'gpu',
        'metric': 'rmse',
        'num_leaves': 50,  # 調整樹的複雜度，< 2^(max_depth)才不會overfitting
        'max_depth': 6,  # 太深會overfitting
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 20,
        'bagging_freq': 5,
        'verbose': -1,
        #'is_unbalance':True,
        'scale_pos_weight': W
    }
    # 5-FOLD
    cv_results = lgb.cv(
        params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

    print('best n_estimators:', len(cv_results['rmse-mean']))
    print('best cv score:', cv_results['rmse-mean'][-1])

    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=1000,
    #                 valid_sets=lgb_eval,
    #                 early_stopping_rounds=200)
    # return gbm
    '''
        # learning_rate 0.1 => [50]    cv_agg's rmse: 0.0860653 + 0.00255888
								[100]   cv_agg's rmse: 0.0659171 + 0.00276962
								[150]   cv_agg's rmse: 0.0573417 + 0.0020961
								[200]   cv_agg's rmse: 0.052938 + 0.0028057
								[250]   cv_agg's rmse: 0.0487741 + 0.0028679
								[300]   cv_agg's rmse: 0.0478702 + 0.00265815
								best n_estimators: 280
								best cv score: 0.0459242908447749

        # learning_rate 0.01 => [50]    cv_agg's rmse: 0.0999004 + 0.00102095
								[100]   cv_agg's rmse: 0.0963473 + 0.000699178
								best n_estimators: 87
								best cv score: 0.09606058128842916
    '''


def sk_lgb(df_train, y_train, W):
    '''
    #'max_depth': 7, 'num_leaves': 80, 'rmse':0.08826635153634467(np.sqrt(-(-0.007790948813537576)))
    #Test2 {'max_depth': 8, 'num_leaves': 92} 'rmse': 0.08080399468126882(-0.00652928555645052
    '''
    model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=50, learning_rate=0.1, scale_pos_weight=W,
                                   n_estimators=43, max_depth=6, metric='rmse', bagging_fraction=0.8, feature_fraction=0.8)

    params_test1 = {
        'max_depth': range(3, 8, 2),
        'num_leaves': range(50, 170, 30)
    }
    params_test2 = {
        'max_depth': [6, 7, 8],
        'num_leaves': [68, 74, 80, 86, 92]}
    gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test2,
                            scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
    gsearch1.fit(df_train, y_train)
    print(gsearch1.best_params_, gsearch1.best_score_, gsearch1.param_grid)


def test_model(model, X, y):
    y_pred = model.predict(X)
    print("The rmse of loaded model's prediction is:",
          mean_squared_error(y, y_pred) ** 0.5)


if __name__ == '__main__':
    train_path = "C:/Users/Jane/Desktop/NTU/Scam/Code/0224_Imbalanced_S2500.csv"
    train = loadtxt(train_path, delimiter=",")
    np.random.shuffle(train)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_X = train[:, 0:44]
    train_y = train[:, 44]
    train_X = min_max_scaler.fit_transform(train_X)  # normalize
    counter_y = Counter(train_y)
    Weight = round((counter_y[0.0]) / (counter_y[1.0]))  # int
    # splitting
    # seed = 21
    # test_size = 0.3
    # X_train, X_test, y_train, y_test = train_test_split(
    #     train_X, train_y, test_size=test_size, random_state=seed)

    # training
    # train_lightGBM(train_X, train_y, Weight)  # 找LearingReat
    sk_lgb(train_X, train_y, Weight)
    # model = lightGBM(train_X, train_y, Weight)
    # saving model

    # testing######
    # test_path = "C:/Users/Jane/Desktop/NTU/Scam/Code/0225_testing_balanced.csv"
    # test = loadtxt(train_path, delimiter=",")
    # np.random.shuffle(test)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # test_X = test[:, 0:44]
    # y = test[:, 44]
    # X = min_max_scaler.fit_transform(train_X)
    # test_model(model, X, y)
