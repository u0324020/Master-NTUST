import time
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

start = time.clock()  # 计时
min_max_scaler = preprocessing.MinMaxScaler()

# 读取训练数据 并数据规整化
raw_data = pd.read_csv('train_data.csv')
raw_datax = raw_data[:20000]
X1_scaled = min_max_scaler.fit_transform(raw_datax.ix[:, 3:7])
y1 = raw_datax['Y1']
y1 = list(y1)

# 读取测试数据 并数据规整化
raw_datat = pd.read_csv('test_data.csv')
raw_datatx = raw_datat[:10000]
X1t_scaled = min_max_scaler.fit_transform(raw_datatx.ix[:, 3:7])
y1t = raw_datatx['Y1']
y1t = list(y1t)

print len(X1_scaled)
print len(X1t_scaled)
end = time.clock()
print '运行时间:', end - start

clf = DecisionTreeClassifier().fit(X1_scaled, y1)
clfb = BaggingClassifier(base_estimator=DecisionTreeClassifier(
), max_samples=0.5, max_features=0.5).fit(X1_scaled, y1)

predict = clf.predict(X1t_scaled)
predictb = clfb.predict(X1t_scaled)

print clf.score(X1t_scaled, y1t)
print clfb.score(X1t_scaled, y1t)

# print Series(predict).value_counts()
# print Series(predictb).value_counts()
