import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import loadtxt
from sklearn import preprocessing
from collections import Counter
from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

train = loadtxt(
    "C:/Users/Jane/Desktop/NTU/Scam/Code/1206-importance.csv", delimiter=",")
min_max_scaler = preprocessing.MinMaxScaler()
np.random.shuffle(train)
X = train[:, 0:44]
y = train[:, 44]
X = min_max_scaler.fit_transform(X)  # normalize
fig = plt.figure()
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
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
y3 = np.arctan2(X[:, 0], X[:, 1], X[:, 2])  # ranbow
ax.scatter(X[:924, 0], X[:924, 1], X[:924, 2], c='red', cmap="brg", s=80, alpha=0.5,
           edgecolor='k', marker='x', label='Scam')
ax.scatter(X[925:, 0], X[925:, 1], X[925:, 2], c='blue', cmap="brg", s=80,
           edgecolor='k', marker='o', label='Malicious')  # overlap
ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.set_zlabel('PCA-3')
plt.legend()
plt.show()
