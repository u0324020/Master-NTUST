import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pylab import *
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from collections import Counter


# 標準化，資料平均值變為0，標準差變為1
sc = StandardScaler()

# 更改label的數值
list1 = ['a', 'b', 'c']
list2 = [1, 50, 189]
list3 = [0, 1, 2]

# 參數
# 特徵數
Input_node = 43
# 類別數
Output_node = label_no = 2
Hidden_stddev = 0.3
Hidden_1 = 100
Hidden_2 = 50
Hidden_3 = 30
Learning_rate = 0.005
beta = 0.001
epoch = 0
Dropout_keep_prob = 0.7
stopping_step = 0
Stop_Threshold = 3000
min_value = 0
min_value_save = 0
# 存放作圖時的x軸及y軸

# 使用第幾顆GPU
os.environ["CUDA_VISIBLE_DEVICES"] ='0'

# 載入數據集
X_no_sc_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/Mark/training.csv", delimiter=',', dtype="float64")

Y_normal_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/Mark/label.csv", delimiter=',', dtype="float64")
print(Counter(Y_normal_original))
X_no_sc, X_test, Y_normal, Y_test = train_test_split(X_no_sc_original, Y_normal_original, test_size=0.3, random_state=56)
sm = SMOTE(ratio={0: 41000}, random_state=1)
X_res, y_res = sm.fit_sample(X_no_sc_original, Y_normal_original)
print(Counter(y_res))
print("OK")
# # 更改訓練集label
# for i in range(len(list1)):
#     list1[i] = (Y_normal == list2[i])
#     Y_normal[list1[i]] = list3[i]

# for i in range(len(list1)):
#     list1[i] = (Y_test == list2[i])
#     Y_test[list1[i]] = list3[i]

# # 訓練集進行標準化及轉成one-hot形式
# X_Train = sc.fit_transform(X_no_sc)
# Y_train = tf.one_hot(Y_normal, label_no)
# X_Test = sc.fit_transform(X_test)
# Y_test = tf.one_hot(Y_test, label_no)

# # 定義兩個placeholder，定義類形及形狀
# x = tf.placeholder(tf.float32, [None, Input_node])
# y = tf.placeholder(tf.float32, [None, Output_node])
# keep_prob = tf.placeholder(tf.float32)

# # 第一層神經網路
# W1 = tf.Variable(tf.truncated_normal([Input_node, Hidden_1], stddev=Hidden_stddev))
# b1 = tf.Variable(tf.zeros([Hidden_1]) + 0.1)
# # 定一個L1的輸出，激活函數使用雙曲函數tanh()
# L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# # 用一個dropout函數，輸入L1(為一層深經元輸出)，keep_prob為設置多少比例的神經元工作(0~1，1為全部神經元work)
# L1_drop = tf.nn.dropout(L1, keep_prob)

# # 第二層神經網路
# W2 = tf.Variable(tf.truncated_normal([Hidden_1, Hidden_2], stddev=Hidden_stddev))
# b2 = tf.Variable(tf.zeros([Hidden_2]) + 0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# L2_drop = tf.nn.dropout(L2, keep_prob)

# # 第三層神經網路
# W3 = tf.Variable(tf.truncated_normal([Hidden_2, Hidden_3], stddev=Hidden_stddev))
# b3 = tf.Variable(tf.zeros([Hidden_3]) + 0.1)
# L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# L3_drop = tf.nn.dropout(L3, keep_prob)

# # 輸出層
# W4 = tf.Variable(tf.truncated_normal([Hidden_3, Output_node], stddev=Hidden_stddev))
# b4 = tf.Variable(tf.zeros([Output_node]) + 0.1)

# # 使用softmax函數，信號總和(x*W)經過softmax函數，可以得到一個概率值
# prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# # 交叉熵
# regularizers = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))+beta*regularizers

# # 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(Learning_rate).minimize(loss)

# # 初始化變量
# init = tf.global_variables_initializer()

# # equal()用來比較裡面兩參數之大小，一樣的話True，不一樣的話False
# # argmax()用來求裡面第一個參數最大的值是哪個位置，返回一維張量中最大的值所在的位置
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# # 準確率
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # 將one-hot 形式轉成normal type，並得到混淆矩陣
# con_Y = tf.argmax(y, axis=1)
# con_prediction = tf.argmax(prediction, axis=1)
# con = tf.confusion_matrix(labels=con_Y, predictions=con_prediction)

# # 定義會話
# with tf.Session(config=tf.ConfigProto()) as sess:
#     sess.run(init)
#     Y_Train = sess.run(Y_train)
#     Y_Test = sess.run(Y_test)

#     # 把所有圖片訓練21次
#     fig_x, fig_tranning_accuracy, fig_validation_accuracy, fig_test_accuracy, fig_validation_loss, fig_test_loss, fig_tranning_loss = [], [], [], [], [], [], []
#     while True:
#         fig_x.append(epoch)
#         sess.run(train_step, feed_dict={x: X_Train, y: Y_Train, keep_prob: 0.8})

#         # 準確率
#         # 測試數據的準確率
#         test_acc = sess.run(accuracy, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1.0})
#         fig_test_accuracy.append(test_acc)
#         # 訓練數據的準確率
#         train_acc = sess.run(accuracy, feed_dict={x: X_Train, y: Y_Train, keep_prob: 1.0})
#         fig_tranning_accuracy.append(train_acc)

#         # 訓練
#         training_loss = sess.run(loss, feed_dict={x: X_Train, y: Y_Train, keep_prob: 1.0})
#         fig_tranning_loss.append(training_loss)
#         # 測試的loss
#         test_loss = sess.run(loss, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1.0})
#         fig_test_loss.append(test_loss)
#         # 印當前是第幾個週期及準確率
#         print("Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc) + ", Training Accuracy " + str(
#             train_acc)+", Test Loss " + str(test_loss))

#         if (epoch == 0):
#             min_value = fig_test_loss[epoch]
#         else:
#             if (min_value > fig_test_loss[epoch]):
#                 min_value = fig_test_loss[epoch]
#                 stopping_step = 0
#             else:
#                 min_value = min_value
#                 stopping_step += 1
#         if (stopping_step >= Stop_Threshold):
#             epoch = epoch
#             print('early stopping', epoch)
#             f = open('back_adam.txt', 'a')
#             lines = ['epoch:' + str(epoch), '\n', 'Testing Accuracy:' + str(test_acc), '\n',
#                      'Training Accuracy:' + str(train_acc), '\n']
#             f.writelines(lines)
#             f.close()
#             break;
#         elif (epoch >= 6e5 and abs(fig_test_loss[epoch - 1] - fig_test_loss[epoch]) < 1e-6):
#             epoch = epoch
#             print('early stopping', epoch)
#             f = open('back_adam.txt', 'a')
#             lines = ['epoch:' + str(epoch), '\n', 'Testing Accuracy:' + str(test_acc), '\n',
#                      'Training Accuracy:' + str(train_acc), '\n']
#             f.writelines(lines)
#             f.close()
#             break;
#         epoch += 1

#     fig_x = np.array(fig_x)
#     fig_tranning_accuracy = np.array(fig_tranning_accuracy)
#     fig_test_accuracy = np.array(fig_test_accuracy)
#     fig_tranning_loss = np.array(fig_tranning_loss)
#     fig_test_loss = np.array(fig_test_loss)

#     print(fig_x.shape)

#     # 畫test_acc圖
#     plt.plot(fig_x[0:epoch], fig_tranning_accuracy[0:epoch], '-b', label='Tranning')
#     plt.plot(fig_x[0:epoch], fig_test_accuracy[0:epoch], '-g', label='Test')
#     plt.legend(loc='upper left')
#     # 設定y軸為0~1區間
#     plt.ylim(0, 1)
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Accuracy', fontsize=14)
#     plt.savefig('dataset10_acc.png', dpi=800)
#     plt.show()

#     # 畫test_loss圖
#     plt.plot(fig_x[0:epoch], fig_tranning_loss[0:epoch], '-b', label='Tranning')
#     plt.plot(fig_x[0:epoch], fig_test_loss[0:epoch], '-g', label='Test')
#     plt.legend(loc='upper right')
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Loss', fontsize=14)
#     plt.savefig('dataset10_loss.png', dpi=800)
#     plt.show()

#     # 畫混淆矩陣
#     confmat = sess.run(con, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1})
#     ax = plt.matshow(confmat, cmap=plt.cm.Blues)

#     for i in range(confmat.shape[1]):
#         for j in range(confmat.shape[0]):
#             ax = plt.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize=8)

#     plt.colorbar()
#     plt.xticks(arange(len(list2)), list2, fontsize=8)
#     plt.yticks(arange(len(list2)), list2, fontsize=8)
#     plt.title('Predict', fontsize=14)
#     plt.ylabel('Actual', fontsize=14)
#     plt.savefig('dataset10_confusionmatrix.png', dpi=800)
#     plt.show()