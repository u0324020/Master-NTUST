import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pylab import *
import numpy as np
import os

# 標準化，資料平均值變為0，標準差變為1
sc = StandardScaler()

# 更改label的數值
list1 = ['a', 'b', 'c']
list2 = [1, 50, 189]
list3 = [0, 1, 2]

# 參數
# 特徵數
Input_node = 37
# 類別數
Output_node = label_no = 3
Hidden_stddev =0.1
Hidden_1 = 60
Hidden_2 = 30
Hidden_3 = 10
Learning_rate = 0.01
beta = 0.001
epoch = 0
Dropout_keep_prob = 0.7
stopping_step = 0
Stop_Threshold = 20000
min_value = 0
min_value_save = 0

# 訓練數據集
X_no_sc_original = np.genfromtxt("/home/uadmin/StupidCode/Code/3Family/3family.csv", delimiter=',', dtype="float64")
Y_normal_original = np.genfromtxt("/home/uadmin/StupidCode/Code/3Family/3family_label.csv", delimiter=',', dtype="float64")

X_no_sc, X_no_sc_Test, Y_normal, Y_Test_normal = train_test_split(X_no_sc_original, Y_normal_original, test_size=0.3, random_state=87)

XX_no_sc, X_test_validation_original, YY_normal, y_test_validation_original = train_test_split(X_no_sc, Y_normal, test_size=0.3, random_state=87)

# 更改訓練集label
for i in range(len(list1)):
    list1[i] = (Y_normal == list2[i])
    Y_normal[list1[i]] = list3[i]

for i in range(len(list1)):
    list1[i] = (y_test_validation_original == list2[i])
    y_test_validation_original[list1[i]] = list3[i]

# 訓練集進行標準化及轉成one-hot形式
X = sc.fit_transform(XX_no_sc)
Y_train = tf.one_hot(YY_normal, label_no)


# X_test_validation 與 y_test_validation 為轉換後的數據
X_test_validation = sc.fit_transform(X_test_validation_original)
y_test_validation = tf.one_hot(y_test_validation_original, label_no)

# 測試數據集




# 更改測試集label
for i in range(len(list1)):
    list1[i] = (Y_Test_normal == list2[i])
    Y_Test_normal[list1[i]] = list3[i]

# 訓練集進行標準化及轉成one-hot形式
X_Test = sc.fit_transform(X_no_sc_Test)
Y_test = tf.one_hot(Y_Test_normal, label_no)

# 定義兩個placeholder，定義類形及形狀
x = tf.placeholder(tf.float32, [None, Input_node])
y = tf.placeholder(tf.float32, [None, Output_node])
keep_prob = tf.placeholder(tf.float32)

# 創建簡單的神經網路，一般權值初始化不給零，效果會比初始值為零好，初始化為重要的環節
# 初始化用一個截斷常態分佈的方式，標準差為0.1
# 此例故意訂定多層、多顆神經，以達到複雜網路，為了得到過擬合，以再使用dropout
# 過擬合的狀況為網路太過複雜，數據太少
# 第一層神經網路
W1 = tf.Variable(tf.truncated_normal([Input_node, Hidden_1], stddev=Hidden_stddev))
b1 = tf.Variable(tf.zeros([Hidden_1]) + 0.1)
# 定一個L1的輸出，激活函數使用雙曲函數tanh()
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 用一個dropout函數，輸入L1(為一層深經元輸出)，keep_prob為設置多少比例的神經元工作(0~1，1為全部神經元work)
L1_drop = tf.nn.dropout(L1, keep_prob)

# 第二層神經網路
W2 = tf.Variable(tf.truncated_normal([Hidden_1, Hidden_2], stddev=Hidden_stddev))
b2 = tf.Variable(tf.zeros([Hidden_2]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 第三層神經網路
W3 = tf.Variable(tf.truncated_normal([Hidden_2, Hidden_3], stddev=Hidden_stddev))
b3 = tf.Variable(tf.zeros([Hidden_3]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

# 輸出層
W4 = tf.Variable(tf.truncated_normal([Hidden_3, Output_node], stddev=Hidden_stddev))
b4 = tf.Variable(tf.zeros([Output_node]) + 0.1)

# 使用softmax函數，信號總和(x*W)經過softmax函數，可以得到一個概率值
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 交叉熵
#regularizers = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))+beta*regularizers

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)

# 初始化變量
init = tf.global_variables_initializer()

# 結果存放在一個布林型列表中
# equal()用來比較裡面兩參數之大小，一樣的話True，不一樣的話False
# argmax()用來求裡面第一個參數最大的值是哪個位置，返回一維張量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求準確率
# cast()將預測對比結果轉換類型(correct_prediction為布林型，將其轉換為float32浮點型)
# 轉換完就可得平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 將one-hot 形式轉成normal type，並得到混淆矩陣
con_Y = tf.argmax(y, axis=1)
con_prediction = tf.argmax(prediction, axis=1)
con = tf.confusion_matrix(labels=con_Y, predictions=con_prediction)

# 定義會話
#with tf.Session(config=tf.ConfigProto(allow_soft_placement = True,log_device_placement = True)) as sess:
with tf.Session(config=tf.ConfigProto()) as sess:
    sess.run(init)
    Y_train = sess.run(Y_train)
    Y_Test = sess.run(Y_test)
    y_test_validation = sess.run(y_test_validation)

    # 把所有圖片訓練21次
    fig_x, fig_tranning_accuracy, fig_validation_accuracy, fig_test_accuracy, fig_validation_loss, fig_test_loss, fig_tranning_loss = [], [], [], [], [], [], []
    while True:
        fig_x.append(epoch)
        # 把所有的圖片循環了一次，每次批次100
        # for batch in range(n_batch):
        # 獲得一個批次，大小為100
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 每次的批次進行訓練
        # 設定所有神經元都是工作的(1.0)，相當於dropout是沒有用的
        # 使用Dropout訓練，0.7表在每一層的輸出使用dropout，70%的神經元工作

        # sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        sess.run(train_step, feed_dict={x: X, y: Y_train, keep_prob: 0.8})

        # 準確率
        # 測試數據的準確率
        test_acc = sess.run(accuracy, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1.0})
        fig_test_accuracy.append(test_acc)
        # 訓練數據的準確率
        train_acc = sess.run(accuracy, feed_dict={x: X, y: Y_train, keep_prob: 1.0})
        fig_tranning_accuracy.append(train_acc)
        # 驗證數據的準確率
        validation_acc = sess.run(accuracy, feed_dict={x: X_test_validation, y: y_test_validation, keep_prob: 1.0})
        fig_validation_accuracy.append(validation_acc)

        # 訓練
        training_loss = sess.run(loss, feed_dict={x: X, y: Y_train, keep_prob: 1.0})
        fig_tranning_loss.append(training_loss)
        # 數據的loss
        validation_loss = sess.run(loss, feed_dict={x: X_test_validation, y: y_test_validation, keep_prob: 1.0})
        fig_validation_loss.append(validation_loss)
        # 測試的loss
        test_loss = sess.run(loss, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1.0})
        fig_test_loss.append(test_loss)
        # 印當前是第幾個週期及準確率
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc) + ", Training Accuracy " + str(
            train_acc) + ", Validation Accuracy " + str(validation_acc) + ", Validation loss = " + str(validation_loss))

        if (epoch == 0):
            min_value = fig_validation_loss[epoch]
        else:
            if (min_value > fig_validation_loss[epoch]):
                min_value = fig_validation_loss[epoch]
                stopping_step = 0
            else:
                min_value = min_value
                stopping_step += 1
        if (stopping_step >= Stop_Threshold):
            epoch = epoch
            print('early stopping', epoch)
            f = open('back_adam.txt', 'a')
            lines = ['epoch:' + str(epoch), '\n', 'Testing Accuracy:' + str(test_acc), '\n',
                     'Training Accuracy:' + str(train_acc), '\n', 'Validation Accuracy:' + str(validation_acc), '\n',
                     'Validation_loss:' + str(validation_loss)]
            f.writelines(lines)
            f.close()
            break;
        elif (epoch >= 6e5 and abs(fig_validation_loss[epoch - 1] - fig_validation_loss[epoch]) < 1e-6):
            epoch = epoch
            print('early stopping', epoch)
            f = open('back_adam.txt', 'a')
            lines = ['epoch:' + str(epoch), '\n', 'Testing Accuracy:' + str(test_acc), '\n',
                     'Training Accuracy:' + str(train_acc), '\n', 'Validation Accuracy:' + str(validation_acc), '\n',
                     'Validation_loss:' + str(validation_loss)]
            f.writelines(lines)
            f.close()
            break;
        epoch += 1

    fig_x = np.array(fig_x)
    fig_tranning_accuracy = np.array(fig_tranning_accuracy)
    fig_validation_accuracy = np.array(fig_validation_accuracy)
    fig_test_accuracy = np.array(fig_test_accuracy)
    fig_tranning_loss = np.array(fig_tranning_loss)
    fig_validation_loss = np.array(fig_validation_loss)
    fig_test_loss = np.array(fig_test_loss)

    print(fig_x.shape)

    # 畫test_acc圖
    plt.plot(fig_x[0:epoch], fig_tranning_accuracy[0:epoch], '-b', label='Tranning')
    plt.plot(fig_x[0:epoch], fig_validation_accuracy[0:epoch], '-r', label='Validation')
    plt.plot(fig_x[0:epoch], fig_test_accuracy[0:epoch], '-g', label='Test')
    plt.legend(loc='upper left')
    # 設定y軸為0~1區間
    plt.ylim(0, 1)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.savefig('3family_acc.png', dpi=800)
    plt.show()

    # 畫test_loss圖
    plt.plot(fig_x[0:epoch], fig_tranning_loss[0:epoch], '-b', label='Tranning')
    plt.plot(fig_x[0:epoch], fig_validation_loss[0:epoch], '-r', label='Validation')
    plt.plot(fig_x[0:epoch], fig_test_loss[0:epoch], '-g', label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.savefig('3family_loss.png', dpi=800)
    plt.show()

    # 畫混淆矩陣
    confmat = sess.run(con, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1})
    ax = plt.matshow(confmat, cmap=plt.cm.Blues)
    # ax = plt.matshow(confmat, cmap=plt.cm.Blues, alpha=5)

    for i in range(confmat.shape[1]):
        for j in range(confmat.shape[0]):
            ax = plt.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize=8)

    # plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(arange(len(list2)), list2, fontsize=8)
    plt.yticks(arange(len(list2)), list2, fontsize=8)
    plt.title('Predict', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.savefig('3family_confusionmatrix.png', dpi=800)
    plt.show()
