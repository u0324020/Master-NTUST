# 訓練0跟208兩類別，不加batch_size，畫出test_acc、test_loss、confusion matrix的圖，丟進去的數據集label的部分使用normal type，label部分使用原本的數值


# from __future__ import division, print_function, absolute_import
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pylab import *
import numpy as np
import csv
from tensorflow.contrib import rnn
import os
from imblearn.over_sampling import SMOTE
from collections import Counter
# 標準化，資料平均值變為0，標準差變為1
sc = StandardScaler()
time = '01'
# Parameters
lrn=learning_rate = 0.01  # 0.01 this learning rate will be better! Tested
epoch = 0
stopping_step = 0
# 要改
Stop_Threshold = 5000
batch_size = 256
display_step = 1
Input_node = 39

# label_no = 20
# batch_size = 50

n_hidden_1 = 30
#n_hidden_2 = 15
Output_node = 25

std=Hidden_stddev = 0.2
Dropout_keep_prob = 0.7

# 存放作圖時的x軸及y軸
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 載入數據集
X_no_sc_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/autoencoder/dataset (original).csv", delimiter=',', dtype="float64")

Y_normal_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/autoencoder/dataset_label.csv", delimiter=',', dtype="float64")
print(Counter(Y_normal_original))
X_no_sc, X_test, Y_normal, Y_test = train_test_split(X_no_sc_original, Y_normal_original, test_size=0.3, random_state=56)
sm = SMOTE(ratio={0: 61500,1: 61500}, random_state=1)
X_no_sc_original, Y_normal_original = sm.fit_sample(X_no_sc_original, Y_normal_original)
print(Counter(Y_normal_original))
# with tf.device('/gpu:0'):
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

# X_no_sc_original = np.genfromtxt("/home/uadmin/bensoncode/lstm_10_family/20family.csv", delimiter=',',dtype="float64")

# Y_normal_original = np.genfromtxt("/home/uadmin/bensoncode/lstm_10_family/20family_labeled.csv",delimiter=',', dtype="float64")

# X_no_sc_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/autoencoder/dataset (original).csv", delimiter=',',dtype="float64")

# Y_normal_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/autoencoder/dataset_label.csv",delimiter=',', dtype="float64")  # y

# X_no_sc_original = np.genfromtxt("C:/Users/m4050/PycharmProjects/tensor/0709_BACKPROPAGATION_WITH_VALIDATION/0720(12family)/malicious_10family.csv", delimiter=',', dtype="float64")
# Y_normal_original = np.genfromtxt("C:/Users/m4050/PycharmProjects/tensor/0709_BACKPROPAGATION_WITH_VALIDATION/0720(12family)/malicious_10family_labeled.csv",delimiter=',', dtype="float64")

# 訓練集進行標準化及轉成one-hot形式

X_auto = sc.fit_transform(X_no_sc_original)

# Y_train = tf.one_hot(Y_normal_original, label_no)

# drop
keep_prob = tf.placeholder(tf.float32)

# 定義兩個placeholder，定義類形及形狀
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, Input_node])

    y = tf.placeholder(tf.float32, [None, Output_node])

# 創建簡單的神經網路，一般權值初始化不給零，效果會比初始值為零好，初始化為重要的環節
# 初始化用一個截斷常態分佈的方式，標準差為0.1
# 此例故意訂定多層、多顆神經，以達到複雜網路，為了得到過擬合，以再使用dropout
# 過擬合的狀況為網路太過複雜，數據太少

with tf.name_scope('weights'):
    enconder_weights1 = tf.Variable(tf.truncated_normal([Input_node, n_hidden_1], stddev=0.1), dtype=tf.float32,
                                    name='enconder_W1')
    #enconder_weights2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1), dtype=tf.float32,
    #                                name='enconder_W2')
    enconder_weights_output = tf.Variable(tf.truncated_normal([n_hidden_1, Output_node], stddev=0.1), dtype=tf.float32,
                                          name='enconder_W_output')

    deconder_weights1 = tf.Variable(tf.truncated_normal([Output_node, n_hidden_1], stddev=0.1), dtype=tf.float32,
                                    name='deconder_W1')
    #deconder_weights2 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.1), dtype=tf.float32,
    #                                name='deconder_W2')
    deconder_weights_output = tf.Variable(tf.truncated_normal([n_hidden_1, Input_node], stddev=0.1), dtype=tf.float32,
                                          name='deconder_W_output')

    # W_in = tf.Variable(tf.random_normal([Input_node, n_hidden_units]))
    # W_out = tf.Variable(tf.random_normal([W_in, Output_node]))

with tf.name_scope('bias'):
    enconder_bias1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_1], name='enconder_b1'))
    #enconder_bias2 = tf.Variable(tf.constant(0.1, shape=[n_hidden_2], name='enconder_b2'))
    enconder_bias_output = tf.Variable(tf.constant(0.1, shape=[Output_node], name='enconder_b_output'))

    #deconder_bias1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_2], name='enconder_b1'))
    deconder_bias1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_1], name='enconder_b2'))
    deconder_bias_output = tf.Variable(tf.constant(0.1, shape=[Input_node], name='enconder_b_output'))

    # b_in =  tf.Variable(tf.zeros([n_hidden_units]) + 0.1)
    # b_out = tf.Variable(tf.zeros([Output_node]) + 0.1)


def encoder(x_1):
    encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_1, enconder_weights1), enconder_bias1))
    #encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, enconder_weights2), enconder_bias2))
    encoder_output_layer = tf.nn.sigmoid(
        tf.add(tf.matmul(encoder_layer_1, enconder_weights_output), enconder_bias_output))
    return encoder_output_layer


def decoder(x_1):
    decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_1, deconder_weights1), deconder_bias1))
    #decoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer_1, deconder_weights2), deconder_bias2))
    decoder_output_layer = tf.nn.sigmoid(
        tf.add(tf.matmul(decoder_layer_1, deconder_weights_output), deconder_bias_output))
    return decoder_output_layer


# Construct model
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# 預測
y_pred = decoder_op
y_true = x

# cross
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    tf.summary.scalar('loss', loss)

# 使用梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

merged = tf.summary.merge_all()

'''
# 交叉熵
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar('loss', loss)

# 使用梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)

# 初始化變量
init = tf.global_variables_initializer()

# 結果存放在一個布林型列表中
# equal()用來比較裡面兩參數之大小，一樣的話True，不一樣的話False
# argmax()用來求裡面第一個參數最大的值是哪個位置，返回一維張量中最大的值所在的位置
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        # 求準確率
        # cast()將預測對比結果轉換類型(correct_prediction為布林型，將其轉換為float32浮點型)
        # 轉換完就可得平均值
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 將one-hot 形式轉成normal type，並得到混淆矩陣
con_Y = tf.argmax(y, axis=1)
con_prediction = tf.argmax(prediction, axis=1)
con = tf.confusion_matrix(labels=con_Y, predictions=con_prediction)
merged = tf.summary.merge_all()
'''

# 定義會話
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #    writer = tf.summary.FileWriter("logs/", sess.graph)

    # 把所有圖片訓練21次
    fig_x, fig_tranning_accuracy, fig_validation_accuracy, fig_test_accuracy, fig_validation_loss, fig_test_loss, fig_tranning_loss = [], [], [], [], [], [], []
    while True:
        fig_x.append(epoch)
        # 把所有的圖片循環了一次，每次批次100
        # for batch in range(n_batch):

        # start = (epoch*batch_size)
        # end = min(start + batch_size,epoch)

        # 獲得一個批次，大小為100
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 每次的批次進行訓練
        # 設定所有神經元都是工作的(1.0)，相當於dropout是沒有用的
        # 使用Dropout訓練，0.7表在每一層的輸出使用dropout，70%的神經元工作
        # writer1 = tf.summary.FileWriter("C:/Users/m4050/PycharmProjects/tensor/lstm/logs/train", sess.graph)
        # writer2 = tf.summary.FileWriter("C:/Users/m4050/PycharmProjects/tensor/lstm/logs/validation", sess.graph)
        # writer3 = tf.summary.FileWriter("C:/Users/m4050/PycharmProjects/tensor/lstm/logs/test", sess.graph)
        # sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        _, c = sess.run([train_step, loss], feed_dict={x: X_auto})

        if epoch % display_step == 0:
            #            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
            print("Epoch:" + str(epoch), "loss=", "{:.9f}".format(c))

        encoder_result = sess.run(encoder_op, feed_dict={x: X_auto})
        decoder_result = sess.run(decoder_op, feed_dict={x: X_auto})
        # summary1 = sess.run(merged, feed_dict={x: X, y: Y_train, keep_prob: 0.8})
        # summary2 = sess.run(merged, feed_dict={x: X_test_validation, y: y_test_validation, keep_prob: 1.0})
        # summary3 = sess.run(merged, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1.0})

        # writer1.add_summary(summary1, epoch)
        # writer1.flush()

        # writer2.add_summary(summary2, epoch)
        # writer2.flush()

        # writer3.add_summary(summary3, epoch)
        # writer3.flush()

        # 準確率
        # 測試數據的準確率
        # test_acc = sess.run(accuracy, feed_dict={x: X_Test, y: Y_Test, keep_prob: 1.0})
        # fig_test_accuracy.append(test_acc)
        ## 訓練數據的準確率
        # train_acc = sess.run(accuracy, feed_dict={x: X, y: Y_train, keep_prob: 1.0})
        # fig_tranning_accuracy.append(train_acc)
        # 驗證數據的準確率
        # validation_acc = sess.run(accuracy, feed_dict={x: X_test_validation, y: y_test_validation, keep_prob: 1.0})
        # fig_validation_accuracy.append(validation_acc)

        # 訓練
        training_loss = sess.run(loss, feed_dict={x: X_auto})
        fig_tranning_loss.append(training_loss)

        # print("loss = " + str(test_loss))
        #
        '''
        if (fig_validation_loss[epoch-1] < fig_validation_loss[epoch]):
            min_value = fig_validation_loss[epoch]
            for i in range(len(fig_validation_loss)):
                if (min_value > fig_validation_loss[i]):
                    min_value_save = fig_validation_loss[epoch]
                    stopping_step += 1
                else:
                    min_value_save = min_value_save
                    stopping_step = 0
        if (stopping_step >= Stop_Threshold):
            epoch = epoch
            print('early stopping', epoch)
            break;
        epoch += 1
        '''
        if (epoch == 0):
            min_value = fig_tranning_loss[epoch]
        else:
            if (min_value > fig_tranning_loss[epoch]):
                min_value = fig_tranning_loss[epoch]
                stopping_step = 0
            else:
                min_value = min_value
                stopping_step += 1
        if (stopping_step >= Stop_Threshold):
            epoch = epoch
            print('early stopping', epoch)

            break;
        elif (epoch >= 7e5 and abs(fig_tranning_loss[epoch - 1] - fig_tranning_loss[epoch]) < 1e-6):
            epoch = epoch
            print('early stopping', epoch)
            break;
        epoch += 1
    # fig_x, fig_tranning_accuracy, fig_validation_accuracy, fig_test_accuracy, fig_validation_loss, fig_test_loss, fig_tranning_loss

    fig_x = np.array(fig_x)
    fig_tranning_accuracy = np.array(fig_tranning_accuracy)
    fig_validation_accuracy = np.array(fig_validation_accuracy)
    fig_test_accuracy = np.array(fig_test_accuracy)
    fig_tranning_loss = np.array(fig_tranning_loss)
    fig_validation_loss = np.array(fig_validation_loss)
    fig_test_loss = np.array(fig_test_loss)
    print(fig_x.shape)

    # 印出prediction及correct_prediction
    #        print(sess.run(prediction,  feed_dict={x: X, y: Y, keep_prob: 0.7}))
    #        print(sess.run(correct_prediction, feed_dict={x: X, y: Y, keep_prob: 0.7}))
    # 印出混淆矩陣
    #        print(sess.run(con, feed_dict={x: X, y: Y_train, keep_prob: 0.7}))
    '''
    #將數值轉換為numpy
    fig_x =  np.array(fig_x)
    fig_y = np.array(fig_y)
    fig_validation_loss = np.array(fig_validation_loss)


    fig_plot_x = np.zeros((1,epoch))
    fig_plot_y = np.zeros((1,epoch))
    fig_plot_loss = np.zeros((1,epoch))

    for i in range(epoch):
        fig_plot_x[0][i] = fig_x[0][i]
        fig_plot_y[0][i] =  fig_y[0][i]
        fig_plot_loss[0][i] = fig_loss[0][i]
        #將迭代次數及測試的準確率存進陣列裡
    '''
    '''
    test_prediction = (sess.run(prediction,feed_dict={x: X_Test, y: Y_Test, keep_prob: 1}))
    L_test_prediction = test_prediction.tolist()
    with open('test_predition','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([list2])
        writer.writerows(L_test_prediction)
    '''
    print(decoder_result.shape)
    csv_encoder_result = encoder_result.tolist()
    csv_decoder_result = decoder_result.tolist()
    Y_normal_original = np.array(Y_normal_original.tolist())
    with open('csv_encoder_result1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_encoder_result)
    # with open('csv_encoder_label1.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(int(Y_normal_original))
    # print(type(Y_normal_original))
    #plot3d = plt.subplot(projection='polar')

    # ax1.axes.get_yaxis().set_visible(False)
    # ax1.axes.get_xaxis().set_visible(False)
    #    if (Input_data1[j] == 0):
    #        feature_0 =ax1.scatter(Input_data_std[j][8],Input_data_std[j][16] ,Input_data_std[j][25] , c= 'k', marker='x',alpha=.5)
    for i in range(len(Y_normal_original)):
        if (Y_normal_original[i] == 0):
            data_0 = plt.scatter(encoder_result[i][7], encoder_result[i][8], s=10, c='r',
                                    marker='+', alpha=0.5)
        elif (Y_normal_original[i] == 1):
            data_1 = plt.scatter(encoder_result[i][7], encoder_result[i][8], s=10, c='g',
                                    marker='o', alpha=0.5)
        '''elif (Y_normal_original[i] == 31):
            data_31 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10, c='b',
                                     marker='+', alpha=0.5)
        elif (Y_normal_original[i] == 50):
            data_50 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10, c='m',
                                     marker='s', alpha=0.5)
        elif (Y_normal_original[i] == 169):
            data_169 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c='y',
                                      marker='p', alpha=0.5)
        elif (Y_normal_original[i] == 174):
            data_174 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c='k',
                                      marker='*', alpha=0.5)
        elif (Y_normal_original[i] == 186):
            data_186 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c='c',
                                      marker='^', alpha=0.5)
        elif (Y_normal_original[i] == 188):
            data_188 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.5,0.9,0.5],
                                      marker='<', alpha=0.5)
        elif (Y_normal_original[i] == 189):
            data_189 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.5, 0.3, 0.4], marker='>', alpha=0.5)
        elif (Y_normal_original[i] == 193):
            data_193 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.6, 0.8, 0.2], marker='D', alpha=0.5)
        elif (Y_normal_original[i] == 194):
            data_194 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.1, 0.5, 0.8], marker='d', alpha=0.5)
        elif (Y_normal_original[i] == 198):
            data_198 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.8, 0.6, 1], marker='H', alpha=0.5)
        elif (Y_normal_original[i] == 203):
            data_203 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.4, 0.7, 0.2], marker='h', alpha=0.5)
        elif (Y_normal_original[i] == 221):
            data_221 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.7, 0.5, 0.7], marker='x', alpha=0.5)
        elif (Y_normal_original[i] == 222):
            data_222 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.9, 0.3, 0.5], marker='1', alpha=0.5)
        elif (Y_normal_original[i] == 228):
            data_228 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[1, 0.7, 0], marker='2', alpha=0.5)
        elif (Y_normal_original[i] == 231):
            data_231 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0, 1, 0.8], marker='3', alpha=0.5)
        elif (Y_normal_original[i] == 232):
            data_232 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0.4, 0.3, 1], marker='4', alpha=0.5)
        elif (Y_normal_original[i] == 242):
            data_242 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[1, 1, 0.4], marker='8', alpha=0.5)
        elif (Y_normal_original[i] == 339):
            data_339 = plot3d.scatter(encoder_result[i][7], encoder_result[i][8], encoder_result[i][17], s=10,
                                      c=[0, 0.8, 0.3], marker='.', alpha=0.5)'''
    #    else:
    #        feature_0 = ax1.scatter(Input_data_std[j][8], Input_data_std[j][26], Input_data_std[j][27], c='b', marker='+', alpha=.5)

    # plt.legend(ax1)
    # ax1.legend([feature_0,feature_193,feature_194,feature_196,feature_198],['0','193','194','196','198'])
    plt.legend(
        [data_0, data_1],
        ['0', '1'])
    plt.set_title('DDOS_Autoencoder')
    # plot3d.set_zlabel('Z')  # 坐标轴
    plt.set_ylabel('Y')
    plt.set_xlabel('X')
    plt.savefig('DDOS_Autoencoder', dpi=1200)
    #plt.show()
    plt.close()

        # anova過後是7、8、17
        # list2 = [0, 1, 31, 42, 50, 169, 174, 188, 189, 193, 194, 203, 205, 221, 222, 228, 231, 232, 242, 339]
    # 畫test_loss圖
    plt.plot(fig_x[0:epoch], fig_tranning_loss[0:epoch], '-b', label='Tranning')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.savefig('test_loss1_20label_Adam.png', dpi=800)
    #plt.show()
    plt.close()


    '''
 #   list_b = ('0', '1', '31', '42', '50', '169', '174', '188', '189', '193', '194', '203', '205', '221', '222', '228', '231','232', '242', '339')
    plt.figure(figsize=(10, 10))
    plt.scatter(decoder_result[:, 7], decoder_result[:, 8], s=10, c=Y_normal_original, alpha=0.5)

    plt.colorbar()
    plt.savefig('test_distri5_20label_Adam.png', dpi=1200)
    plt.show()
    '''