import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from pylab import *
import numpy as np
import csv
from tensorflow.contrib import rnn
import os

# 標準化，資料平均值變為0，標準差變為1
sc = StandardScaler()

# Parameters
learning_rate = 0.05  # 0.01 this learning rate will be better! Tested
epoch = 0
stopping_step = 0
# 要改
Stop_Threshold = 500 #2000
batch_size = 256
display_step = 1
Input_node = 39

# label_no = 20
# batch_size = 50

n_hidden_1 = 39
#n_hidden_2 = 15
Output_node = 39

Hidden_stddev = 0.1
Dropout_keep_prob = 0.8

# 存放作圖時的x軸及y軸
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

X_no_sc_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/autoencoder/dataset (original).csv", delimiter=',',dtype="float64")

Y_normal_original = np.genfromtxt("C:/Users/Jane/Desktop/NTUST-A/中正/autoencoder/dataset_label.csv",delimiter=',', dtype="float64")  # y

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

# 定義會話
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 把所有圖片訓練21次
    fig_x, fig_tranning_accuracy, fig_validation_accuracy, fig_test_accuracy, fig_validation_loss, fig_test_loss, fig_tranning_loss = [], [], [], [], [], [], []
    while True:
        fig_x.append(epoch)
   
        _, c = sess.run([train_step, loss], feed_dict={x: X_auto})

        if epoch % display_step == 0:
            #            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
            print("Epoch:" + str(epoch), "loss=", "{:.9f}".format(c))

        encoder_result = sess.run(encoder_op, feed_dict={x: X_auto})
        decoder_result = sess.run(decoder_op, feed_dict={x: X_auto})
        # 訓練
        training_loss = sess.run(loss, feed_dict={x: X_auto})
        fig_tranning_loss.append(training_loss)
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

    fig_x = np.array(fig_x)
    fig_tranning_accuracy = np.array(fig_tranning_accuracy)
    fig_validation_accuracy = np.array(fig_validation_accuracy)
    fig_test_accuracy = np.array(fig_test_accuracy)
    fig_tranning_loss = np.array(fig_tranning_loss)
    fig_validation_loss = np.array(fig_validation_loss)
    fig_test_loss = np.array(fig_test_loss)
    print(fig_x.shape)
    print(decoder_result.shape)
    csv_encoder_result = encoder_result.tolist()
    csv_decoder_result = decoder_result.tolist()
    with open('csv_encoder_result5.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_encoder_result)

    plot3d = plt.subplot(projection='polar')

    for i in range(len(Y_normal_original)):
        if (Y_normal_original[i] == 0):
            data_0 = plt.scatter(encoder_result[i][7], encoder_result[i][8], s=10, c='r',
                                    marker='+', alpha=0.5)
        elif (Y_normal_original[i] == 1):
            data_1 = plt.scatter(encoder_result[i][7], encoder_result[i][8], s=10, c='g',
                                    marker='o', alpha=0.5)
    plot3d.legend(
        [data_0, data_1],
        ['0', '1'])
    plot3d.set_title('DDoS_Autoencoder')
    plot3d.set_zlabel('Z')  # 坐标轴
    plot3d.set_ylabel('Y')
    plot3d.set_xlabel('X')
    plt.savefig('DDoS_Autoencoder', dpi=1200)
    plt.show()
    plt.close()

        # anova過後是7、8、17
        # list2 = [0, 1, 31, 42, 50, 169, 174, 188, 189, 193, 194, 203, 205, 221, 222, 228, 231, 232, 242, 339]
    # 畫test_loss圖
    plt.plot(fig_x[0:epoch], fig_tranning_loss[0:epoch], '-b', label='Tranning')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.savefig('test_loss5_20label_Adam.png', dpi=800)
    plt.show()
    plt.close()


    '''
 #   list_b = ('0', '1', '31', '42', '50', '169', '174', '188', '189', '193', '194', '203', '205', '221', '222', '228', '231','232', '242', '339')
    plt.figure(figsize=(10, 10))
    plt.scatter(decoder_result[:, 7], decoder_result[:, 8], s=10, c=Y_normal_original, alpha=0.5)

    plt.colorbar()
    plt.savefig('test_distri5_20label_Adam.png', dpi=1200)
    plt.show()
    '''



