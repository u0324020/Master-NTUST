
from imblearn.over_sampling import SMOTE
import numpy as np
np.random.seed(1337)  # for reproducibility
from numpy import loadtxt
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


def Smote_upsampling(train_X, train_y):
    print(Counter(train_y))
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(train_X, train_y)
    print(Counter(y_smo))
    return (X_smo, y_smo)


if __name__ == '__main__':
    train = loadtxt(
        "C:/Users/Jane/Desktop/NTU/Scam/Code/0220-Imbalanced-testing.csv", delimiter=",")
    # np.random.shuffle(train)
    X = train[:, 0:44]  # 1:1236
    y = train[:, 44]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25)
    #x_train, y_train = Smote_upsampling(x_train, y_train)
    # data pre-processing
    x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
    x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    print(x_train.shape)
    print(x_test.shape)

    # in order to plot in a 2D figure
    encoding_dim = 3

    # this is our input placeholder
    input_img = Input(shape=(44,))

    # encoder layers
    encoded = Dense(36, activation='relu')(input_img)
    encoded = Dense(24, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(24, activation='relu')(decoded)
    decoded = Dense(36, activation='relu')(decoded)
    decoded = Dense(44, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(inputs=input_img, outputs=decoded)

    # construct the encoder model for plotting
    encoder = Model(inputs=input_img, outputs=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=256,
                    shuffle=True)

    # plotting
    encoded_imgs = encoder.predict(x_test)
    # plt.scatter(encoded_imgs[:, 1], encoded_imgs[:, 0], c=y_test)
    # plt.colorbar()
    # plt.show()
    # ======3D========
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax = Axes3D(fig)
    ax2 = Axes3D(fig2)
    ax3 = Axes3D(fig3)
    # ===========
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # overlap
    Encoder_Scams_0 = []
    Encoder_Scams_1 = []
    Encoder_Scams_2 = []
    Encoder_Malicious_0 = []
    Encoder_Malicious_1 = []
    Encoder_Malicious_2 = []
    #limited = 23093
    #Half_limited = 11546
    #=====array======
    for i in range(len(encoded_imgs)):
        if y_test[i] == 1:
            Encoder_Scams_0.append(encoded_imgs[i, 0])
            Encoder_Scams_1.append((encoded_imgs[i, 1]))
            Encoder_Scams_2.append((encoded_imgs[i, 2]))
        elif y_test[i] == 0:
            Encoder_Malicious_0.append(encoded_imgs[i, 0])
            Encoder_Malicious_1.append(encoded_imgs[i, 1])
            Encoder_Malicious_2.append(encoded_imgs[i, 2])
    #====2D=====
    # plt_1 = plt.scatter(Encoder_Malicious_0, Encoder_Malicious_1,
    #                     c='b', marker='o', s=40, cmap=plt.cm.Spectral)
    # plt_2 = plt.scatter(Encoder_Scams_0, Encoder_Scams_1,
    #                     c='r', marker='x', s=50, cmap=plt.cm.Spectral)
    # # X,Y
    # plt.show()
    #====3D=====
    ax.scatter(Encoder_Scams_0[:1500], Encoder_Scams_1[:1500], Encoder_Scams_2[:1500], c='red', cmap="brg", s=100,
               edgecolor='k', marker='x', label='Scam')
    ax2.scatter(Encoder_Malicious_0[:1500], Encoder_Malicious_1[:1500], Encoder_Malicious_2[:1500],
                c='blue', cmap="brg", s=40, alpha=0.3, edgecolor='k', marker='o', label='Malicious')
    ax3.scatter(Encoder_Scams_0[:1500], Encoder_Scams_1[:1500], Encoder_Scams_2[:1500], c='red', cmap="brg", s=100,
                edgecolor='k', marker='x', label='Scam')
    ax3.scatter(Encoder_Malicious_0[:1500], Encoder_Malicious_1[:1500], Encoder_Malicious_2[:1500],
                c='blue', cmap="brg", s=40, alpha=0.3, edgecolor='k', marker='o', label='Malicious')
    ax.view_init(elev=45, azim=135)
    ax.set_xlabel('AutoEncoder-1')
    ax.set_ylabel('AutoEncoder-2')
    ax.set_zlabel('AutoEncoder-3')
    plt.legend()
    plt.show()
    ax2.set_xlabel('AutoEncoder-1')
    ax2.set_ylabel('AutoEncoder-2')
    ax2.set_zlabel('AutoEncoder-3')
    plt.legend()
    plt.show()
    ax3.set_xlabel('AutoEncoder-1')
    ax3.set_ylabel('AutoEncoder-2')
    ax3.set_zlabel('AutoEncoder-3')
    plt.legend()
    plt.show()
