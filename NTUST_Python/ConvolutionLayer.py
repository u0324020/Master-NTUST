# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

with tf.Session() as sess:
    img =tf.constant([[[[1],[2],[0],[0]],
                       [[3],[0],[0],[1]],
                       [[0],[0],[0],[3]],
                       [[1],[0],[0],[1]]]],tf.float32)
    # shape of img: [batch, in_height, in_width, in_channels]

    filter_ = tf.constant([[[[1]],[[2]]],
                           [[[3]],[[0]]]],tf.float32)
    # shape of filter: [filter_height, filter_width, in_channels, out_channels]

    conv_strides = (1,1)
    padding_method = 'VALID'

    conv = tf.nn.conv2d(img, filter_, 
                        strides=[1,conv_strides[0],conv_strides[1],1], 
                        padding=padding_method)
    pooling = tf.nn.max_pool(conv,
                    ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    print("Shape of conv:")
    print(conv.eval().shape)
    print("Conv:")
    print(conv.eval())
    print("Shape of pooling:")
    print(pooling.eval().shape)
    print("Pooling:")
    print(pooling.eval())