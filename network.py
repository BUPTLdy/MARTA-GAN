#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:52:00 2017

@author: ldy
"""


import numpy as np

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from utils import *



def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 256
    k = 4
    # 128, 64, 32, 16
    s2, s4, s8, s16, s32, s64 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16), int(image_size/32), int(image_size/64)

    batch_size = 64
    gf_dim = 16 # Dimension of gen filters in first conv layer. [64]

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*32*s64*s64, W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s64, s64, gf_dim*32], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*16, (k, k), out_size=(s32, s32), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*8, (k, k), out_size=(s16, s16), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim*4, (k, k), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, gf_dim*2, (k, k), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h4/batch_norm')

        net_h5 = DeConv2d(net_h4, gf_dim*1, (k, k), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h5/decon2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h5/batch_norm')

        net_h6 = DeConv2d(net_h5, 3, (k, k), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h6/decon2d')
        logits = net_h6.outputs
        net_h6.outputs = tf.nn.tanh(net_h6.outputs)
    return net_h6, logits

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    k = 5
    df_dim = 16 # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (k, k), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (k, k), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (k, k), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (k, k), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')
        
        global_max1 = MaxPool2d(net_h3, filter_size=(4, 4), strides=None, padding='SAME', name='maxpool1')
        global_max1 = FlattenLayer(global_max1, name='d/h3/flatten')
        
        net_h4 = Conv2d(net_h3, df_dim*16, (k, k), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h4/batch_norm')
        
        global_max2 = MaxPool2d(net_h4, filter_size=(2, 2), strides=None, padding='SAME', name='maxpool2')
        global_max2 = FlattenLayer(global_max2, name='d/h4/flatten')
        
        net_h5 = Conv2d(net_h4, df_dim*32, (k, k), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h5/batch_norm')
        global_max3 = FlattenLayer(net_h5, name='d/h5/flatten')
        
        feature = ConcatLayer(layer = [global_max1, global_max2, global_max3], name ='d/concat_layer1')
        net_h6 = DenseLayer(feature, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h6/lin_sigmoid')
        logits = net_h6.outputs
        net_h6.outputs = tf.nn.sigmoid(net_h6.outputs)
    return net_h6, logits, feature.outputs