# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:52:43 2020

@author: surface
"""

import tensorflow as tf 
import math


def ResNet(x,kernel_size,channel):
    layer1=tf.layers.conv2d(x,channel,kernel_size,strides=1, padding='same')
    layer1=tf.contrib.layers.batch_norm(layer1)
    layer1=tf.nn.relu(layer1)
    layer1=tf.layers.conv2d(layer1,channel,kernel_size,strides=1, padding='same')
    layer1=tf.contrib.layers.batch_norm(layer1)
    layer1=tf.nn.relu(layer1)
    result=tf.nn.relu(tf.layers.batch_normalization(layer1+x))
    return result

def position_attention(x,H,W,C):
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V=tf.reshape(V,[-1,H*W,C])
    K=tf.transpose(K,[0,2,1])
    result=tf.matmul(Q,K)
    result=tf.nn.softmax(result)
    V=tf.matmul(result,V)
    V=tf.reshape(V,[-1,H,W,C])
    return V
    
def channel_attention(x,H,W,C):
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V=tf.reshape(V,[-1,H*W,C])
    K=tf.transpose(K,[0,2,1])
    result=tf.matmul(K,Q)
    result=tf.nn.softmax(result)
    V=tf.matmul(V,result)
    V=tf.reshape(V,[-1,H,W,C])
    return V



x = tf.placeholder(tf.float32,[None, 224, 224, 3])#输入图片大小
x1=tf.layers.conv2d(x,40,3,strides=1, padding='same')
Res1=ResNet(x1,3,40)
pos=position_attention(x1,224,224,40)
cha=channel_attention(x1,224,224,40)
dual=pos+cha

