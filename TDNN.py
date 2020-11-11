# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:45:39 2019

@author: surface
"""
import tensorflow as tf 


length=None#帧长
input=39#MFCC特征维数

x = tf.placeholder(tf.float32,[1,length,input])#输入数据
print(x)
conv1=tf.layers.conv1d(x,32,3,strides=1, padding='valid')#第一层TDNN
print(conv1)
conv2=tf.layers.conv1d(conv1,64,3,strides=2, padding='valid')#第二层TDNN
print(conv2)
mean, variance = tf.nn.moments(conv2, axes=1)
print(mean)
print(variance)