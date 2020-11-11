# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:11:37 2019

@author: surface
"""

import tensorflow as tf

x = tf.placeholder(tf.float32,[1,None,None,1])
print(x)
conv1=tf.layers.conv2d(x,32,[3,3],strides=1, padding='SAME')
print(conv1)
conv2=tf.layers.conv2d(conv1,64,[3,3],strides=1, padding='SAME')
print(conv2)
globel_average_pooling=tf.reduce_mean(conv2,[1,2])
print(globel_average_pooling)