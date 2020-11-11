# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:55:42 2020

@author: surface
"""

#################弱监督CAM
import tensorflow as tf


x = tf.placeholder(tf.float32,[1, 224, 224, 3])#输入图片大小
y = tf.placeholder(tf.float32,[1, 10])#图像级别的标签

############主干网络
layer1=tf.layers.conv2d(x,40,3,strides=1, padding='same')
layer1=tf.nn.relu(layer1)
print(layer1)
layer1=tf.layers.conv2d(layer1,40,3,strides=2, padding='same')
layer1=tf.nn.relu(layer1)
print(layer1)
layer1=tf.layers.conv2d(layer1,40,3,strides=2, padding='same')
layer1=tf.nn.relu(layer1)
print(layer1)
layer1=tf.layers.conv2d(layer1,40,3,strides=1, padding='same')
layer1=tf.nn.relu(layer1)
print(layer1)
layer1=tf.layers.conv2d(layer1,40,3,strides=2, padding='same')
layer1=tf.nn.relu(layer1)
print(layer1)


###########分类
layer1_transpose=tf.layers.conv2d_transpose(layer1,10,3,strides=8, padding='same')
layer1_transpose=tf.nn.relu(layer1_transpose)
print(layer1_transpose)
GAP=tf.keras.layers.GlobalAvgPool2D()(layer1_transpose)
print(GAP)

Weight=tf.Variable(tf.zeros([10, 10]))
print(Weight)
result=tf.matmul(GAP,Weight)
print(result)

##########CAM
idx=tf.argmax(result,1)
print(idx)
idx=idx[0]
Weight_choose=Weight[idx]
print(Weight_choose)
layer1_transpose=layer1_transpose*Weight_choose
print(layer1_transpose)
CAM=tf.reduce_sum(layer1_transpose,3)
print(CAM)







