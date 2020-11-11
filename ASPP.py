# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:17:01 2020

@author: ALW
"""

#################ASPP

import tensorflow as tf


x = tf.placeholder(tf.float32,[None, 500, 500, 3])#输入图片大小

def ASPP(x, rate1, rate2, rate3, rate4, channel):
    ##第一层
    layer1_1=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate1)
    layer1_2=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate2)
    layer1_3=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate3)
    layer1_4=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate4)
    ##第二层
    layer1_1=tf.layers.conv2d(layer1_1,channel,1,strides=1, padding='same')
    layer1_2=tf.layers.conv2d(layer1_2,channel,1,strides=1, padding='same')
    layer1_3=tf.layers.conv2d(layer1_3,channel,1,strides=1, padding='same')
    layer1_4=tf.layers.conv2d(layer1_4,channel,1,strides=1, padding='same')    
    #第三层
    layer1_1=tf.layers.conv2d(layer1_1,channel,1,strides=1, padding='same')
    layer1_2=tf.layers.conv2d(layer1_2,channel,1,strides=1, padding='same')
    layer1_3=tf.layers.conv2d(layer1_3,channel,1,strides=1, padding='same')
    layer1_4=tf.layers.conv2d(layer1_4,channel,1,strides=1, padding='same')      
    
    output=layer1_1+layer1_2+layer1_3+layer1_4
    return output
    
    
    
    
layer1=tf.layers.conv2d(x,256,3,strides=1, padding='same')
layer1=tf.nn.relu(layer1)
layer2=ASPP(layer1,2,4,6,8,256)
print(layer2)