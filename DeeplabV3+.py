# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:07:47 2020

@author: ALW
"""

import tensorflow as tf


x = tf.placeholder(tf.float32,[None, 500, 500, 3])#输入图片大小


def Encoder(x, rate1, rate2, rate3, rate4, channel):
    
    layer1_1=tf.layers.conv2d(x,channel,1,strides=1, padding='same')
    layer1_2=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate2)
    layer1_3=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate3)
    layer1_4=tf.layers.conv2d(x,channel,3,strides=1, padding='same',dilation_rate=rate4)
    layer1_5=tf.layers.max_pooling2d(x, 3, 1, padding='same')
         
    encoder_output=tf.concat([layer1_1,layer1_2,layer1_3,layer1_4,layer1_5],-1)
    encoder_output=tf.layers.conv2d(encoder_output,channel,1,strides=1, padding='same')
    
    return encoder_output  


def DeeplabV3_(x, y, rate1, rate2, rate3, rate4, channel1, channel2):
    input1=Encoder(x, rate1, rate2, rate3, rate4, channel1)
    input1=tf.layers.conv2d_transpose(input1,channel2,3,strides=4, padding='same')
    input2=tf.layers.conv2d(y,channel2,1,strides=1, padding='same')
    decoder_output=tf.concat([input1, input2],-1)
    decoder_output=tf.layers.conv2d(decoder_output, channel2, 3, strides=1, padding='same')
    return decoder_output    
    
layer1=tf.nn.relu(tf.layers.conv2d(x,10,3,strides=1, padding='same'))
layer2=tf.nn.relu(tf.layers.conv2d(layer1,10,3,strides=4, padding='same'))
layer3=Encoder(layer2,2,4,6,8,10)
print(layer3)
layer4=DeeplabV3_(layer2,layer1,2,4,6,8,10,20)
print(layer4)