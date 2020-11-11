# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:22:52 2020

@author: surface
"""

#################Sknet

import tensorflow as tf


x = tf.placeholder(tf.float32,[None, 224, 224, 3])#输入图片大小

def SK_block(x,kernel1,kernel2,channel):
    ############Spilt
    U1=tf.layers.conv2d(x,channel,kernel1,strides=1, padding='same')
    U2=tf.layers.conv2d(x,channel,kernel2,strides=1, padding='same')
    ############Fuse    
    U=U1+U2
    S=tf.keras.layers.GlobalAvgPool2D()(U)
    print(S)
    S=tf.reshape(S,[-1,1,1,channel])
    print(S)
    Z=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(S,32,1,strides=1, padding='same'),axis=-1,momentum=0.99,epsilon=0.001, center=True, scale=True,))
    print(Z)
    a=tf.layers.conv2d(Z,channel,1,strides=1, padding='same')
    b=tf.layers.conv2d(Z,channel,1,strides=1, padding='same')
    print(a,b)
    combine=tf.concat([a,b],1)
    print(combine)
    combine=tf.nn.softmax(combine,axis=1)
    print(combine)
    a,b=tf.split(combine,num_or_size_splits=2, axis=1)
    print(a,b)
    V=a*U1+b*U2
    print(V)
    return V
    
    
    
    
    
    
    
layer1=tf.layers.conv2d(x,256,3,strides=1, padding='same')
layer1=tf.nn.relu(layer1)
SK_block(layer1,3,5,256)