# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:34:42 2020

@author: surface
"""

import tensorflow as tf


x = tf.placeholder(tf.float32,[None, 224, 224, 3])#输入图片大小


def FarSeg(x,channel):
    #########################Multi-Branch Encoder
    C2=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(x,channel,3,strides=2, padding='same')))
    C3=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(C2,channel,3,strides=2, padding='same')))
    C4=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(C3,channel,3,strides=2, padding='same')))
    C5=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(C4,channel,3,strides=2, padding='same')))
    P5=tf.layers.conv2d(C5,channel,1,strides=1, padding='same')
    P4=tf.layers.conv2d(C4,channel,1,strides=1, padding='same')+tf.layers.conv2d_transpose(P5,channel,3,strides=2, padding='same')
    P3=tf.layers.conv2d(C3,channel,1,strides=1, padding='same')+tf.layers.conv2d_transpose(P4,channel,3,strides=2, padding='same')
    P2=tf.layers.conv2d(C2,channel,1,strides=1, padding='same')+tf.layers.conv2d_transpose(P3,channel,3,strides=2, padding='same') 
    C6=tf.keras.layers.GlobalAvgPool2D()(C5)
    #########################Foreground-Scene Relation   
    C6=tf.reshape(C6,[-1,1,1,channel])
    U=tf.layers.conv2d(C6,channel,1,strides=1, padding='same')
    V5=tf.nn.relu(tf.layers.conv2d(P5,channel,1,strides=1, padding='same'))
    V4=tf.nn.relu(tf.layers.conv2d(P4,channel,1,strides=1, padding='same'))
    V3=tf.nn.relu(tf.layers.conv2d(P3,channel,1,strides=1, padding='same'))
    V2=tf.nn.relu(tf.layers.conv2d(P2,channel,1,strides=1, padding='same'))
    R5=V5*U
    R4=V4*U
    R3=V3*U
    R2=V2*U
    Z2=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(V2,channel,1,strides=1, padding='same')))*tf.nn.sigmoid(R2)
    Z3=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(V3,channel,1,strides=1, padding='same')))*tf.nn.sigmoid(R3)
    Z4=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(V4,channel,1,strides=1, padding='same')))*tf.nn.sigmoid(R4)
    Z5=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(V5,channel,1,strides=1, padding='same')))*tf.nn.sigmoid(R5)
    print(Z2,Z3,Z4,Z5)
    #########################Light-weight Decoder       
    Z3=tf.layers.conv2d_transpose(tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(Z3,channel,3,strides=1, padding='same'))),channel,3,strides=2, padding='same')
    Z4=tf.layers.conv2d_transpose(tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(Z4,channel,3,strides=1, padding='same'))),channel,3,strides=4, padding='same')
    Z5=tf.layers.conv2d_transpose(tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(Z5,channel,3,strides=1, padding='same'))),channel,3,strides=8, padding='same')
    Z=Z2+Z3+Z4+Z5
    print(Z)
FarSeg(x,256)

























