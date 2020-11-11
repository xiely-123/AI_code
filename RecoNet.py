# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:14:37 2020

@author: surface
"""

#################RecoNet

import tensorflow as tf


x = tf.placeholder(tf.float32,[None, 224, 224, 3])#输入图片大小


def TGM_TRM(x,Rank):
    x_height=tf.transpose(x,(0,3,2,1))
    print(x_height)
    x_width=tf.transpose(x,(0,1,3,2))
    print(x_width)
    x_channel=x
    ######################TGM
    height_pooling=tf.keras.layers.GlobalAvgPool2D()(x_height)
    width_pooling=tf.keras.layers.GlobalAvgPool2D()(x_width)
    channel_pooling=tf.keras.layers.GlobalAvgPool2D()(x_channel)

    height_pooling=tf.reshape(height_pooling,[-1,height_pooling.get_shape().as_list()[1],1,1])
    width_pooling=tf.reshape(width_pooling,[-1,1,width_pooling.get_shape().as_list()[1],1])
    channel_pooling=tf.reshape(channel_pooling,[-1,1,1,channel_pooling.get_shape().as_list()[1]])
    
    height_feature=tf.sigmoid(tf.layers.conv2d(height_pooling,Rank,1,strides=1, padding='same'))
    width_feature=tf.sigmoid(tf.layers.conv2d(width_pooling,Rank,1,strides=1, padding='same'))
    channel_feature=tf.sigmoid(tf.layers.conv2d(channel_pooling,Rank*channel_pooling.get_shape().as_list()[-1],1,strides=1, padding='same'))    
    
    #print(height_feature,width_feature,channel_feature)     
    ######################TRM
    height_pooling=tf.keras.layers.GlobalAvgPool2D()(x_height)
    width_pooling=tf.keras.layers.GlobalAvgPool2D()(x_width)
    channel_pooling=tf.keras.layers.GlobalAvgPool2D()(x_channel)

    height_pooling=tf.reshape(height_pooling,[-1,height_pooling.get_shape().as_list()[1],1,1])
    width_pooling=tf.reshape(width_pooling,[-1,1,width_pooling.get_shape().as_list()[1],1])
    channel_pooling=tf.reshape(channel_pooling,[-1,1,1,channel_pooling.get_shape().as_list()[1]])
    
    height_feature=tf.sigmoid(tf.layers.conv2d(height_pooling,Rank,1,strides=1, padding='same'))
    width_feature=tf.sigmoid(tf.layers.conv2d(width_pooling,Rank,1,strides=1, padding='same'))
    channel_feature=tf.sigmoid(tf.layers.conv2d(channel_pooling,Rank*channel_pooling.get_shape().as_list()[-1],1,strides=1, padding='same'))
        
        
    
    
    
    
    
    
    
    
TGM_TRM(x,60)


print("aaaaaa")