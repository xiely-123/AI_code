# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:20:59 2020

@author: surface
"""
import tensorflow as tf 
import math


length=50#帧长
input=39#MFCC特征维数
###########输入数据
x = tf.placeholder(tf.float32,[None,length,input])#输入数据


def self_attention(x,hidden_layer):
    x=tf.layers.conv1d(x,hidden_layer*3,1,strides=1, padding='same')
    Q,K,V=tf.split(x, 3, axis=2)
    K=tf.transpose(K,[0,2,1])
    result=tf.reduce_sum(tf.matmul(Q,K)/math.sqrt(hidden_layer),axis=1)
    result=tf.reshape(result,[-1,50,1])
    result=tf.nn.softmax(result)
    V=V*result
    return V

def multi_head_attention(x,head,output_channel):
    xn=tf.split(x,head,axis=2)
    print(xn)
    V1=xn[0]
    print(V1)
    V1=self_attention(V1,32)
    for a in xn[1:]:
        V=self_attention(a,32)
        V1=tf.concat([V1,V],axis=2)
    print(V1)   
    V1=tf.layers.conv1d(V1,output_channel,1,strides=1, padding='same')
    return V1
x=tf.layers.conv1d(x,100,1,strides=1, padding='same')
multi_head_attention(x,5,100)   
