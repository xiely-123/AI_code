# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:27:54 2019

@author: surface
"""

import tensorflow as tf 


length=50#帧长
input=39#MFCC特征维数

def memory_block(x,front,delay,input_size, output_size):
    #############pad 0
    x_pad_front=tf.pad(x,[[0,0],[front,0],[0,0]],"CONSTANT")
    x_pad_delay=tf.pad(x,[[0,0],[0,delay],[0,0]],"CONSTANT")
    h_memory=tf.random_normal([-1,1,39], 0)
    print(h_memory)
    print(x_pad_front)
    print(x_pad_delay)
    #############sum（a*h）+sum(c*h)
    for step in range(input_size):
        memory_block_front=x_pad_front[:,step:step+front,:]
        #print(memory_block_front)
        memory_block_delay=x_pad_front[:,step:step+delay,:]
        #print(memory_block_delay)
        FIR_1=tf.layers.conv1d(memory_block_front,39,1,strides=1, padding='same')
        FIR_1 = tf.reduce_sum(FIR_1, 1)  
        FIR_2=tf.layers.conv1d(memory_block_delay,39,1,strides=1, padding='same')
        FIR_2 = tf.reduce_sum(FIR_2, 1)  
        FIR = FIR_1+FIR_2
        FIR = tf.reshape(FIR,[-1,1,39])
        h_memory=tf.concat([h_memory,FIR],1)
    h_memory=h_memory[:,1:,:]    
    print(h_memory)
    ############ all
    h_memory=tf.layers.conv1d(h_memory,output_size,1,strides=1, padding='same')
    x=tf.layers.conv1d(x,output_size,1,strides=1, padding='same')
    h_next=h_memory+x
    h_next=tf.layers.conv1d(h_next,output_size,1,strides=1, padding='same')
    print(h_next)
    return h_next
    
    
    

x = tf.placeholder(tf.float32,[None,length,input])#输入数据
FSMN1=memory_block(x,10,15,50,50)