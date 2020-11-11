# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:14:19 2020

@author: surface
"""

import tensorflow as tf 
x = tf.placeholder(tf.float32,[None, 224, 224, 3])#输入图片大小


c=tf.layers.conv2d(x,40,3,strides=1, padding='same')
b=tf.layers.conv2d(c,40,3,strides=2, padding='same')
a=tf.layers.conv2d(b,40,3,strides=2, padding='same')


def ST(x,H,W,C,N): 
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V=tf.reshape(V,[-1,H*W,C])
    K_trans=tf.transpose(K,[0,2,1])
    result_all=tf.Variable(tf.zeros([1,H*W,H*W]))
    for a in range(N):
        WT=tf.Variable(tf.random_normal([H*W]))
        k_mean,K_var=tf.nn.moments(K[:,:,int(C/N)*(a):int(C/N)*(a+1)],2)
        result=tf.matmul(Q[:,:,int(C/N)*(a):int(C/N)*(a+1)],K_trans[:,int(C/N)*(a):int(C/N)*(a+1),:])        
        result_all=result_all+tf.nn.softmax(k_mean*WT)*tf.nn.softmax(result)
        print(result_all)
    V=tf.matmul(result_all,V)
    V=tf.reshape(V,[-1,H,W,C])
    return V
def GT(x,y,BZ,H,W,C,Expand,N): 
    x=tf.layers.conv2d_transpose(x,C*2,1,strides=Expand, padding='same')
    K=tf.layers.conv2d(y,C,1,strides=1, padding='same')
    Q,V=tf.split(x, 2, axis=3)
    Q=tf.reshape(Q,[BZ,H*W,C])
    K=tf.reshape(K,[BZ,H*W,C])
    V=tf.reshape(V,[BZ,H*W,C])
    print(Q,K,V)
    result_all=tf.zeros([BZ,H*W,1])
    for a in range(N):
        WT=tf.Variable(tf.random_normal([H*W]))
        k_mean,K_var=tf.nn.moments(K[:,:,int(C/N)*(a):int(C/N)*(a+1)],2)
        result=-tf.square(Q[:,:,int(C/N)*(a):int(C/N)*(a+1)]-K[:,:,int(C/N)*(a):int(C/N)*(a+1)])   
        result=tf.nn.softmax(tf.reshape(k_mean*WT,[-1,H*W,1]))*tf.nn.softmax(result)
        print(result)
        result_all=tf.concat([result_all,result],-1)
        print(result_all)
    V=result_all[:,:,1:]*V
    V=tf.reshape(V,[-1,H,W,C])
    print(V)
    #return V
def RT(x,y,H,W,H1,W1,C,narrow): #x:high-level feature map y:low-level feature map
    y=tf.layers.conv2d(y,C*2,1,strides=1, padding='same')
    K,V =tf.split(y, 2, axis=3)
    print(V)
    Q=tf.layers.conv2d(x,C,1,strides=1, padding='same')
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H1*W1,C])
    #
    w=tf.keras.layers.GlobalAvgPool1D()(K)
    Qatt=Q*tf.reshape(w,[-1,1,C])
    Vdow=tf.layers.conv2d(V,C,3,strides=narrow, padding='same')
    Qatt=tf.reshape(Qatt,[-1,H,W,C])
    Xc=tf.layers.conv2d(Qatt,C,3,strides=1, padding='same')+Vdow
    print(Xc)
    
    
    
    
GT(a,b,8,112,112,40,2,4)