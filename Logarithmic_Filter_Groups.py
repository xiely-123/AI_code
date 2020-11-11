# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:46:12 2020

@author: surface
"""

import tensorflow as tf 


length=50#帧长
input=39#MFCC特征维数



def Logarithmic_Filter_Groups(x,kernel_size,outchannel):
    split0,split1=tf.split(x ,2 ,2)
    outchannel1=outchannel/2
    print(split0,split1)
    split0=tf.layers.conv1d(split0,outchannel1,kernel_size,strides=1, padding='same')
    split1_0,split1_1=tf.split(split1 ,2 ,2)
    outchannel2=outchannel1/2
    print(split1_0,split1_1)
    split1_0=tf.layers.conv1d(split1_0,outchannel2,kernel_size,strides=1, padding='same')
    split1_1_0,split1_1_1=tf.split(split1_1 ,2 ,2)
    outchannel3=outchannel2/2
    print(split1_1_0,split1_1_1)
    split1_1_0=tf.layers.conv1d(split1_1_0,outchannel3,kernel_size,strides=1, padding='same')    
    split1_1_1=tf.layers.conv1d(split1_1_1,outchannel3,kernel_size,strides=1, padding='same')
    output=tf.concat( [split0, split1_0, split1_1_0, split1_1_1],-1)
    return output
x = tf.placeholder(tf.float32,[None,length,input])#输入数据
conv1=tf.layers.conv1d(x,40,3,strides=1, padding='same')#第一层TDNN
log_layer1=Logarithmic_Filter_Groups(conv1,9,40)