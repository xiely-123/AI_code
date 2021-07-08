import numpy as np

def normalization(data):   ########def定义归一化函数，主要的目的：使模型更快收敛
    #_range = np.max(data) - np.min(data)
    #return (data - np.min(data)) / _range

    _range = np.max(abs(data))      ######找到图像中绝对值的最大值
    return data / _range    #######数据除以最大值，使数据在[-1,1]之间，让模型收敛更快，使训练效果更好

def batch_data_read(data_path):
    data_all = []
    labels = []
    for i in range(len(data_path)):
        data = np.loadtxt(data_path[i][0])
        #print(data.shape)
        label = int(data_path[i][1])
        #print(mel.shape)
        data_all.append(data)
        #label = data[i][1]
        labels.append(label)        
        #print(data.shape,label)
    return np.array(data_all), np.array(labels)

"""
data = np.loadtxt("./train.txt", dtype=str, delimiter=',')
a,b = batch_data_read(data)
print(a.shape,b.shape)
"""








