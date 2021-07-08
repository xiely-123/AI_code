import numpy as np


data = np.loadtxt("./data_label.txt", dtype=str, delimiter=',')
train=open('train.txt',mode='w')
test=open('test.txt',mode='w')
import random
Test_10 = random.sample(range(0, 99), 20)
Test_8 = random.sample(range(100, 199), 20)
Test_7 = random.sample(range(200, 299), 20)
Test_9 = random.sample(range(300, 399), 20)
Test_11 = random.sample(range(400, 499), 20)

Test_list = Test_10+Test_8+Test_7+Test_9+Test_11

print(Test_list)
for i in range(len(data)):
    if i in Test_list:
       print(data[i],"test")
       test.write(data[i][0]+","+data[i][1]+' \n')
    else:
       print(data[i],"train")
       train.write(data[i][0]+","+data[i][1]+' \n')

train.close()
test.close()