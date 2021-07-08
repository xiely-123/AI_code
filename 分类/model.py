# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:12:28 2020

@author: ALW
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding = 0),
        )
        self.BN_Relu = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = self.BN_Relu(x)
        return x

class ASPP(nn.Module):
    ###联合网络
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Encoder1 = nn.Sequential(
            oneConv(in_channels,out_channels,3,1,1),
            
        )
        self.Encoder2 = nn.Sequential(
            oneConv(in_channels,out_channels,5,4,2),
            
        )
        self.Encoder3 = nn.Sequential(
            oneConv(in_channels,out_channels,7,6,2),
            
        )
        self.Encoder4 = nn.Sequential(
            oneConv(in_channels,out_channels,3,4,4),
            
        )
        self.Encoder5 = nn.Sequential(
            nn.MaxPool1d(kernel_size = 5, stride=1,padding = 2),
            oneConv(in_channels,out_channels,1,0,1),
        )
        self.layer = oneConv(out_channels*5,out_channels,1,0,1)
    def forward(self, x):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x)
        x3 = self.Encoder3(x)
        x4 = self.Encoder4(x)
        x5 = self.Encoder5(x)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x_all = torch.cat((x1,x2,x3,x4,x5),1)
        x_all = self.layer(x_all)
        return x_all   


class SK_block(nn.Module):
    ###联合网络
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Encoder1 = nn.Sequential(
            oneConv(in_channels,out_channels,3,1,1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            
        )
        self.Encoder2 = nn.Sequential(
            oneConv(in_channels,out_channels,5,2,1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.Fgp = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            oneConv(out_channels,50,1,0,1),
            #nn.ReLU(inplace=True),
            oneConv(50,out_channels,1,0,1),
            #nn.ReLU(inplace=True),            
        )
        self.SE1 = oneConv(out_channels,out_channels,1,0,1)
        self.SE2 = oneConv(out_channels,out_channels,1,0,1)
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, x):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x)
        x_f = x1+x2
        #print(x_f.size())
        Fgp = self.Fgp(x_f)
        #print(Fgp.size())
        x_se = self.layer1(Fgp)
        #x_se = x_se.permute(0, 2, 1)
        
        #print(x_se.size())
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        x_se = torch.cat([x_se1,x_se2],2)
        #print(x_se.size())
        x_se = self.softmax(x_se)

        att_3 = torch.unsqueeze(x_se[:,:,0],2)
        att_5 = torch.unsqueeze(x_se[:,:,1],2)
        x1 = att_3*x1
        x2 = att_5*x2
        x_all = x1+x2
        return x_all   


class ResNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.M_layer1 = ResBlock(n_channels,20)
        self.pooling1 = nn.MaxPool1d(kernel_size = 3, stride=2,padding = 1)
        self.M_layer2 = ResBlock(20,40)
        self.pooling2 = nn.MaxPool1d(kernel_size = 5, stride=2,padding = 2)
        self.M_layer5 = ResBlock(40,80)
        self.fc = nn.Linear(4*80, n_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
    
         x = self.M_layer1(x)
         #print(x.size())
         x = self.pooling1(x)
         #print(x.size())
         x = self.M_layer2(x)
         x = self.pooling2(x)
         #print(x.size())
         x = self.M_layer5(x)
         #print(x.size())         
         out = x.view(-1, 4*80)
         out = self.fc(out)
         out = self.softmax(out)
         return out

class ASPP_net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ASPP_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.M_layer1 = oneConv(n_channels,256,3,1,1)
        self.pooling1 = nn.MaxPool1d(kernel_size = 3, stride=2,padding = 1)
        self.M_layer2 = ASPP(256,128)
        self.pooling2 = nn.MaxPool1d(kernel_size = 5, stride=2,padding = 2)
        self.M_layer3 = ASPP(128,64)
        self.fc = nn.Linear(25*64, n_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
    
         x = self.M_layer1(x)
         #print(x.size())
         x = self.pooling1(x)
         #print(x.size())
         x = self.M_layer2(x)
         x = self.pooling2(x)
         #print(x.size())
         x = self.M_layer3(x)
         #print(x.size())         
         out = x.view(-1, 25*64)
         out = self.fc(out)
         out = self.softmax(out)
         return out

class SKnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SKnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.M_layer1 = oneConv(n_channels,20,3,1,1)
        self.pooling1 = nn.MaxPool1d(kernel_size = 3, stride=2,padding = 1)
        self.M_layer2 = SK_block(20,40)
        self.pooling2 = nn.MaxPool1d(kernel_size = 5, stride=2,padding = 2)
        self.M_layer3 = SK_block(40,80)
        self.fc = nn.Linear(240, n_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
    
         x = self.M_layer1(x)
         #print(x.size())
         x = self.pooling1(x)
         #print(x.size())
         x = self.M_layer2(x)
         x = self.pooling2(x)
         #print(x.size())
         x = self.M_layer3(x)
         #print(x.size())         
         out = x.view(-1, 240)
         out = self.fc(out)
         out = self.softmax(out)
         #print(out.size())
         return out
   




"""
x=torch.randn(5, 128, 100)
net = SK_block(128,150)
output = net(x)
print(output.size())

"""

