#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import os
import sys
import copy
import math
import numpy as np
import torch.nn.init as init


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx = idx.to(device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.k = 20  # Set the number of neighbors; adjust based on requirements
        self.conv1 = torch.nn.Conv2d(2*3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64*2, 64, 1)
        self.conv3 = torch.nn.Conv2d(64*2, 128, 1)
        self.conv4 = torch.nn.Conv2d(128*2, 256, 1)
        self.conv5 = torch.nn.Conv2d(256*2, 512, 1)
        self.conv6 = torch.nn.Conv2d(512*2, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)


    def forward(self, x):
        x = x.transpose(1,2)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = F.relu(self.bn1(self.conv1(x)))     # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = F.relu(self.bn2(self.conv2(x)))     # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = F.relu(self.bn3(self.conv3(x)))     # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x_128 = torch.unsqueeze(x3,3)           # (batch_size, 128, num_points) -> (batch_size, 128, num_points, 1)
        x_128 = torch.squeeze(self.maxpool(x_128),2)      # (batch_size, 128, num_points) -> (batch_size, 128, 1)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = F.relu(self.bn4(self.conv4(x)))     # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x_256 = torch.unsqueeze(x4,3)           # (batch_size, 256, num_points) -> (batch_size, 256, num_points, 1)
        x_256 = torch.squeeze(self.maxpool(x_256),2)      # (batch_size, 256, num_points) -> (batch_size, 256, 1)

        x = get_graph_feature(x4, k=self.k)     # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
        x = F.relu(self.bn5(self.conv5(x)))     # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
        x5 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)
        x_512 = torch.unsqueeze(x5,3)           # (batch_size, 512, num_points) -> (batch_size, 512, num_points, 1)
        x_512 = torch.squeeze(self.maxpool(x_512),2)      # (batch_size, 512, num_points) -> (batch_size, 512, 1)

        x = get_graph_feature(x5, k=self.k)     # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = F.relu(self.bn6(self.conv6(x)))     # (batch_size, 512*2, num_points, k) -> (batch_size, 1024, num_points, k)
        x6 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 1024, num_points, k) -> (batch_size, 1024, num_points)
        x_1024 = torch.unsqueeze(x6,3)           # (batch_size, 256, num_points) -> (batch_size, 256, num_points, 1)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)     # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)

        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


# class Convlayer(nn.Module):
#     def __init__(self,point_scales):
#         super(Convlayer,self).__init__()
#         self.point_scales = point_scales
#         self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))  # 1 input channel, 64 output channels, kernel size (1, 3)
#         self.conv2 = torch.nn.Conv2d(64, 64, 1)  # 64 input channels, 64 output channels, kernel size 1
#         self.conv3 = torch.nn.Conv2d(64, 128, 1)  # 64 input channels, 128 output channels, kernel size 1
#         self.conv4 = torch.nn.Conv2d(128, 256, 1)  # 128 input channels, 256 output channels, kernel size 1
#         self.conv5 = torch.nn.Conv2d(256, 512, 1)  # 256 input channels, 512 output channels, kernel size 1
#         self.conv6 = torch.nn.Conv2d(512, 1024, 1)  # 512 input channels, 1024 output channels, kernel size 1
#         self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)  # Max pooling with kernel size (self.point_scales, 1)
#         self.bn1 = nn.BatchNorm2d(64)  # Batch normalization with 64 channels
#         self.bn2 = nn.BatchNorm2d(64)  # Batch normalization with 64 channels
#         self.bn3 = nn.BatchNorm2d(128)  # Batch normalization with 128 channels
#         self.bn4 = nn.BatchNorm2d(256)  # Batch normalization with 256 channels
#         self.bn5 = nn.BatchNorm2d(512)  # Batch normalization with 512 channels
#         self.bn6 = nn.BatchNorm2d(1024)  # Batch normalization with 1024 channels
    
#     def forward(self,x):
#         x = torch.unsqueeze(x,1)  # Add a dimension of size 1 at index 1
#         x = F.relu(self.bn1(self.conv1(x)))  # Apply convolution, batch normalization, and ReLU activation
#         x = F.relu(self.bn2(self.conv2(x)))  # Apply convolution, batch normalization, and ReLU activation
#         x_128 = F.relu(self.bn3(self.conv3(x)))  # Apply convolution, batch normalization, and ReLU activation
#         x_256 = F.relu(self.bn4(self.conv4(x_128)))  # Apply convolution, batch normalization, and ReLU activation
#         x_512 = F.relu(self.bn5(self.conv5(x_256)))  # Apply convolution, batch normalization, and ReLU activation
#         x_1024 = F.relu(self.bn6(self.conv6(x_512)))  # Apply convolution, batch normalization, and ReLU activation
#         x_128 = torch.squeeze(self.maxpool(x_128),2)  # Apply max pooling and remove the dimension of size 2
#         x_256 = torch.squeeze(self.maxpool(x_256),2)  # Apply max pooling and remove the dimension of size 2
#         x_512 = torch.squeeze(self.maxpool(x_512),2)  # Apply max pooling and remove the dimension of size 2
#         x_1024 = torch.squeeze(self.maxpool(x_1024),2)  # Apply max pooling and remove the dimension of size 2
#         L = [x_1024,x_512,x_256,x_128]  # Create a list of tensors
#         x = torch.cat(L,1)  # Concatenate tensors along dimension 1
#         return x

class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs,2)
        latentfeature = latentfeature.transpose(1,2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature,1)

        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class _netG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        self.fc1_1 = nn.Linear(1024,256*512)   # 128 256 形式
        self.fc2_1 = nn.Linear(512,128*256)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,128*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/256),1)  # 第二层的个数改成
        self.conv2_1 = torch.nn.Conv1d(256,6,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        
    def forward(self,x):
        x = self.latentfeature(x)
        x_1 = F.relu(self.fc1(x)) #1024
        x_2 = F.relu(self.fc2(x_1)) #512
        x_3 = F.relu(self.fc3(x_2))  #256
        
        
        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1,128,3) #64x3 center1
        
        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1,256,128)
        pc2_xyz =self.conv2_1(pc2_feat) #6x64 center2
        
        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1,512,256)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat) #12x128 fine 
        
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)
        pc2_xyz = pc2_xyz.transpose(1,2)
        pc2_xyz = pc2_xyz.reshape(-1,128,2,3)
        pc2_xyz = pc1_xyz_expand+pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1,256,3) 
        
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)
        pc3_xyz = pc3_xyz.transpose(1,2)
        pc3_xyz = pc3_xyz.reshape(-1,256,int(self.crop_point_num/256),3)  #第二层的个数改成
        pc3_xyz = pc2_xyz_expand+pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3) 
        
        return pc1_xyz,pc2_xyz,pc3_xyz #center1 ,center2 ,fine

class _netlocalD(nn.Module):
    def __init__(self,crop_point_num):
        super(_netlocalD,self).__init__()
        self.crop_point_num = crop_point_num
        self.k = 20  # Set the number of neighbors; adjust based on requirements
        self.conv1 = torch.nn.Conv2d(2*3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64*2, 64, 1)
        self.conv3 = torch.nn.Conv2d(64*2, 128, 1)
        self.conv4 = torch.nn.Conv2d(128*2, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = torch.squeeze(x,1)
        x = x.transpose(1,2)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = F.relu(self.bn1(self.conv1(x)))     # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = F.relu(self.bn2(self.conv2(x)))     # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x_64 = torch.unsqueeze(x2,3)           # (batch_size, 128, num_points) -> (batch_size, 128, num_points, 1)
        x_64 = torch.squeeze(self.maxpool(x_64))      # (batch_size, 128, num_points) -> (batch_size, 128, 1)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = F.relu(self.bn3(self.conv3(x)))     # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x_128 = torch.unsqueeze(x3,3)           # (batch_size, 128, num_points) -> (batch_size, 128, num_points, 1)
        x_128 = torch.squeeze(self.maxpool(x_128))      # (batch_size, 128, num_points) -> (batch_size, 128, 1)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = F.relu(self.bn4(self.conv4(x)))     # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x_256 = torch.unsqueeze(x4,3)           # (batch_size, 256, num_points) -> (batch_size, 256, num_points, 1)
        x_256 = torch.squeeze(self.maxpool(x_256))      # (batch_size, 256, num_points) -> (batch_size, 256, 1)

        Layers = [x_256,x_128,x_64]
        x = torch.cat(Layers,1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x

# class _netlocalD(nn.Module):
#     def __init__(self,crop_point_num):
#         super(_netlocalD,self).__init__()
#         self.crop_point_num = crop_point_num
#         self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
#         self.conv2 = torch.nn.Conv2d(64, 64, 1)
#         self.conv3 = torch.nn.Conv2d(64, 128, 1)
#         self.conv4 = torch.nn.Conv2d(128, 256, 1)
#         self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(448,256)
#         self.fc2 = nn.Linear(256,128)
#         self.fc3 = nn.Linear(128,16)
#         self.fc4 = nn.Linear(16,1)
#         self.bn_1 = nn.BatchNorm1d(256)
#         self.bn_2 = nn.BatchNorm1d(128)
#         self.bn_3 = nn.BatchNorm1d(16)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x_64 = F.relu(self.bn2(self.conv2(x)))
#         x_128 = F.relu(self.bn3(self.conv3(x_64)))
#         x_256 = F.relu(self.bn4(self.conv4(x_128)))
#         x_64 = torch.squeeze(self.maxpool(x_64))
#         x_128 = torch.squeeze(self.maxpool(x_128))
#         x_256 = torch.squeeze(self.maxpool(x_256))
#         Layers = [x_256,x_128,x_64]
#         x = torch.cat(Layers,1)
#         x = F.relu(self.bn_1(self.fc1(x)))
#         x = F.relu(self.bn_2(self.fc2(x)))
#         x = F.relu(self.bn_3(self.fc3(x)))
#         x = self.fc4(x)
#         return x

if __name__=='__main__':
    input1 = torch.randn(8,2048,3)
    input2 = torch.randn(8,512,3)
    input3 = torch.randn(8,256,3)
    input_ = [input1,input2,input3]
    netG = _netG(3,1,[2048,512,256],1024)
    netD = _netlocalD(1024)
    output1, output2, output3 = netG(input_)
    real_center = torch.unsqueeze(output3,1)   
    D_out = netD(real_center)
    print(output3)
    print(D_out)
