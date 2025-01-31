#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils

# from model_PFNet import _netlocalD,_netG

from Model_RP_PCN import _netlocalD,_netG

import RP_PCN_loader

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=2048,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='../Trained_Model/point_netG80.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,512,256],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)

# test_dset = mydataset_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Airplane', npoints=opt.pnum, split='test')
# test_dset = mydataset_loader.PartDataset( root='D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/dataset/mydata',classification=True, class_choice=None, npoints=opt.pnum, split='test')
test_dset = RP_PCN_loader.PartDataset( root='D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/dataset/mydata',classification=True, class_choice=None, npoints=opt.pnum, split='test', crop_point_num=opt.crop_point_num)
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=False,num_workers = int(opt.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num) 
point_netG = torch.nn.DataParallel(point_netG)
point_netG.to(device)
point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
point_netG.eval()


criterion_PointLoss = PointLoss().to(device)
errG_min = 100
n = 0
for i, data in enumerate(test_dataloader, 0):
        
    real_point, fake_point, target = data
    
    input_cropped1 = real_point
    real_center = fake_point
         
    
    real_center.to(device) 
    input_cropped1 = input_cropped1.to(device)
    # real_center = torch.squeeze(real_center,1)
    
    # input_cropped1 = torch.squeeze(input_cropped1,1)
    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
    input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
    input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
    input_cropped2 = input_cropped2.to(device)
    input_cropped3 = input_cropped3.to(device)      
    input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
    

    fake_center1,fake_center2,fake=point_netG(input_cropped)
    fake = fake.cuda()
    real_center = real_center.cuda()
    real_center =real_center.cuda()
    errG = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))#+0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
    errG = errG.cpu()
    a = random.randint(4,8)
    b = 6
    if errG.detach().numpy()>errG_min:
#    if a!=b:
        pass
    
    else:
        # errG_min = errG.detach().numpy()
        # print(errG_min)
        fake =fake.cpu()
        np_fake = fake[0].detach().numpy()  #256
        real_center = real_center.cpu()
        np_real = real_center.data[0].detach().numpy() #256
        input_cropped1 = input_cropped1.cpu()
        np_inco = input_cropped1[0].detach().numpy() #1024
        np_crop = np_inco
        n=n+1
        # k = 0
        # for m in range(opt.pnum):
        #     if distance_squre1(np_inco[m],p_origin)==0.00000 and k<opt.crop_point_num:
        #         k += 1
        #         pass
        #     else:
        #         np_crop.append(np_inco[m])
        # np.savetxt('D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/test_example/crop'+str(n)+'.csv', np_crop, fmt = "%f,%f,%f")
        # np.savetxt('D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/test_example/fake'+str(n)+'.csv', np_fake, fmt = "%f,%f,%f")
        # np.savetxt('D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/test_example/real'+str(n)+'.csv', np_real, fmt = "%f,%f,%f")
        np.savetxt('D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/test_example/crop_txt'+str(n)+'_G80.txt', np_crop, fmt = "%f,%f,%f")
        np.savetxt('D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/test_example/fake_txt'+str(n)+'_G80.txt', np_fake, fmt = "%f,%f,%f")
        np.savetxt('D:/PhD_Study/Code/PF-Net-Point-Fractal-Network/test_example/real_txt'+str(n)+'_G80.txt', np_real, fmt = "%f,%f,%f")    
    
    
