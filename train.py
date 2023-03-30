import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from time import time
import os
# from skimage.io import imread
import copy
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import vgg19
# from torchsummary import summary
from cityscape import CityscapesDataSet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_transform= transforms.Compose([
        transforms.ToTensor()])
trainLoader = data.DataLoader( CityscapesDataSet(),batch_size = 16, shuffle = True, num_workers =1, drop_last = True)






# def tarin_model(args):
#     return 0
if __name__ == '__main__':
        
#     parser = argparse.ArgumentParser(description="train yang net")
#     parser.add_argument("--datasets_dir",default="/home/wawa/yang_net/datasets",type=str)
#     parser.add_argument("--batch_size",default=64,type=int)
    
#     args = parser.parse_args()
#     train_model(args)
        #----------------------------------------------#
        # 是否使用cuda
        Cuda = True 
        #----------------------------------------------#
        # 分类数  cityscapes为19
        num_classes = 19
        #----------------------------------------------#
        # 是否使用主干网络的与训练权重
        pretained = False
        #----------------------------------------------#
        # 模型权值文件保存地址
        model_path = "model_data/yang_net.pth"
        #----------------------------------------------#
        # 输入图片大小
        input_shape = [512,512]
        #----------------------------------------------#
        # 学习率
        lr = 1e-4
        #----------------------------------------------#
        # 数据集路径
        cityscapes_dir = "/home/wawa/yang_net/datasets/cityscapes"
        #----------------------------------------------#
        # 设置多线程读取数据,开启后会加快读取速度，但会占用更多内存  内存较小的电脑设置为2或者0
        num_workers = 1
        
        
        
        
        # model = yang_net()
        
    
    
    
    
    