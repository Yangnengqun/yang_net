import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from time import time
import os
from skimage.io import imread
import copy
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import vgg19
from torchsummary import summary


classes = ['background','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
colormap = [[0, 0, 0], # 0 = background
            [128, 0, 0], # 1 = aeroplane
            [0, 128, 0], # 2 = bicycle
            [128, 128, 0], # 3 = bird
            [0, 0, 128], # 4 = boat
            [128, 0, 128], # 5 = bottle
            [0, 128, 128], # 6 = bus
            [128, 128, 128], # 7 = car
            [64, 0, 0], # 8 = cat
            [192, 0, 0], # 9 = chair
            [64, 128, 0], # 10 = cow
            [192, 128, 0], # 11 = dining table
            [64, 0, 128], # 12 = dog
            [192, 0, 128], # 13 = horse
            [64, 128, 128], # 14 = motorbike
            [192, 128, 128], # 15 = person
            [0, 64, 0], # 16 = potted plant
            [128, 64, 0], # 17 = sheep
            [0, 192, 0], # 18 = sofa
            [128, 192, 0], # 19 = train
            [0, 64, 128]] # 20 = tv/monitor

## 将一个标记好的图像转化为类别标签图像
def image2label(image, colormap):
    # 将标签转化为每个像素值为一类数据
    cm2lbl = np.zeros(256**3)
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1]*256+cm[2])] = i
    # 对一张图像转换
    image = np.array(image, dtype="int64")
    ix = (image[:,:,0]*256+image[:,:,1]*256+image[:,:,2])
    image2 = cm2lbl[ix]
    return image2

# 随机裁剪图像
def rand_crop(data,label,high,width):
    im_width,im_high = data.size
    # 生成图像随机点的位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_high - high)
    right = left + width
    bottom = top + high
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    return data,label

# 单组图像的转换操作
def img_transforms(data, label, high, width, colormap):
# 数据的随机裁剪、将图像数据进行标准化、将标记图像数据进行二维标签化的操作，输出原始图像和类别标签的张量数据
    data, label = rand_crop(data, label, high, width)
    data_tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    data = data_tfs(data)
    label = torch.from_numpy(image2label(label, colormap))
    return data, label

# 定义需要读取的数据路径的函数
def read_image_path(root=r"D:\毕业设计\VOC2012\ImageSets\Segmentation\train.txt"):
# 原始图像路径输出为data，标签图像路径输出为label
    image = np.loadtxt(root, dtype=str)
    n =len(image)
    data, label = [None]*n, [None]*n
    for i,fname in enumerate(image):
        data[i] = r"D:\毕业设计\VOC2012\JPEGImages\%s.jpg" % (fname)
        label[i] = r"D:\毕业设计\VOC2012\SegmentationClass\%s.png" % (fname)
    return data, label
