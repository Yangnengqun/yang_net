import torch
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torchvision
import cv2
from torch.utils import data
import pickle
from PIL import Image
from torchvision import transforms
# 当前地址
# now_path = osp.dirname(__file__)
# print(now_path)
# cityscapes_path = osp.join(now_path,'datasets/cityscapes')
# list_realpath = osp.join(now_path,'datasets/cityscapes/cityscapes_train_list.txt')
# print(cityscapes_path,list_realpath)

from transform import *
#
class CityscapesDataSet(data.Dataset):
    def __init__(self, root= '/home/wawa/yang_net/datasets/cityscapes', list_path = '/home/wawa/yang_net/datasets/cityscapes/cityscapes_train_list.txt',mode='train', max_iters=None, 
                 cropsize=(640, 480),randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)):
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]    # i_id.strip()删除字符串字符串头尾的指定的字符（默认）空格
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))  # max_iters最大训练次数 np.ceil(ndarray) 计算大于等于该值的最小整数
        self.files = []


        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])  # split()[0]表示输出空格前面的 osp.join()将字符串拼接，常用于地址拼接  img_file为单个图片地址
            #print(img_file)
            label_file = osp.join(self.root, name.split()[1])  # label_file为单个标签地址
            #print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })   # 将图片地址和全名导入files
            
            
            
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])
        

        print("length of dataset: ",len(self.files)) # 统计图片数量

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        img = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        # label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = np.array(label).astype(np.int64)
        return img, label


if __name__== "__main__":
    ds = CityscapesDataSet()
    dat = data.DataLoader(ds,batch_size=4,num_workers=0)
    for file1 in dat:
        img,lable = file1
        print(img.shape,lable.shape)
        