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

class CityscapesDataSet(data.Dataset):
    def __init__(self, root= '/home/wawa/yang_net/datasets/cityscapes', list_path = '/home/wawa/yang_net/datasets/cityscapes/cityscapes_train_list.txt', max_iters=None, 
                 crop_size=(512, 1024),scale=True, ignore_label=255 ):
        self.transform = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        )
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
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

        print("length of dataset: ",len(self.files)) # 统计图片数量

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        # if self.scale:   # 改变水平和垂直比例因子
        #     f_scale = 0.5 + random.randint(0, 15) / 10.0  #random resize between 0.5 and 2  比例因子
        #     image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        #     label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        
        image = image[:, :, ::-1]  # 普通的图片是RGB，通过image = image[:, :, ::-1]转换成BGR
        # image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:               # 如果图像小了，则在图像加边框
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,    
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)   # 对图像进行裁剪或者填充
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        
        
        image = image.transpose((2, 0, 1)) # HWC -> CHW
        
        # if self.is_mirror:
        #     flip = np.random.choice(2) * 2 - 1
        #     image = image[:, :, ::flip]
        #     label = label[:, ::flip]


        return image.copy(), label.copy(), np.array(size), name


if __name__== "__main__":
    a = CityscapesDataSet()