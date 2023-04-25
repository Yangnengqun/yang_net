import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from time import time
import os
# from skimage.io import imread
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
# from torchsummary import summary
from cityscape import CityscapesDataSet
from utils import color,loss,miou
from net import FCN

from torch.utils import tensorboard

from torch.utils.tensorboard import SummaryWriter
import torchvision





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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
        epoch = 300
        total_train_step  = 0
        #----------------------------------------------#
        # 数据集路径
        cityscapes_dir = "/home/wawa/yang_net/datasets/cityscapes"
        #----------------------------------------------#
        # 设置多线程读取数据,开启后会加快读取速度，但会占用更多内存  内存较小的电脑设置为2或者0
        num_workers = 1
        
        
        trainLoader = data.DataLoader( CityscapesDataSet(),batch_size = 16, shuffle = True, num_workers =1, drop_last = True)
        model = torchvision.models.MobileNetV2(num_classes=19)
        print(model)
        model = model.to(device)
        loss_fn = loss.CrossEntropyLoss2d()
        loss_fn = loss_fn.to(device)
        critizer = torch.optim.SGD(model.parameters(),lr = lr)
        
        writer = SummaryWriter("./logs_train")
        start_time = time.time()
        model.train()
        for i in range(epoch):
                print("-----第{}轮训练开始-----".format(i+1))
                for data in trainLoader:
                        image,label = data
                        image = image.to(device)
                        label = label.to(device)
                        
                        output = model(image)
                        loss_now = loss_fn(output,label)
                        critizer.zero_grad()
                        loss_now.backward()
                        critizer.step()
                        
                        total_train_step +=1
                        if total_train_step %100 ==0:
                                end_time = time.time()
                                print(end_time-start_time)
                                print("训练次数:{},loss:{}".format(total_train_step,loss))
                                writer.add_scalar("train_loss", loss.item(), total_train_step)
                # 用于断点训练
                checkpoint = {
                        "net": model.state_dict(),
                        'optimizer':critizer.state_dict(),
                        "epoch": i
                }
                if not os.path.isdir("/home/wawa/yang_net/model_data"):
                        os.mkdir("/home/wawa/yang_net/model_data")
                torch.save(checkpoint, '/home/wawa/yang_net/model_data/ckpt_best_%s.pth' %(str(i)))  
        writer.close()  
#tensorboard --logdir="logs_train" --port=6007         
                        
                        
                        
        


        
        
        
        
        # model = yang_net()
        
    
    
    
    
    