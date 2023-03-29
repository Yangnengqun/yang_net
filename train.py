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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# def tarin_model(args):
#     return 0
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="train yang net")
#     parser.add_argument("--datasets_dir",default="/home/wawa/yang_net/datasets",type=str)
#     parser.add_argument("--batch_size",default=64,type=int)
    
#     args = parser.parse_args()
#     train_model(args)
    
    
    
    
    