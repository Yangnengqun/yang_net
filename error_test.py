# import sys
# sys.path.append
# import torch
# import cv2
# import numpy as np
# from torch.utils import data
# import torchvision.transforms as transforms
# path = '/home/wawa/yang_net/datasets/cityscapes/leftImg8bit/test/bielefeld/bielefeld_000000_000321_leftImg8bit.png'
# image = cv2.imread(path,cv2.IMREAD_COLOR)
# image = np.asarray(image, np.float32)
# print(image.shape)
# # image = image.transpose(2,0,1)
# image = image[:,:,::-1]
# image = transforms.ToTensor(image)
# print(image)

import torch
a = torch.randn((3,3))
mask =  a>0

print(a)
print(a[mask])