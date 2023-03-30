import torch
import torch.tensor as tensor
import numpy as np
a = torch.randn((1,2,4,5))
a = a.cpu().data[0]
b = torch.argmax(a,dim=0)
print(a.shape)
print(a)
print(b)
print(b.size())
# a=a.transpose(2,0)
# b=b.transpose(2,1,0)
# print(a.shape)
# print(a)
# print(b)
# print(torch.__version__)