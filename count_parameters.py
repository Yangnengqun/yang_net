import torch
import thop
# import torch.nn as nn
from net import Fasternet,lednet
from idea import yang_5_4


# model = lednet.Net(20)
model = Fasternet.Net(20)
# model = yang_5_4.Net()




input = torch.randn(1,3,512,1024)
flops, params = thop.profile(model, inputs=(input,))  #input 输入的样本

print('GFLOPs:',flops/(10**9))

print("Net has {}M parameters".format(params/1024/1024))
