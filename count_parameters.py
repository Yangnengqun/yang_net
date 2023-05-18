import torch
import thop
# import torch.nn as nn
from net import Fasternet,lednet, ghostnet,ghostv2
from idea import yang_5_4,yang_5_16
from aaa import Sea_Attention

# model = lednet.Net(20)
# model = Fasternet.Net(20)
model = yang_5_16.Net()
# model = ghostv2.ghostnetv2()
# model = Sea_Attention(128,16,4,activation=torch.nn.ReLU)
# input = torch.randn(1,128,32,64)


input = torch.randn(1,3,512,1024)
flops, params = thop.profile(model, inputs=(input,))  #input 输入的样本

print('GFLOPs:',flops/(10**9))

print("Net has {}M parameters".format(params/1024/1024))
