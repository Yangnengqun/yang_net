import torch
import torch.nn as nn

class Depthwise_Separable_conv(nn.Module):
    def __init__(self,in_channel,out_channel,s= 1):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride =s, groups=in_channel)
        self.wise_conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1)
        
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = self.depth_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return out
        