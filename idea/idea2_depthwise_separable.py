import torch
import torch.nn as nn

class Depthwise_Separable_conv(nn.Module):
    def __init__(self,in_channel,out_channel,s= 1):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride =s, groups=in_channel)
        self.bn_1 = nn.BatchNorm2d(in_channel)
        self.relu6_1 = nn.ReLU6()
        self.wise_conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1)
        
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.relu6_2 = nn.ReLU6()
        
    def forward(self,x):
        out = self.depth_conv(x)
        out = self.bn_1(out)
        out = self.relu6_1(out)
        out = self.wise_conv(out)
        out = self.bn_2(out)
        out = self.relu6_2(out)
        
        return out


if __name__ == "__main__":
    a = torch.randn(2,16,128,128)
    model = Depthwise_Separable_conv(in_channel=16,out_channel=32)
    out = model(a)
    print(out.shape)
        