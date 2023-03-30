import torch
import torch.nn as nn

class Resdual_18(nn.Module):
    def __init__(self,in_channel, out_channel, use_conv1_1 = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Sequential( nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1,stride=strides),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU()
        )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1,stride=strides),
                                   nn.BatchNorm2d(out_channel))
        if use_conv1_1:
            self.con1_1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=strides)
        else:
            self.con1_1 = None
        self.relu = nn.ReLU()
        
    def forward(self,input):
        y = self.conv1(input)
        y = self.conv2(y)
        if self.con1_1:
            input = self.con1_1(input)
        out = y+input
        out = self.relu(out)
        return out

if __name__ == "__main__":
    x = torch.randn(2,3,16,16)
    print(x.shape)
    model = Resdual_18(3,6,use_conv1_1 = True)
    out = model(x)
    print(out.shape)