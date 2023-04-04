import torch
import torch.nn as nn

class SE_Block(nn.Module):                    
    def __init__(self, in_planes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = x
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        out = x*out1
        return out
if __name__ == "__main__":
    a = torch.randn(2,96,16,16)
    _,in_plane,_,_ = a.size()
    model = SE_Block(in_plane)
    out = model(a)
    print(out.shape)