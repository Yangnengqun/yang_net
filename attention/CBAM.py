import torch
import torch.nn as nn


class CAM(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        
        self.con1_1 = nn.Conv2d(in_channels=in_channel,out_channels= in_channel//16,kernel_size=1)
        self.relu = nn.ReLU()
        self.con1_2 = nn.Conv2d(in_channels=in_channel//16,out_channels= in_channel,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
 
 
    def forward(self,x):
        
        avg_out = self.avg(x)
        avg_out = self.con1_1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.con1_2(avg_out)
        
        
        max_out = self.max(x)
        max_out = self.con1_1(max_out)
        max_out = self.relu(max_out)
        max_out = self.con1_2(max_out)
        
        out = avg_out+max_out
        out = self.sigmoid(out)
        out = out*x
        return out
         
class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        avg_out = torch.mean(x,dim = 1,keepdim=True)
        max_out,_ = torch.max(x,dim= 1,keepdim=True)
        out = torch.cat([avg_out,max_out],dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = out*x
        
        return out
    
class BasicBlock(nn.Module):
    def __init__(self,in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.CAM = CAM(planes)
        self.SAM = SAM()
        
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.CAM(out)
        out = self.SAM(out)
        out += x
        out = self.relu(out)
        return out

if __name__ == "__main__":
    model = BasicBlock(3,16)
    print(model)
    print(model.SAM)
    