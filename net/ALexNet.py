import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

# 构建神经网络
class first_yang(nn.Module):
    def __init__(self):
        super(first_yang,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2,),
            nn.MaxPool2d(kernel_size=2 ,ceil_mode=False),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)        
        )
    
    def forward(self,x):
        x = self.model(x)
        return x


if __name__ =='__main__':
    yang = first_yang()
    input = torch.ones((64,3,32,32))
    output = yang(input)
    print(output.shape)