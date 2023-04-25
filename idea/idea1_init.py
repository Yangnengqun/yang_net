
import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__ (self,in_channel = 3,out_channel = 13):
        super().__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = nn.Conv2d(in_channel, 
                                out_channel,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1)

        self.prelu = nn.PReLU(16)

        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.con1_1 = nn.Conv2d(in_channels=out_channel+3,out_channels=32,kernel_size=1)
  
    def forward(self, x):
        
        main = self.conv(x)
        main = self.batchnorm(main)
        
        side = self.maxpool(x)
        
        x = torch.cat([main, side], dim=1)
        x = self.prelu(x)
        
        return x


if __name__ == "__main__":
    a = torch.randn(2,3,128,128)
    model = InitialBlock()
    out = model(a)
    print(out.shape)