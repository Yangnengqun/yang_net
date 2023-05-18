import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
from typing import List
import math


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    # flatten
    x = x.view(batchsize,-1,height,width)
    
    return x

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()
		
        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		nn.BatchNorm2d(out_ch, eps=1e-3),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)

class DownsamplerBlock (nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock,self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output
    

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, dialations=1,relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1, padding=dialations,dilation=dialations, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class SS_nbt_module(nn.Module):
    def __init__(self, chann, dilated=1):        
        super().__init__()

        oup_inc = chann//2
        self.conv3x1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        self.conv1x3_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
        self.conv3x1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
        

        # self.conv3x1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        # self.conv1x3_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
        # self.bn_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        # self.conv1x3_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
        # self.conv3x1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        self.bn_r= nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        self.gostconv = GhostModule(inp = oup_inc*2, oup = oup_inc*2, kernel_size=1,ratio=2,dw_size=3,stride=1,dialations=dilated,relu=True)	
        
        self.relu = nn.ReLU(inplace=True)
    def _concat(self,x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_l(output1)
        output1 = self.bn_l(output1)

        output2 = self.conv3x1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv1x3_r(output2)
        output2 = self.bn_r(output2)

        out = self._concat(output1,output2)
        out = F.relu(out, inplace=True)
        out = self.gostconv(out)
        out = F.relu(residual + out, inplace=True)
        out = channel_shuffle(out,2)
        return out

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,32)

        self.layers = nn.ModuleList()

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(32,1))
        

        self.layers.append(DownsamplerBlock(32,64))
        

        for x in range(0, 4):
            self.layers.append(SS_nbt_module(64,1))
  
        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128,1))
            self.layers.append(SS_nbt_module(128,2))
            self.layers.append(SS_nbt_module(128,5))
            self.layers.append(SS_nbt_module(128,9))
            
        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128,2))
            self.layers.append(SS_nbt_module(128,5))
            self.layers.append(SS_nbt_module(128,9))
            self.layers.append(SS_nbt_module(128,17))
                    

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
        # midddle branch
        self.mid = nn.Sequential(
		Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)
		
        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)
		
        self.down3 = nn.Sequential(
		Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
		Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
	)
		
        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)
	
    def forward(self, x):
        
        h = x.size()[2]
        w = x.size()[3]
        
        b1 = self.branch1(x)
        # b1 = Interpolate(size=(h, w), mode="bilinear")(b1)
        b1= interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)
	
        mid = self.mid(x)
		
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x3 = Interpolate(size=(h // 4, w // 4), mode="bilinear")(x3)
        x3= interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)	
        x2 = self.conv2(x2)
        x = x2 + x3
        # x = Interpolate(size=(h // 2, w // 2), mode="bilinear")(x)
        x= interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
       		
        x1 = self.conv1(x1)
        x = x + x1
        # x = Interpolate(size=(h, w), mode="bilinear")(x)
        x= interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        		
        x = torch.mul(x, mid)

        x = x + b1
       
       
        return x
          



class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = APN_Module(in_ch=128,out_ch=20)
        # self.upsample = Interpolate(size=(512, 1024), mode="bilinear")
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
  
    def forward(self, input):
        
        output = self.apn(input)
        out = interpolate(output, size=(512, 1024), mode="bilinear", align_corners=True)
        # out = self.upsample(output)
        # print(out.shape)
        return out


class Net(nn.Module):
    def __init__(self, num_classes=20, encoder=None):  
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    
            return self.decoder.forward(output)


model = Net(20,None)
input = torch.randn(4,3,512,1024)
out = model(input)
print(out.shape)