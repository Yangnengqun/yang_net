import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
from typing import List
import math

# ------------------------------------------32 48 128 256  ---------------------------------
# GFLOPs: 11.54220032
# Net has 3.3623199462890625M parameters

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
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1,dilation=1,bias=True):
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
    def __init__(self, in_chann, out_chann, down = False, dilated_1=1, dilated_2 = 1):        
        super().__init__()      
        
        stride3_3 = 1
        if down:
            stride3_3 = 2
        self.down = down
        oup_inc = in_chann//2
        self.conv3x3_l = nn.Conv2d(oup_inc, oup_inc, kernel_size=3, stride=stride3_3, padding = dilated_1,dilation=dilated_1, bias=True)        
        self.conv3x3_r = nn.Conv2d(oup_inc, oup_inc, kernel_size=3, stride=stride3_3, padding = dilated_2,dilation=dilated_2, bias=True)
        self.bn= nn.BatchNorm2d(oup_inc*2, eps=1e-03)
        self.gostconv = GhostModule(inp = oup_inc*2, oup = out_chann, kernel_size=1,ratio=2,dw_size=3,stride=1,dialations=1,relu=True)
        if down:    
            self.down =nn.Sequential(
                nn.AvgPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=in_chann,out_channels=out_chann,kernel_size=1),
                nn.BatchNorm2d(out_chann)
                )

        
        self.relu = nn.ReLU(inplace=True)
    def _concat(self,x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        residual = input
        if self.down:
            residual = self.down(residual)
        x1, x2 = split(input)

        output1 = self.conv3x3_l(x1)
        output2 = self.conv3x3_r(x2)
        out = self._concat(output1,output2)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = self.gostconv(out)
        out = F.relu(residual + out, inplace=True)
        out = channel_shuffle(out,2)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,32)

        self.layers1_4_down = SS_nbt_module(in_chann=32,out_chann=48,down=True,dilated_1=1,dilated_2=1)
        
        self.layers1_8_down = SS_nbt_module(in_chann=48,out_chann=128,down=True,dilated_1=1,dilated_2=1)
        self.layers1_8 = nn.ModuleList()
        for x in range(0, 2):
            self.layers1_8.append(SS_nbt_module(in_chann=128,out_chann=128,down=False,dilated_1=1,dilated_2=1))
            
        
        self.layers1_16_down = SS_nbt_module(in_chann=128,out_chann=256,down=True,dilated_1=1,dilated_2=1)
        self.layers1_16 = nn.ModuleList()
        for x in range(0, 1):    
            self.layers1_16.append(SS_nbt_module(in_chann=256, out_chann=256, down=False, dilated_1=1, dilated_2 = 1))
        for x in range(0, 1):    
            self.layers1_16.append(SS_nbt_module(in_chann=256, out_chann=256, down=False, dilated_1=1, dilated_2 = 2))
        for x in range(0, 2):    
            self.layers1_16.append(SS_nbt_module(in_chann=256, out_chann=256, down=False, dilated_1=1, dilated_2 = 4))
        for x in range(0, 4):    
            self.layers1_16.append(SS_nbt_module(in_chann=256, out_chann=256, down=False, dilated_1=1, dilated_2 = 14))


    def forward(self, input):
        
        output_2 = self.initial_block(input)
        output_4 = self.layers1_4_down(output_2)
        output_8 = self.layers1_8_down(output_4)
        for layer in self.layers1_8:
            output_8 = layer(output_8)
        output_16 = self.layers1_16_down(output_8)
        for layer in self.layers1_16:
            output_16 = layer(output_16)


        return output_16, output_8, output_4

class Decoder(nn.Module):
    def __init__(self, in_ch_1_16 = 256, in_ch_1_8 = 128, in_ch_1_4 = 48, num_class=20):
        super().__init__() 
        
        self.branch_16 = nn.Sequential(
            nn.Conv2d(in_channels= in_ch_1_16, out_channels=128, kernel_size=1),
		    nn.BatchNorm2d(128, eps=1e-3),
        )
        
        self.branch_8_1 = nn.Sequential(
            nn.Conv2d(in_channels = in_ch_1_8, out_channels=128, kernel_size=1),
		    nn.BatchNorm2d(128, eps=1e-3),
        )
        self.relu_8 = nn.ReLU()
        self.branch_8_2 = Conv2dBnRelu(in_ch = in_ch_1_8,out_ch=64,kernel_size=3,stride=1,padding=1)
        
        self.branch_4_1 = Conv2dBnRelu(in_ch = in_ch_1_4,out_ch=8)
        self.branch_4_2 = Conv2dBnRelu(in_ch = 72, out_ch=64, kernel_size=3, stride=1, padding=1)
        self.branch_4_3 = nn.Conv2d(in_channels=64,out_channels=num_class,kernel_size=1)
        
	
    def forward(self, input_16, input_8, input_4):
        
        h = input_4.size()[2]
        w = input_4.size()[3]
        
        output_16 = self.branch_16(input_16)
        output_16 = interpolate(output_16, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        
        output_8 = self.branch_8_1(input_8)
        output_8 = self.relu_8(output_8 + output_16)
        output_8 = self.branch_8_2(output_8)
        output_8 = interpolate(output_8, size=(h, w), mode="bilinear", align_corners=True)
        
        output_4 = self.branch_4_1(input_4)
        output_4 = torch.cat([output_8,output_4], dim = 1)
        output_4 = self.branch_4_2(output_4)
        output_4 = self.branch_4_3(output_4)
        	
        
        return output_4
          





class Net(nn.Module):
    def __init__(self, num_classes=20):  
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder(in_ch_1_16 = 256, in_ch_1_8 = 128, in_ch_1_4 = 48, num_class=num_classes)

    def forward(self, input):
            output_16, output_8, output_4 = self.encoder(input)
            # print(output_16.shape, output_8.shape, output_4.shape, )
            output = self.decoder(output_16, output_8, output_4)
            output = interpolate(output, size=(512, 1024), mode="bilinear", align_corners=True)   
            return output


model = Net(20)
input = torch.randn(4,3,512,1024)
out = model(input)
print(out.shape)