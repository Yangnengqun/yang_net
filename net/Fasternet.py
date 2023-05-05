import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate

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
    
# Pconv 
class Partial_conv3(nn.Module):

    def __init__(self,
                 dim, 
                 n_div=4, 
                 forward="split_cat"):
        # dim 输入维度
        # n_div通常为1/4
        # forward pconv的形式

        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

# MLP表示为一个shortcut+（PCONV+PWCONV+[laynorm+GELU]+CONV)
class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer=nn.GELU(),
                 pconv_fw_type = 'split_cat'
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)


    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x


# 一个basicstage 里面有depth个MLP
class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x

# 输入层
class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        #添加一个maxpoll层
        # self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        # x2 = self.maxpool(x)
        # x = self.norm(torch.cat((x1,x2),dim=1))
        return x

# 2×2下采样
class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x




class Encoder(nn.Module):

    def __init__(self,
                 in_chans=3,  # 输入通道数
                 num_classes=20,  # 类别数
                 embed_dim=40,    # embedding的输出维度
                 depths=(1, 2, 8),  # 每个stage的MLP数量
                 mlp_ratio=2.,         # MLP中PwCONV（1×1）通道的扩大倍数
                 n_div=4,    # PCONV取通道数的1/ndiv
                 patch_size=2, # 输入层的卷积核
                 patch_stride=2, # 输入层的步长
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,  # 下采样层
                 patch_norm=True,
                 drop_path_rate=0.1,
                 norm_layer='BN',
                 act_layer='RELU',
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.stage0= BasicStage(dim=int(embed_dim * 2 ** 0),
                               n_div=n_div,
                               depth=depths[0],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
        self.merge0 = PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** 0),
                                 norm_layer=norm_layer)
        
        self.stage1 = BasicStage(dim=int(embed_dim * 2 ** 1),
                               n_div=n_div,
                               depth=depths[1],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
        self.merge1 = PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** 1),
                                 norm_layer=norm_layer)
        
        self.stage2 = BasicStage(dim=int(embed_dim * 2 ** 2),
                               n_div=n_div,
                               depth=depths[2],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
        # self.merge2 = PatchMerging(patch_size2=patch_size2,
        #                          patch_stride2=patch_stride2,
        #                          dim=int(embed_dim * 2 ** 2),
        #                          norm_layer=norm_layer)
        # self.stage3 = BasicStage(dim=int(embed_dim * 2 ** 3),
        #                        n_div=n_div,
        #                        depth=depths[3],
        #                        mlp_ratio=self.mlp_ratio,
        #                        drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
        #                        norm_layer=norm_layer,
        #                        act_layer=act_layer,
        #                        pconv_fw_type=pconv_fw_type
        #                        )
        self.output_conv = nn.Conv2d(160, num_classes, 1, stride=1, padding=0, bias=True)
        

        
        
    def forward(self, x, predict=False):
    # output only the features of last layer for image classification
        _,_,h,w = x.size()
        x = self.patch_embed(x)
        x0 = self.stage0(x)
        x1 = self.merge0(x0)
        x1 = self.stage1(x1)
        x2 = self.merge1(x1)
        output = self.stage2(x2)
        # out = self.merge2(x1)
        # out = self.stage3(out)
        if predict:
            output = self.output_conv(output)


        
        
        return output

class Interpolate(nn.Module):
    def __init__(self,size,mode):
        super(Interpolate,self).__init__()
        
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
    def forward(self,x):
        x = self.interp(x,size=self.size,mode=self.mode,align_corners=True)
        return x
        

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

        self.apn = APN_Module(in_ch=160,out_ch=20)
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
            self.encoder = Encoder()
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    
            return self.decoder.forward(output)


if __name__ == "__main__":
    input = torch.randn(4,3,512,1024)
    model = Net(20)
    out = model(input)
    print(out.shape)
