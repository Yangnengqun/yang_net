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

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

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

    # def forward_split_cat(self, x):
    #     # for training/inference
    #     x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
    #     x1 = self.partial_conv3(x1)
    #     x = torch.cat((x1, x2), 1)

    #     return x

class SS_nbt_module(nn.Module):
    def __init__(self, chann, dilated=1):        
        super().__init__()

        oup_inc = chann//2
        self.conv3x1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        self.conv1x3_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
        self.conv3x1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
        

        self.bn_r= nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        self.gostconv = GhostModule(inp = oup_inc*2, oup = oup_inc*2, kernel_size=1,ratio=2,dw_size=3,stride=1,dialations=dilated,relu=True)	
        
        self.relu = nn.ReLU(inplace=True)
    def _concat(self,x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_l(output1)
        output1 = self.bn_l(output1)

        output2 = self.conv3x1_r(x2)
        output2 = self.relu(output2)
        print(output2.shape)
        output2 = self.conv1x3_r(output2)
        output2 = self.bn_r(output2)

        out = self._concat(output1,output2)
        out = F.relu(out, inplace=True)
        out = self.gostconv(out)
        out = F.relu(residual + out, inplace=True)
        out = channel_shuffle(out,2)
        return out

class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernelsize=1, stride=1, padding=0, dilation=1,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernelsize, stride, kernelsize//2, bias=False,groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        return x
        
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6   
        
class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2,
                 activation=nn.ReLU):
        # dim 输入维度
        # key_dim token的长度
        # num_head 多头
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        # Cqk = 0.5Cv
        self.to_q = Conv2d_BN(dim, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1)
        
        # detail enhance
        self.dwconv = Conv2d_BN(2*self.dh, 2*self.dh, kernelsize=3, stride=1, padding=1, dilation=1,
                 groups=2*self.dh)
        self.act = activation()
        self.pwconv = Conv2d_BN(2*self.dh, dim, kernelsize=1)
        self.sigmoid = h_sigmoid()
        

        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_row = torch.nn.Sequential(activation(), 
                                Conv2d_BN(self.dh, self.dh))
        
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(),
                                Conv2d_BN(self.dh, self.dh))
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, ))
        

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.to_q(x)  # Q为（B, key_dim * num_heads, H, W）
        k = self.to_k(x)  # K为（B, key_dim * num_heads, H, W）
        v = self.to_v(x)  # V为（B, int(attn_ratio * key_dim) * num_heads, H, W）
        
        # detail enhance
        qkv = torch.cat([q,k,v],dim=1)
        qkv = self.act(self.dwconv(qkv))  # 深度可分离卷积   计算复杂度9HW(2C_qk+C_v)
        qkv = self.pwconv(qkv)    # 点卷积   计算复杂度HWC(2C_qk+Cv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)  # 得到有position的（B, num_heads,H, key_dim）（B, num_heads, key_dim, H）
        # q.mean(-1)输出为（B, key_dim * num_heads, H）
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)  #得到有position的（B, num_heads, key_dim, H）
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)   # （B, num_heads,H, attn_ratio*key_dim）
        
        attn_row = torch.matmul(qrow, krow) * self.scale   #复杂度H*H*C_qv  （B, num_heads,H, key_dim）
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C （B, num_heads,H, attn_ratio*key_dim）
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        
        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)   # ？？？
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        return xx

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        dw_out_feature = hidden_features-in_features
        
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_features, dw_out_feature, kernel_size = 3 , stride=1, padding=1, dilation=1, groups=in_features, bias=False),
            nn.BatchNorm2d(dw_out_feature),
            nn.ReLU(inplace=True)
        )
        self.fc2 = Conv2d_BN(hidden_features,out_features,kernelsize=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        dw_x = self.dwconv(x)
        x = torch.cat([dw_x,x],dim = 1)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class transformer_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.1,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Sea_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                       activation=act_layer)
   
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        x1 = F.relu(x1)
        return x1

class Injecttion(nn.Module):
    def __init__(self,semantics_channel = 128,local_channal= 64,):
        super().__init__()
        self.mul_semanyics = nn.Sequential(
            nn.Conv2d(in_channels=semantics_channel,out_channels=local_channal,kernel_size=1),
            nn.BatchNorm2d(local_channal)
        )
        self.add_semanyics = nn.Sequential(
            nn.Conv2d(in_channels=semantics_channel,out_channels=local_channal,kernel_size=1),
            nn.BatchNorm2d(local_channal)
        )
    def forward(self,semantics_input,local_input):
        mul_input = self.mul_semanyics(semantics_input)
        mul_input = F.sigmoid(mul_input)
        mul_input = F.interpolate(mul_input,size=(64,128),mode="bilinear")
        add_input = self.add_semanyics(semantics_input)
        add_input = F.interpolate(add_input,size=(64,128),mode="bilinear")
        output = local_input*mul_input
        output = add_input+output
        
        return F.relu6(output)
        
        
        
        
        
class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(16,1))
        

        self.layers.append(DownsamplerBlock(16,24))
        
        

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(24,1))
  
        self.layers.append(DownsamplerBlock(24,32))

        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(32,1))
            self.layers.append(SS_nbt_module(32,2))
            self.layers.append(SS_nbt_module(32,5))
            self.layers.append(SS_nbt_module(32,9))
        
        self.layers_16 = nn.Sequential(
            DownsamplerBlock(32,64),
            SS_nbt_module(64,1)
            
        )
        self.layers_32 = nn.Sequential(
            DownsamplerBlock(64,128),
            SS_nbt_module(128,1)
        )
        
        self.transformer_1 = transformer_Block(dim = 128, key_dim=16, num_heads=4, mlp_ratio=4, attn_ratio=2, drop_path=0.1)
        self.transformer_2 = transformer_Block(dim = 128, key_dim=16, num_heads=4, mlp_ratio=4, attn_ratio=2, drop_path=0.1)
        
        
        
        self.fusion = Conv2d_BN(in_channel=96,out_channel=64,kernelsize=3,stride=1,padding=1)
        
        self.injection = Injecttion(semantics_channel=128,local_channal=64)
        
        
    

                    

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        B,C,H,W = input.size()
        
        output = self.initial_block(input)

        for layer in self.layers:
            output_8= layer(output)    # (batch_size,32,H/8,W/8)
            
        output_16 = self.layers_16(output_8)  # (batch_size,64,H/16,W/16)
        
        output_32 = self.layers_32(output_16)
        # print("------")
        # print(output_8.shape)
        output_32 = self.transformer_1(output_32)
        output_32 = self.transformer_2(output_32)   # (batch_size,128,H/16,W/16)
        
        print("------")
        print(H)
        out16 = F.interpolate(output_16,size=(H//8,W//8),mode="bilinear")
        print("------")
        print(output_8.shape)
        print("------")
        print(out16.shape)
        output_8 = torch.cat([output_8,out16],dim=1)
        output_8 = self.fusion(output_8)
        output = self.injection(output_32,output_16)
        
        
        
        
        
        
        
        

        if predict:
            output = self.output_conv(output)

        return output


          



class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = nn.Conv2d(in_channels= 64,out_channels= num_classes,kernel_size=3,stride=1,padding=1)

    def forward(self, input):
        
        output = self.apn(input)
        out = interpolate(output, size=(512, 1024), mode="bilinear", align_corners=True)

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