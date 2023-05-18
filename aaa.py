# import os

# path = "/home/wawa/yang_net/datasets/cityscapes/gtFine/train"


# # for root, dirs, files in os.walk(file):
# #     for file in files:
# #         path = os.path.join(root, file)
# #         print(path)

# for path,dirs,files in os.walk(path):
#     print(path)
#     print(dirs)
#     print("\n")
import math
import torch
from torch import nn
import torch.nn.functional as F

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
                 activation=None):
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


if __name__=="__main__":
    input = torch.randn(1,128,32,64)
    activate = nn.ReLU
    model = Sea_Attention(128,16,4,2,activate)
    output = model(input)
    print(output.shape)