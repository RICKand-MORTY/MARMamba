import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention
from timm.layers import DropPath, to_2tuple, trunc_normal_
import os
import torchvision.utils as tvu
from torchvision.ops import DeformConv2d
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm import Mamba

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)
    
class DWConv(nn.Module):
    def __init__(self, dim, stride=1, kernel=3, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, stride, padding, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type= 'WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DRSA(nn.Module):
    def __init__(self,dim,num_heads,bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Sequential(
            DWConv(dim,stride=2),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )
        self.k = nn.Sequential(
            DWConv(dim, stride=2),
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)
        )
        self.alpha = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias),
            DWConv(dim//2)
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        
    def forward(self,x):
        b,c,h,w = x.shape
        #print(f"x.shape={x.shape}")
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #print(f"q.shape={q.shape}, k.shape={k.shape}")
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.alpha
        attn = attn.softmax(dim=-1) @ v
        out = rearrange(attn, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        #print(f"out.shape={out.shape}")
        return out
        
class P2FFN(nn.Module):
    def __init__(self,dim,expansion=2,bias=True):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*expansion, kernel_size=1, bias=bias),
            nn.GELU(),
            DWConv(dim*expansion),
            nn.GELU(),
            nn.Conv2d(dim*expansion, dim, kernel_size=1, bias=bias)
        )
    
    def forward(self,x):
        return self.ffn(x)
    
class MARformerBlock(nn.Module):
    def __init__(self,dim,num_heads,bias=True):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = DRSA(dim,num_heads,bias=bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = P2FFN(dim,bias=bias)
    
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class MARformer(nn.Module):
    def __init__(self,dim,depth,num_heads,in_channels=1,bias=True):
        super().__init__()
        
        self.en1 = nn.Sequential(
            *[MARformerBlock(dim,num_heads[0],bias=bias) for i in range(depth[0])]
            
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2)
        )
        
        self.en2 = nn.Sequential(
            *[MARformerBlock(dim* 2 ** 1,num_heads[1],bias=bias) for i in range(depth[1])]
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 1 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2)
        )
        self.en3 = nn.Sequential(
            *[MARformerBlock(dim* 2 ** 2,num_heads[2],bias=bias) for i in range(depth[2])]
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2)
        )
        self.middle = nn.Sequential(
            *[MARformerBlock(dim* 2 ** 3,num_heads[3],bias=bias) for i in range(depth[3])],
        )
        self.de3 = nn.Sequential(
            *[MARformerBlock(dim * 2 ** 2,num_heads[2],bias=bias) for i in range(depth[2])],
        )
        self.de2 = nn.Sequential(
            *[MARformerBlock(dim * 2 ** 1,num_heads[1],bias=bias) for i in range(depth[1])],
        )
        self.de1 = nn.Sequential(
            *[MARformerBlock(dim,num_heads[0],bias=bias) for i in range(depth[0])],
        )
        
        self.embedder = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.unembedder = nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.d1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*1, H/2, W/2]
        self.d2 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 1 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*2, H/4, W/4]
        self.d3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*3, H/8, W/8]

        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 3 * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.f3 = nn.Sequential(
            nn.Conv2d(in_channels=(dim * 2 ** 2) * 2, out_channels=(dim * 2 * 2), kernel_size=1, bias=bias),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(in_channels=(dim * 2 ** 1) * 2, out_channels=(dim * 2 ** 1), kernel_size=1, bias=bias),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=bias),
        )
        
    def forward(self,x0):
        x = self.embedder(x0)
        x1 = self.en1(x)
        x = self.down1(x)
        x2 = self.en2(x)
        x = self.down2(x)
        x3 = self.en3(x)
        x = self.down3(x)
        x = self.middle(x)

        x = self.up3(x)
        #print(f"x.shape = {x.shape}, x3.shape={x3.shape}")
        x = torch.cat((x, x3), dim=1)
        x = self.f3(x)
        x = self.de3(x)

        x = self.up2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.f2(x)
        x = self.de2(x)
        
        x = self.up1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.f1(x)
        x = self.de1(x)
        x = self.unembedder(x)
        return x+x0
"""
large = MARformer(dim=48,depth=[1,2,4,8],num_heads=[1,2,4,8])
baseline = MARformer(dim=48,depth=[1,2,3,4],num_heads=[1,2,4,8])
tiny = MARformer(dim=48,depth=[1,2,3,4],num_heads=[1,1,1,1])
"""