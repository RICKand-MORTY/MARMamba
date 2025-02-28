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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        #print(f"x.shape = {x.shape}")
        x = x.flatten(2).transpose(1, 2)
        #print(f"x.shape = {x.shape}")
        x = self.norm(x)
        return x

class UnPatchEmbedding(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=256, out_chans=3):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = patch_size // 2
        self.proj = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size // 2, patch_size // 2))

    def forward(self, x, H, W):
        # Reshape the embeddings back into patches
        #print(f"size = {size}")
        new_H = (H + 2 * self.padding - self.patch_size) // self.stride + 1
        new_W = (W + 2 * self.padding - self.patch_size) // self.stride + 1
        #print(f"sqrt= {int(math.sqrt(size))}")
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_H, w=new_W)
        # Reconstruct the image
        #print(f"x.shape = {x.shape}")
        x = self.proj(x)
        #print(f"x.shape = {x.shape}")
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        #print(f"x.shape = {x.shape}")
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
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class AMConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        #self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.gate = nn.Sigmoid()
        #self.conv_transpose = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        __, _, H, W = x.shape
        x_ = self.conv0(x)
        #max_pooled = self.max_pool(x_)
        avg_pooled = self.avg_pool(x_)
        gate = self.gate(avg_pooled)
        #gate = self.gate(max_pooled + avg_pooled)
        #fusion = gate * max_pooled + (1 - gate) * avg_pooled
        fusion = gate * avg_pooled
        fusion = F.interpolate(fusion, size=(H, W), mode='nearest')
        #fusion = self.conv_transpose(fusion)
        return fusion + x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = AMConv(hidden_features)
        self.act = act_layer()
        #self.fc_branch = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        #self.dwconv_branch = DWConv(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PVTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sra_size=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.sra_size = sra_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if self.sra_size != 0:
            self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
            self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias)
            self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
            self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=qkv_bias)

            # SRA
            self.pool = nn.AdaptiveAvgPool2d(sra_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = LayerNorm(dim, 'WithBias')
            self.act = nn.GELU()
        else:
            #NO SRA
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.sra_size != 0:
            q = self.q_dwconv(self.q(x))
            # Note that c need to be the last dim to ensure metric multiplication can run
            q = rearrange(q, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
            q = torch.nn.functional.normalize(q, dim=-1)
            # SRA
            x_ = self.sr(self.pool(x))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            #Generate kv
            kv = self.kv_dwconv(self.kv(x_))
            k, v = kv.chunk(2, dim=1)
            k = rearrange(k, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
            # print(f"k={k.shape}, v={v.shape}, q={q.shape}")
        else:
            #NO SRA
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        if self.sra_size != 0:
            x = rearrange(x, 'b head (h w) c  -> b (head c) h w', head=self.num_heads, h=H, w=W)
        else:
            x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
"""
class MambaAttention(nn.Module):
    def __init__(self, dim, bias=True, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        #self.act = nn.GELU()
        self.mamba_norm = nn.Sequential(
            Mamba(
            d_model=dim // 3,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ),
            nn.LayerNorm(dim // 3),
            nn.GELU()
        )
        self.mamba_flip1 = nn.Sequential(
            Mamba(
                d_model=dim // 3,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ),
            nn.LayerNorm(dim // 3 ),
            nn.GELU()
        )

        self.mamba_flip2 = nn.Sequential(
            Mamba(
                d_model=dim // 3,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ),
            nn.LayerNorm(dim // 3),
            nn.GELU()
        )
        #self.gate = nn.Sigmoid()
        self.proj_2 = nn.Conv2d(dim // 3, dim, kernel_size=1, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        __, _, H, W = x.shape
        shorcut = x.clone()
        x = self.proj_1(x)
        x, flip_x1, flip_x2 = torch.chunk(x, 3, dim=1)
        flip_x1 = flip_x1.flatten(2).transpose(1, 2)
        flip_x1 = torch.flip(flip_x1, dims=[-1])
        flip_x2 = flip_x2.flatten(2).transpose(1, 2)
        flip_x2 = torch.flip(flip_x2, dims=[-2])
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba_norm(x)
        flip_x1 = self.mamba_flip1(flip_x1)
        flip_x2 = self.mamba_flip2(flip_x2)
        flip_x1 = torch.flip(flip_x1, dims=[-1])
        flip_x2 = torch.flip(flip_x2, dims=[-2])
        x = x * flip_x1 * flip_x2
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        out = self.proj_2(x)
        #out = attn + shorcut
        return out
"""
class ViTBlock(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., LayerNorm_type='WithBias', sra_size=7, bias=True):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = PVTAttention(dim, qkv_bias=qkv_bias, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, sra_size=sra_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=nn.GELU, bias=bias)
        """
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Stage(nn.Module):

    def __init__(self,dim, depth, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., LayerNorm_type='WithBias', sra_size=7):
        super().__init__()
        #self.emb = PatchEmbedding(patch_size=patch_size, stride=stride, embed_dim=embed_dim, in_chans=in_chans)
        #self.unemb = UnPatchEmbedding(patch_size=patch_size, stride=stride, in_chans=embed_dim, out_chans=in_chans)
        self.blocks = nn.ModuleList([
            ViTBlock(dim=dim,qkv_bias=qkv_bias, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, sra_size=sra_size)
            for j in range(depth)
        ])

    def forward(self, x):
        #__, _, H, W = x.shape
        #x = self.emb(x)
        for block in self.blocks:
            x = block(x)
        #x = self.unemb(x, H, W)
        return x


class PVTFormer(nn.Module):

    def __init__(
            self,
            input_size=256,
            in_channels=3,
            depth=[1, 2, 2, 4, 1],
            num_heads=[4, 6, 6, 8],
            sra_size=[7, 7, 7, 7],
            dim = 12,
            bias=True,
            qkv_bias=True,
            LayerNorm_type='WithBias',
            attn_drop=0., 
            proj_drop=0.,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        #self.num_heads = num_heads

        print(
            f"depth={depth}, input_size={input_size}, dim={dim}, in_chann={in_channels}")
        self.down1_stage = Stage(dim=dim,depth=depth[0], num_heads=num_heads[0], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[0])
        self.down2_stage = Stage(dim=dim * 2 ** 1, depth=depth[1], num_heads=num_heads[1], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[1])
        self.down3_stage = Stage(dim=dim * 2 ** 2, depth=depth[2], num_heads=num_heads[2], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[2])
        self.down4_stage = Stage(dim=dim * 2 ** 3, depth=depth[3], num_heads=num_heads[3], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[3])
        self.up1_stage = Stage(dim=dim, depth=depth[0], num_heads=num_heads[2], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[2])
        self.up2_stage = Stage(dim=dim * 2 ** 1, depth= depth[1], num_heads=num_heads[1], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[1])
        self.up3_stage = Stage(dim=dim * 2 ** 2, depth= depth[2], num_heads=num_heads[0], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[0])
        self.embedder = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.unembedder = nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # Downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*1, H/2, W/2]

        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 1 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*2, H/4, W/4]
        self.down3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*3, H/8, W/8]

        # Upsample
        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 3 * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=(dim * 2 ** 2) * 2, out_channels=(dim * 2 * 2), kernel_size=1, bias=bias),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(in_channels=(dim * 2 ** 1) * 2, out_channels=(dim * 2 ** 1), kernel_size=1, bias=bias),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.fusion4 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=bias),
        )

        if depth[4] != 0:
            self.refine_stage = Stage(dim=dim, depth=depth[4], num_heads=num_heads[0], qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, LayerNorm_type='WithBias', sra_size=sra_size[0])
        else:
            print("No refineblock")
            self.refine = False
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def pad_to_multiple_of_eight(self, x):
        _, _, h, w = x.size()
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        x_padded = F.pad(x, (0, w_pad, 0, h_pad), 'constant', 0)
        return x_padded

    def forward(self, x):
        x_ori = x.clone()
        #print(f"x.shape={x.shape}")
        _, _, ori_h, ori_w = x.shape
        #align to input size
        x = self.pad_to_multiple_of_eight(x)
        #x = self.pad_to_multiple_of_eight(x)
        # print(f"t.shape={t.shape}")
        # print(f"x.shape={x.shape}")
        _, _, H, W = x.shape
        x = self.embedder(x)

        #Down 1
        x = self.down1_stage(x)

        # Down2
        x_to_s1 = x.clone()
        # print(f"bef down1 x.shape={x.shape}")
        x = self.down1(x)
        #print(f"aft down1 x.shape={x.shape}")
        x2 = x.clone()
        # Stage
        x = self.down2_stage(x)

        # Down 3
        x_to_s2 = x.clone()
        # print(f"bef down2 x.shape={x.shape}")
        x = self.down2(x)
        # print(f"aft down2 x.shape={x.shape}")
        # Stage
        x = self.down3_stage(x)

        # Down 4
        x_to_s3 = x.clone()
        x = self.down3(x)
        # Stage
        x = self.down4_stage(x)

        # Up 3
        x = self.up1(x)
        # print(f"x.shape={x.shape}")
        x = torch.cat((x, x_to_s3), dim=1)
        # print(f"x.shape={x.shape}")
        x = self.fusion2(x)
        # Stage
        x = self.up3_stage(x)


        # Up 2
        x = self.up2(x)
        x = torch.cat((x, x_to_s2), dim=1)
        x = self.fusion3(x)
        # Stage
        x = self.up2_stage(x)

        # Up 3
        x = self.up3(x)
        x = torch.cat((x, x_to_s1), dim=1)
        x = self.fusion4(x)

        # Stage
        x = self.up1_stage(x)

        # Refinement
        x = self.refine_stage(x)
        x = self.unembedder(x)
        x = x[:, :, :ori_h, :ori_w] + x_ori
        return x

