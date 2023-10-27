from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from typing import Tuple
from ..utils import get_3d_sincos_pos_embed
from .common import LayerNorm3d

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        out_chans=256,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(kernel_size=patch_size, stride=patch_size, in_chans=in_chans,
        embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) **3

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # print("self.pos_embed shape:", self.pos_embed.shape, "img_size:", img_size)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans
        self.img_size = img_size

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
        )

    def forward(self, x):
        B, C, H, W, Z = x.shape
        print("x:",x.device)
        
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        b,l,c = x.shape
        x = x.permute(0,2,1).reshape(b,c,14,14,14)
        # print("before neck:", x.shape)
        x = self.neck(x)
        # print("endoder output shape:", x.shape)

        return x
    
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.embed_dim = embed_dim
        self.patch_size = kernel_size
        self.lin = nn.Linear(kernel_size*3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W,Z = x.shape

        xy = torch.zeros(B,self.embed_dim,H // self.patch_size,W // self.patch_size,Z).to(x.device)
        xz = torch.zeros(B,self.embed_dim,H // self.patch_size,W,Z // self.patch_size).to(x.device)
        yz = torch.zeros(B,self.embed_dim,H,W // self.patch_size,Z // self.patch_size).to(x.device)
        for j in range(Z):
            # print(xy[:,:,:,:,j].shape, self.patch_embed(x[:,:,:,:,j]).shape)
            xy[:,:,:,:,j] = self.proj(x[:,:,:,:,j])
            xz[:,:,:,j,:] = self.proj(x[:,:,:,j,:])
            yz[:,:,j,:,:] = self.proj(x[:,:,j,:,:])
        xy = xy.reshape(B,self.embed_dim,H // self.patch_size,W // self.patch_size,Z // self.patch_size,self.patch_size)
        xz = xz.permute(0,1,2,4,3).reshape(B,self.embed_dim,H // self.patch_size,Z // self.patch_size,W // self.patch_size,self.patch_size).permute(0,1,2,4,3,5)
        yz = yz.permute(0,1,3,4,2).reshape(B,self.embed_dim,W // self.patch_size,Z // self.patch_size,H // self.patch_size,self.patch_size).permute(0,1,4,2,3,5)

        x = torch.cat((xy,xz,yz),dim=-1).to(x.device)
        x = self.lin(x).squeeze(-1)
        x = x.flatten(2).transpose(1, 2)
        print("patch embed x:", x.shape, x.dtype, "x grad_fn:", x.grad_fn)
        return x