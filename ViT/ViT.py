"""
    Re-Implementation of Visual Transformers (ViT) based on:
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). 
    An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

    Has similarity with: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

class PatchEmbed(nn.Module):
    """Splits an image into patches and embeds them."""
    def __init__(self, img_size, patch_size, in_chan, embed_dim) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chan, embed_dim, patch_size, patch_size)
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        proj = self.proj(x).flatten(2).transpose(1, 2)
        return proj

class Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2).flatten(2)
        return self.proj_drop(self.proj(weighted_avg))

class MLP(nn.Module):
    """Feedforward MLP with GELU activation."""
    def __init__(self, input_features, hidden_features, p=0.) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, input_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))

class Block(nn.Module):
    """Transformer block consisting of multi-head attention and MLP."""
    def __init__(self, dim, n_heads=12, mlp_ratio=4, qkv_bias=True, p=0., attn_p=0.) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, dim * mlp_ratio, p)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class ViT(nn.Module):
    """Vision Transformer (ViT) model."""
    def __init__(self, args) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(args.img_size, args.patch_size, args.in_channel, args.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, args.embed_dim))
        self.pos_drop = nn.Dropout(args.p)
        self.blocks = nn.ModuleList([Block(args.embed_dim, args.n_heads, args.mlp_ratio, args.qkv_bias, args.p, args.atten_p) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.embed_dim, eps=1e-6)
        self.head = nn.Linear(args.embed_dim, args.n_classes)
    
    def forward(self, x):
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), self.patch_embed(x)], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_size', type=int, default=384)
    parser.add_argument('-patch_size', type=int, default=16)
    parser.add_argument('-in_channel', type=int, default=3)
    parser.add_argument('-embed_dim', type=int, default=768)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-mlp_ratio', type=int, default=4)
    parser.add_argument('-qkv_bias', type=bool, default=True)
    parser.add_argument('-p', type=float, default=0.1)
    parser.add_argument('-atten_p', type=float, default=0.1)
    parser.add_argument('-n_layers', type=int, default=12)
    parser.add_argument('-n_classes', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-lr', type=float, default=3e-4)
    parser.add_argument('-device', type=str, default='cuda')
    args = parser.parse_args()
    
    model = ViT(args).to(args.device)