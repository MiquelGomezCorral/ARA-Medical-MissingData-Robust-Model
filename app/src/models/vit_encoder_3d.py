"""
3D Vision Transformer Encoder.

Patchifies (B, C, H, W, D) MRI volumes and produces
a sequence of L patch tokens of dimension `embed_dim`.
A learnable [CLS] token is prepended.
"""

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 16,
        embed_dim: int = 256,
        vol_size: int = 96,
    ):
        super().__init__()
        assert vol_size % patch_size == 0, (
            f"vol_size ({vol_size}) must be divisible by patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        n = vol_size // patch_size
        self.num_patches = n ** 3

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D = x.shape[:2]
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        vol_size: int = 96,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.vol_size = vol_size

        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim, vol_size)
        L = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, L + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] == (self.vol_size, self.vol_size, self.vol_size):
            return x
        return torch.nn.functional.interpolate(
            x,
            size=(self.vol_size, self.vol_size, self.vol_size),
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_input(x)
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)