"""
Self-Supervised Pretraining Module (Figure 1-A).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vit_encoder_3d import ViTEncoder3D


class ContrastiveHead(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(cls_token), dim=-1)


class ReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int, patch_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, patch_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(patch_tokens)


class SSLPretraining(nn.Module):
    def __init__(
        self,
        encoder: ViTEncoder3D,
        embed_dim: int = 256,
        patch_size: int = 16,
        in_channels: int = 4,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.in_channels = in_channels

        patch_dim = in_channels * (patch_size ** 3)
        self.contrastive_head = ContrastiveHead(embed_dim, proj_dim)
        self.reconstruction_head = ReconstructionHead(embed_dim, patch_dim)

    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        target_size = getattr(self.encoder, "vol_size", x.shape[-1])
        if x.shape[-3:] == (target_size, target_size, target_size):
            return x
        return torch.nn.functional.interpolate(
            x,
            size=(target_size, target_size, target_size),
            mode="trilinear",
            align_corners=False,
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p, D // p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, -1, C * p * p * p)
        return x

    @staticmethod
    def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / temperature
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        temperature: float = 0.5,
    ) -> tuple[torch.Tensor, float, float]:
        x_i = self._resize_input(x_i)
        x_j = self._resize_input(x_j)
        tokens_i = self.encoder(x_i)
        tokens_j = self.encoder(x_j)

        z_i = self.contrastive_head(tokens_i[:, 0])
        z_j = self.contrastive_head(tokens_j[:, 0])
        l_contrast = self._nt_xent_loss(z_i, z_j, temperature)

        patch_tokens_i = tokens_i[:, 1:]
        recon = self.reconstruction_head(patch_tokens_i)
        target = self._extract_patches(x_i)
        l_recon = F.mse_loss(recon, target)

        l_ssl = l_recon + l_contrast * l_recon
        return l_ssl, l_contrast.item(), l_recon.item()