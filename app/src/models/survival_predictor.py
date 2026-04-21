"""
Multimodal Survival Predictor (Figure 1-B).
"""

import torch
import torch.nn as nn

from src.models.vit_encoder_3d import ViTEncoder3D
from src.models.tabular_tokenizer import TabularTokenizer
from src.models.cross_attention import CrossAttentionBlock


class MultimodalSurvivalPredictor(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        in_channels: int = 4,
        patch_size: int = 16,
        vit_depth: int = 4,
        vol_size: int = 96,
        tabular_in: int = 14,
        tabular_tokens: int = 8,
        tabular_hidden: int = 128,
    ):
        super().__init__()
        self.image_encoder = ViTEncoder3D(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
            dropout=dropout,
            vol_size=vol_size,
        )
        self.tabular_tokenizer = TabularTokenizer(
            in_features=tabular_in,
            num_tokens=tabular_tokens,
            embed_dim=embed_dim,
            hidden_dim=tabular_hidden,
            dropout=dropout,
        )

        self.ca_tab_on_img = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.ca_img_on_tab = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.ca_final = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_tokens = self.image_encoder(image)
        tab_tokens = self.tabular_tokenizer(tabular)

        tab_updated = self.ca_tab_on_img(query=tab_tokens, context=img_tokens)
        img_updated = self.ca_img_on_tab(query=img_tokens, context=tab_tokens)
        fused = self.ca_final(query=tab_updated, context=img_updated)
        pooled = fused.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits

    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False, freeze: bool = False):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        encoder_sd = {
            k.removeprefix("encoder."): v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        if not strict:
            current_sd = self.image_encoder.state_dict()
            encoder_sd = {
                k: v
                for k, v in encoder_sd.items()
                if k in current_sd and current_sd[k].shape == v.shape
            }
        missing, unexpected = self.image_encoder.load_state_dict(encoder_sd, strict=strict)
        print(f"[pretrained encoder] missing={len(missing)}  unexpected={len(unexpected)}")

        if freeze:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False
            self.image_encoder.eval()