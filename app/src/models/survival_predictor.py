"""
Multimodal Survival Predictor (Figure 1-B).
"""

import torch
import torch.nn as nn

from src.models.vit_encoder_3d import ViTEncoder3D
from src.models.tabular_tokenizer import TabularTokenizer
from src.models.cross_attention import CrossAttentionBlock
from src.models.radiomic_normalizer import RadiomicTokenizer

REGIONS = ['ED', 'ET', 'NC']

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
        radiomic_n_features: int = 144,
    ):
        super().__init__()

        # ── Encoders existentes ──────────────────────────────────
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

        # ── Tokenizadores radiómicos (uno por región) ────────────
        self.radiomic_tokenizers = nn.ModuleDict({
            region: RadiomicTokenizer(
                n_features=radiomic_n_features,
                embed_dim=embed_dim,
                dropout=dropout,
            )
            for region in REGIONS
        })

        # ── Cross-attention: radiomic Q, imagen K/V ──────────────
        # Cada región tiene su propio bloque
        self.ca_radiomic = nn.ModuleDict({
            region: CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for region in REGIONS
        })

        # ── Cross-attention existente tab ↔ img ──────────────────
        self.ca_tab_on_img = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.ca_img_on_tab = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)

        # ── Fusión final ─────────────────────────────────────────
        # Proyectar la concat de todo a embed_dim antes de clasificar
        total_streams = 3                    # tab_updated + img_updated + radiomic_fused
        self.fusion_proj = nn.Linear(embed_dim * total_streams, embed_dim)
        self.ca_final = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,                          # (B, 4, D, D, D)
        tabular: torch.Tensor,                        # (B, N_tab)
        radiomic: dict[str, torch.Tensor],            # {'ED': (B,4,F), 'ET':…, 'NC':…}
        radiomic_mask: dict[str, torch.Tensor],       # {'ED': (B,4), …}
    ) -> torch.Tensor:

        # 1. Encoders base
        img_tokens = self.image_encoder(image)        # (B, S_img, D)
        tab_tokens = self.tabular_tokenizer(tabular)  # (B, S_tab, D)

        # 2. Cross-attention existente tab ↔ img
        tab_updated = self.ca_tab_on_img(query=tab_tokens, context=img_tokens)
        img_updated = self.ca_img_on_tab(query=img_tokens, context=tab_tokens)

        # 3. Ramas radiómicas: Q=radiomic, K/V=img_tokens
        region_tokens = []
        for region in REGIONS:
            rad_tok = self.radiomic_tokenizers[region](
                radiomic[region],           # (B, 4, F)
                radiomic_mask[region],      # (B, 4)
            )                               # → (B, radiomic_tokens, D)

            rad_updated = self.ca_radiomic[region](
                query=rad_tok,              # Q: radiómicos buscan en imagen
                context=img_tokens,         # K/V: tokens de imagen
            )                               # → (B, radiomic_tokens, D)

            region_tokens.append(rad_updated.mean(dim=1))  # (B, D) por región

        # (B, 3*D) → (B, D)
        radiomic_fused = self.fusion_proj(
            torch.cat(region_tokens, dim=-1)
        )

        # 4. Pool tab e img para fusión final
        tab_pooled = tab_updated.mean(dim=1)           # (B, D)
        img_pooled = img_updated.mean(dim=1)           # (B, D)

        # 5. Concatenar los tres streams y clasificar
        fused = torch.stack(
            [tab_pooled, img_pooled, radiomic_fused], dim=1
        )                                              # (B, 3, D)
        out = self.ca_final(query=fused, context=fused)
        pooled = out.mean(dim=1)                       # (B, D)
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