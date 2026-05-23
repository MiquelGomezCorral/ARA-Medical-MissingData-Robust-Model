"""
Multimodal Survival Predictor — token-wise sigmoid-gated fusion.

Architecture
============
  Image  → ViT3D       → img_tokens   (B, S, D)         S=217 with default config
  Rads   → shared MLP  → rad_tokens   (B, R*4, D)        R=3 regions × 4 channels = 12 tokens
  Masks  → concat      → rad_mask     (B, R*4)           True = valid token

  rad_tokens --masked mean pool--> rad_summary  (B, D)

  gate_input = cat(img_tokens, expand(rad_summary))      (B, S, 2D)
  sigma      = sigmoid( Linear(2D→D) )                   (B, S, D)
  fused      = img_tokens * sigma + rad_summary * (1-sigma)   (B, S, D)

  Tabular → TabularTokenizer → tab_tokens  (B, T, D)

  ca_fused_on_tab : Q=fused,      KV=tab_tokens  → fused_out  (B, S, D)
  ca_tab_on_fused : Q=tab_tokens, KV=fused       → tab_out    (B, T, D)

  fused_pool = fused_out.mean(1)    (B, D)
  tab_pool   = tab_out.mean(1)      (B, D)

  ca_final : Q=KV=stack([fused_pool, tab_pool])  → (B, 2, D)
  logits   = classifier(mean-pool)               → (B, num_classes)

Mask conventions
================
  radiomic_mask  (B, 4)  per region, bool  True = channel token is VALID
  After concat across regions → (B, R*4)

  tab_pad_mask   (B, T)  bool  True = PAD token  (standard PyTorch MHA convention)

Missing modality philosophy
============================
There are NO explicit if-branches for missing modalities.
Missingness is handled entirely through masks:
  - invalid radiomic tokens are zeroed before and after MLP
  - masked mean pool ignores them for the summary vector
  - a fully-absent sample produces rad_summary≈0, gate learns σ→1 (pure image)
  - the model generalises to any missingness pattern at inference time
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.vit_encoder_3d import ViTEncoder3D
from src.models.tabular_tokenizer import TabularTokenizer
from src.models.cross_attention import CrossAttentionBlock

REGIONS = ["ED", "ET", "NC"]       # R = 3
CHANNELS_PER_REGION = 4            # → 12 radiomic tokens total


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def masked_mean_pool(
    tokens: torch.Tensor,   # (B, N, D)
    mask: torch.Tensor,     # (B, N)  bool  True=valid
) -> torch.Tensor:          # (B, D)
    """
    Mean-pool over valid tokens only.
    Samples where every token is masked return a zero vector;
    the sigmoid gate will respond by pushing sigma toward 1 (pure image).
    """
    valid = mask.unsqueeze(-1).float()           # (B, N, 1)
    summed = (tokens * valid).sum(dim=1)         # (B, D)
    n_valid = valid.sum(dim=1).clamp(min=1.0)    # (B, 1)  avoid /0
    return summed / n_valid                      # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Radiomic token MLP
# ─────────────────────────────────────────────────────────────────────────────

class RadiomicTokenMLP(nn.Module):
    """
    Shared MLP that projects each radiomic channel-token independently.

    Input:  x     (B, N_tok, F)
            mask  (B, N_tok)  bool  True=valid
    Output: proj  (B, N_tok, D)   — invalid tokens are zeroed out
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        hidden_dim: int | None = None,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2

        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embed_dim))

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,       # (B, N_tok, F)
        mask: torch.Tensor,    # (B, N_tok)  bool  True=valid
    ) -> torch.Tensor:         # (B, N_tok, D)
        proj = self.norm(self.mlp(x))
        # Zero invalid tokens so they contribute nothing to downstream ops
        proj = proj * mask.unsqueeze(-1).float()
        return proj


# ─────────────────────────────────────────────────────────────────────────────
# Token-wise sigmoid gate
# ─────────────────────────────────────────────────────────────────────────────

class TokenWiseSigmoidGate(nn.Module):
    """
    Computes a per-token, per-dimension gate between image tokens and the
    radiomic summary.

    gate_input : cat(img_tokens, rad_summary_expanded)   (B, S, 2D)
    sigma       = sigmoid( Linear(2D → D) )              (B, S, D)
    output      = img_tokens * sigma + rad_summary * (1 - sigma)

    Because rad_summary comes from masked_mean_pool, a sample with no valid
    radiomic tokens will have rad_summary≈0 and the network will learn
    sigma→1, recovering pure image tokens — no explicit if-branch needed.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        img_tokens: torch.Tensor,    # (B, S, D)
        rad_summary: torch.Tensor,   # (B, D)
    ) -> torch.Tensor:               # (B, S, D)
        S = img_tokens.size(1)
        # Expand summary to match sequence length
        rad_exp = rad_summary.unsqueeze(1).expand(-1, S, -1)   # (B, S, D)

        gate_in = torch.cat([img_tokens, rad_exp], dim=-1)     # (B, S, 2D)
        sigma = torch.sigmoid(self.gate_fc(gate_in))           # (B, S, D)

        return img_tokens * sigma + rad_exp * (1.0 - sigma)    # (B, S, D)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalSurvivalPredictor(nn.Module):
    """
    Multimodal survival predictor with token-wise sigmoid-gated radiomic fusion.

    Parameters
    ----------
    num_classes          : output classes (e.g. 2 for binary survival)
    embed_dim            : shared token dimension D
    num_heads            : attention heads (must divide embed_dim)
    dropout              : dropout probability
    in_channels          : MRI modalities fed to the ViT (default 4)
    patch_size           : ViT 3-D patch size
    vit_depth            : number of ViT transformer blocks
    vol_size             : spatial cube size of the input volume
    tabular_in           : raw tabular feature count
    tabular_tokens       : number of tokens from TabularTokenizer
    tabular_hidden       : hidden dim inside TabularTokenizer
    radiomic_n_features  : feature count F per (region, channel) pair
    radiomic_mlp_hidden  : hidden dim of RadiomicTokenMLP (None → 2*embed_dim)
    radiomic_mlp_layers  : depth of RadiomicTokenMLP
    """

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
        radiomic_mlp_hidden: int | None = None,
        radiomic_mlp_layers: int = 3,
        pos_embed: str = "1d",
        use_radiomics: bool = False,
    ):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────
        self.image_encoder = ViTEncoder3D(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
            dropout=dropout,
            vol_size=vol_size,
            pos_embed=pos_embed,
        )
        self.tabular_tokenizer = TabularTokenizer(
            in_features=tabular_in,
            num_tokens=tabular_tokens,
            embed_dim=embed_dim,
            hidden_dim=tabular_hidden,
            dropout=dropout,
        )

        self.use_radiomics = use_radiomics

        if self.use_radiomics:
            self.radiomic_mlp = RadiomicTokenMLP(
                n_features=radiomic_n_features,
                embed_dim=embed_dim,
                hidden_dim=radiomic_mlp_hidden,
                n_layers=radiomic_mlp_layers,
                dropout=dropout,
            )

            self.image_mask_proj = nn.Sequential(
                nn.Linear(in_channels, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(self.image_mask_proj[-1].weight)
            nn.init.zeros_(self.image_mask_proj[-1].bias)

            self.sigmoid_gate = TokenWiseSigmoidGate(embed_dim)

        # ── Two-way cross-attention: fused ↔ tabular ─────────────
        self.ca_fused_on_tab = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.ca_tab_on_fused = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)

        # ── Final cross-attention over pooled streams ─────────────
        self.ca_final = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)

        # ── Survival head ─────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,                          # (B, C, D, D, D)
        tabular: torch.Tensor,                        # (B, N_tab)
        radiomic: dict[str, torch.Tensor],            # {'ED': (B,4,F), 'ET':…, 'NC':…}
        radiomic_mask: dict[str, torch.Tensor],       # {'ED': (B,4), …}  True=valid
        tabular_mask: torch.Tensor | None = None,     # (B, N_tab)  True=present
        image_mask: torch.Tensor | None = None,       # (B, C)      True=present
    ) -> torch.Tensor:                                # (B, num_classes)

        # ── 1. Image encoding ──────────────────────────────────────────
        img_tokens = self.image_encoder(image)        # (B, S, D)

        # ── 2-4. Radiomic pipeline (optional) ───────────────────────────
        if self.use_radiomics:
            rad_x = torch.cat(
                [radiomic[r] for r in REGIONS], dim=1
            )                                             # (B, R*4, F)
            rad_mask = torch.cat(
                [radiomic_mask[r] for r in REGIONS], dim=1
            )                                             # (B, R*4)  bool

            rad_tokens = self.radiomic_mlp(rad_x, rad_mask)   # (B, R*4, D)

            rad_summary = masked_mean_pool(rad_tokens, rad_mask)   # (B, D)

            if image_mask is not None:
                img_mask_feat = self.image_mask_proj(image_mask)
                rad_summary = rad_summary + img_mask_feat

            fused_tokens = self.sigmoid_gate(img_tokens, rad_summary)  # (B, S, D)
        else:
            fused_tokens = img_tokens  # (B, S, D)

        # ── 5. Tabular tokenisation ───────────────────────────────────
        tab_tokens = self.tabular_tokenizer(tabular, mask=tabular_mask)   # (B, T, D)

        # ── 6. Two-way cross-attention ────────────────────────────────
        # A: fused image tokens attend to tabular tokens
        fused_out = self.ca_fused_on_tab(
            query=fused_tokens,
            context=tab_tokens,
            # context_key_padding_mask=tab_pad_mask,    # (B,T) True=PAD
        )                                             # (B, S, D)

        # B: tabular tokens attend to fused image tokens
        tab_out = self.ca_tab_on_fused(
            query=tab_tokens,
            context=fused_tokens,
            # image tokens are always fully valid — no key_padding_mask
        )                                             # (B, T, D)

        # ── 7. Pool both streams ──────────────────────────────────────
        fused_pool = fused_out.mean(dim=1)            # (B, D)
        tab_pool   = tab_out.mean(dim=1)              # (B, D)

        # ── 8. Final cross-attention ──────────────────────────────────
        # 2-token sequence; self-attention learns to weight both streams
        combined  = torch.stack([fused_pool, tab_pool], dim=1)  # (B, 2, D)
        final_out = self.ca_final(query=combined, context=combined)  # (B, 2, D)
        pooled    = final_out.mean(dim=1)             # (B, D)

        # ── 9. Survival head ──────────────────────────────────────────
        logits = self.classifier(self.dropout(pooled))  # (B, num_classes)
        return logits

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    def load_pretrained_encoder(
        self,
        checkpoint_path: str,
        strict: bool = False,
        freeze: bool = False,
    ) -> None:
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
        missing, unexpected = self.image_encoder.load_state_dict(
            encoder_sd, strict=strict
        )
        print(
            f"[pretrained encoder] missing={len(missing)}  "
            f"unexpected={len(unexpected)}"
        )
        if freeze:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self.image_encoder.eval()