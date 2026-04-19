"""
Cross-Attention Block.

Standard Transformer cross-attention where:
  - Q  comes from the "query" modality tokens
  - K,V come from the "context" modality tokens

The query tokens are updated; context tokens are unchanged.
Includes a post-attention LayerNorm + FFN (pre-norm style).

This implements the cross-attention boxes shown in Figure 1-B.
"""

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv)
        query = query + attn_out
        query = query + self.ffn(self.norm2(query))
        return query