"""
Tabular Tokenizer (FC Network Branch).

Maps a flat clinical feature vector of shape (B, in_features)
to a sequence of M tokens of shape (B, num_tokens, embed_dim).
"""

import torch
import torch.nn as nn


class TabularTokenizer(nn.Module):
    def __init__(
        self,
        in_features: int = 14,
        num_tokens: int = 8,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.net(x)
        tokens = tokens.view(B, self.num_tokens, self.embed_dim)
        return self.norm(tokens)