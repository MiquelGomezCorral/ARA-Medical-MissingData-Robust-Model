import torch
from torch import nn

class RadiomicTokenizer(nn.Module):
    def __init__(self, n_features, embed_dim, dropout=0.1):
        super().__init__()
        # Una proyección por modalidad: (N_feat,) → (embed_dim,)
        self.proj = nn.Linear(n_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x:    (B, 4, N_feat)
        # mask: (B, 4)  — 1=válido, 0=nulo
        tokens = self.proj(x)                    # (B, 4, embed_dim)
        tokens = self.norm(self.dropout(tokens))
        tokens = tokens * mask.unsqueeze(-1)     # anular modalidades nulas
        return tokens                            # (B, 4, embed_dim)