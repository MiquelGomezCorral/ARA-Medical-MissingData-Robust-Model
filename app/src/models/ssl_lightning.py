"""Lightning module for SSL pretraining."""

import pytorch_lightning as L
import torch
import torch.nn as nn

from src.models.ssl_module import SSLPretraining
from src.models.vit_encoder_3d import ViTEncoder3D


class SSLPretrainingLightningModule(L.LightningModule):
    """Lightning wrapper around SSLPretraining."""

    def __init__(
        self,
        embed_dim: int = 256,
        patch_size: int = 16,
        in_channels: int = 4,
        proj_dim: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        temperature: float = 0.5,
        vit_depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        vol_size: int = 96,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = ViTEncoder3D(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
            dropout=dropout,
            vol_size=vol_size,
        )
        self.model = SSLPretraining(
            encoder=self.encoder,
            embed_dim=embed_dim,
            patch_size=patch_size,
            in_channels=in_channels,
            proj_dim=proj_dim,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        return self.model(x_i, x_j, target=target, temperature=self.temperature)

    def training_step(self, batch, batch_idx: int):
        if len(batch) == 3:
            x_i, x_j, target = batch
        else:
            x_i, x_j = batch
            target = None

        loss, l_contrast, l_recon = self(x_i, x_j, target=target)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_contrast", l_contrast, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_recon", l_recon, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(self.trainer.max_epochs), 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }