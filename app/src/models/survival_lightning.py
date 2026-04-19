"""Lightning module for multimodal survival prediction."""

import pytorch_lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from src.models.survival_predictor import MultimodalSurvivalPredictor


class MultimodalSurvivalLightningModule(L.LightningModule):
    """Lightning wrapper around the multimodal survival predictor."""

    def __init__(
        self,
        num_classes: int,
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
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultimodalSurvivalPredictor(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            in_channels=in_channels,
            patch_size=patch_size,
            vit_depth=vit_depth,
            vol_size=vol_size,
            tabular_in=tabular_in,
            tabular_tokens=tabular_tokens,
            tabular_hidden=tabular_hidden,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.model(image, tabular)

    def _step(self, batch, stage: str):
        images = batch["image"]
        tabular = batch["tabular"]
        labels = batch["label"].long()
        logits = self(images, tabular)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        bacc = balanced_accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_bacc", bacc, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        self._step(batch, stage="val")

    def test_step(self, batch, batch_idx: int):
        self._step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
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