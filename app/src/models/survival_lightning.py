"""Lightning module for multimodal survival prediction."""

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, f1_score

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
        radiomic_n_features: int = 144,
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
            radiomic_n_features=radiomic_n_features
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor, radiomic: torch.Tensor, radiomic_mask: torch.Tensor,
                tabular_mask: torch.Tensor | None = None, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(image, tabular, radiomic, radiomic_mask,
                          tabular_mask=tabular_mask, image_mask=image_mask)

    def _step(self, batch, stage: str):
        images = batch["image"]
        tabular = batch["tabular"]
        labels = batch["label"].long()
        radiomic={
            'ED': batch['radiomic_ED'],
            'ET': batch['radiomic_ET'],
            'NC': batch['radiomic_NC'],
        }
        radiomic_mask={
            'ED': batch['radiomic_mask_ED'],
            'ET': batch['radiomic_mask_ET'],
            'NC': batch['radiomic_mask_NC'],
        }

        logits = self(images, tabular, radiomic, radiomic_mask,
                      tabular_mask=batch.get("tabular_mask"),
                      image_mask=batch.get("image_mask"))
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        bacc = balanced_accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_bacc", bacc, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
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
        total = max(int(self.trainer.max_epochs), 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }