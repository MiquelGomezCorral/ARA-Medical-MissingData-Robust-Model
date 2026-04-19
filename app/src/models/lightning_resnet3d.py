"""Lightning module for training FastResNet3DClassifier."""

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18

from src.models.resnet3d import FastResNet3DClassifier


class FastResNet3DLightningModule(L.LightningModule):
    """Lightning module wrapping FastResNet3DClassifier training logic."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        learning_rate: float = 1e-3,
        input_size: int | None = 96,
        dropout: float = 0.2,
        label_smoothing: float = 0.0,
        weight_decay: float = 1e-4,
        use_binary_bce: bool = False,
        use_pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_binary_bce = use_binary_bce and num_classes == 2
        output_dim = 1 if self.use_binary_bce else num_classes

        self.model = self._build_model(
            num_classes=output_dim,
            base_channels=base_channels,
            dropout=dropout,
            use_pretrained_backbone=use_pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )
        if self.use_binary_bce:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _build_model(
        self,
        num_classes: int,
        base_channels: int,
        dropout: float,
        use_pretrained_backbone: bool,
        freeze_backbone: bool,
    ) -> nn.Module:
        if not use_pretrained_backbone:
            return FastResNet3DClassifier(
                in_channels=4,
                num_classes=num_classes,
                base_channels=base_channels,
                layers=(1, 1, 1, 1),
                dropout=dropout,
            )


        model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Adapt the stem to 4 MRI channels while reusing pretrained RGB filters.
        old_conv = model.stem[0]
        if old_conv.in_channels != 4:
            new_conv = nn.Conv3d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            model.stem[0] = new_conv

        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
        if self.hparams.input_size is not None:
            images = F.interpolate(
                images,
                size=(self.hparams.input_size, self.hparams.input_size, self.hparams.input_size),
                mode="trilinear",
                align_corners=False,
            )
        return images

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = self._prepare_images(batch["image"])
        labels = batch["label"].view(-1)

        logits = self.model(images)
        if self.use_binary_bce:
            logits = logits.view(-1)
            targets = labels.float()
            loss = self.criterion(logits, targets)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            labels_for_acc = labels.long()
        else:
            labels = labels.long()
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)
            labels_for_acc = labels

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered. Consider disabling AMP and/or checking input tensors for bad values.")

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", (preds == labels_for_acc).float().mean(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._step(batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
