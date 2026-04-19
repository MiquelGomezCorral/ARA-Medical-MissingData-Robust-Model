"""Downstream survival training entrypoint built on PyTorch Lightning."""

import os
from typing import Optional

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.config import Configuration
from src.data import UPennGBMDataModule
from src.models.survival_lightning import MultimodalSurvivalLightningModule
from .device import resolve_lightning_accelerator


def train_multimodal_survival(
    config: Configuration,
    ssl_checkpoint: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    num_workers: int = 2,
    embed_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.1,
    patch_size: int = 16,
    vit_depth: int = 4,
    vol_size: int = 96,
    tabular_tokens: int = 8,
    tabular_hidden: int = 128,
    freeze_encoder: bool = False,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    checkpoint_name: str = "survival_checkpoint.pt",
) -> MultimodalSurvivalLightningModule:
    """Train or fine-tune the multimodal survival predictor with Lightning."""
    accelerator, devices, device_label = resolve_lightning_accelerator(config)
    num_classes = len(config.labels)

    print(f"[Survival] device: {device_label}  num_classes: {num_classes}")

    data_module = UPennGBMDataModule(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup("fit")
    data_module.setup("test")
    train_ds = data_module.train_ds
    val_ds = data_module.val_ds
    test_ds = data_module.test_ds
    print(f"[Survival] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    model = MultimodalSurvivalLightningModule(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        in_channels=4,
        patch_size=patch_size,
        vit_depth=vit_depth,
        vol_size=vol_size,
        tabular_in=len(train_ds.tabular_cols),
        tabular_tokens=tabular_tokens,
        tabular_hidden=tabular_hidden,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )

    if ssl_checkpoint is not None:
        ckpt_path = os.path.join(config.MODELS_PATH, ssl_checkpoint)
        model.model.load_pretrained_encoder(ckpt_path, strict=False)
        print(f"[Survival] loaded ViT weights from {ckpt_path}")

    if freeze_encoder:
        for p in model.model.image_encoder.parameters():
            p.requires_grad = False
        print("[Survival] ViT encoder frozen")

    ckpt_path = os.path.join(config.MODELS_PATH, checkpoint_name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.MODELS_PATH,
        filename=os.path.splitext(checkpoint_name)[0],
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
    )
    callbacks = [checkpoint_cb]
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_bacc",
                mode="max",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        default_root_dir=config.LOGS_PATH,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    if checkpoint_cb.best_model_path:
        model.load_state_dict(torch.load(checkpoint_cb.best_model_path, map_location="cpu", weights_only=True)["state_dict"])
        print(f"[Survival] checkpoint saved to {checkpoint_cb.best_model_path}")
    else:
        torch.save({"state_dict": model.state_dict()}, ckpt_path)
        print(f"[Survival] checkpoint saved to {ckpt_path}")

    return model