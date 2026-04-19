"""SSL pretraining entrypoint built on PyTorch Lightning."""

import os

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.config import Configuration
from src.data.ssl_datamodule import SSLDataModule
from src.models.ssl_lightning import SSLPretrainingLightningModule
from .device import resolve_lightning_accelerator


def train_ssl(
    config:           Configuration,
    epochs:           int   = 50,
    batch_size:       int   = 4,
    learning_rate:    float = 1e-4,
    weight_decay:     float = 1e-4,
    num_workers:      int   = 2,
    embed_dim:        int   = 256,
    patch_size:       int   = 16,
    vit_depth:        int   = 4,
    num_heads:        int   = 8,
    dropout:          float = 0.1,
    vol_size:         int   = 96,
    temperature:      float = 0.5,
    proj_dim:         int   = 128,
    noise_std:        float = 0.05,
    crop_scale:       float = 0.85,
    checkpoint_name:  str   = "ssl_checkpoint.pt",
    enable_early_stopping:    bool  = True,
    early_stopping_patience:  int   = 8,
    early_stopping_min_delta: float = 1e-4,
) -> SSLPretrainingLightningModule:
    """Pretrain the ViT encoder with SSL using a Lightning trainer."""
    accelerator, devices, device_label = resolve_lightning_accelerator(config)
    print(f"[SSL] device: {device_label}")

    data_module = SSLDataModule(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        vol_size=vol_size,
        noise_std=noise_std,
        crop_scale=crop_scale,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    print(f"[SSL] dataset size: {len(data_module.train_ds)}  batches/epoch: {len(train_loader)}")

    ssl_model = SSLPretrainingLightningModule(
        embed_dim=embed_dim,
        patch_size=patch_size,
        in_channels=4,
        proj_dim=proj_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        vit_depth=vit_depth,
        num_heads=num_heads,
        dropout=dropout,
        vol_size=vol_size,
    )

    ckpt_path = os.path.join(config.MODELS_PATH, checkpoint_name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.MODELS_PATH,
        filename=os.path.splitext(checkpoint_name)[0],
        monitor="train_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [checkpoint_cb]
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                mode="min",
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
    trainer.fit(ssl_model, datamodule=data_module)

    if checkpoint_cb.best_model_path:
        ssl_model.load_state_dict(torch.load(checkpoint_cb.best_model_path, map_location="cpu", weights_only=True)["state_dict"])
        print(f"[SSL] checkpoint saved to {checkpoint_cb.best_model_path}")
    else:
        torch.save({"state_dict": ssl_model.state_dict()}, ckpt_path)
        print(f"[SSL] checkpoint saved to {ckpt_path}")

    return ssl_model