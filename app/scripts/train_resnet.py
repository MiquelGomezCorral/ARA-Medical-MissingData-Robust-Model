"""Fast training script for UPenn GBM tensors with Lightning and 3D ResNet modules."""

import os
from glob import glob

import torch

from src.config import Configuration
from src.data import UPennGBMDataModule
from src.models import FastResNet3DLightningModule
from src.training.device import resolve_lightning_accelerator


def train_fast_resnet(
    config: Configuration,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    num_workers: int = 4,
    input_size: int | None = 96,
    base_channels: int = 16,
    dropout: float = 0.2,
    label_smoothing: float = 0.0,
    weight_decay: float = 1e-4,
    use_binary_bce: bool = False,
    use_amp: bool = True,
    use_pretrained_backbone: bool = False,
    freeze_backbone: bool = False,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    gradient_clip_val: float = 1.0,
):
    """Train a 3D ResNet quickly using PyTorch Lightning."""
    import pytorch_lightning as L  # type: ignore[reportMissingImports]
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore[reportMissingImports]

    accelerator, devices, device_label = resolve_lightning_accelerator(config)
    precision = "16-mixed" if use_amp and accelerator == "gpu" else "32-true"
    print(f"[ResNet] device: {device_label}")

    data_module = UPennGBMDataModule(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = FastResNet3DLightningModule(
        num_classes=len(config.labels),
        base_channels=base_channels,
        learning_rate=learning_rate,
        input_size=input_size,
        dropout=dropout,
        label_smoothing=label_smoothing,
        weight_decay=weight_decay,
        use_binary_bce=use_binary_bce,
        use_pretrained_backbone=use_pretrained_backbone,
        freeze_backbone=freeze_backbone,
    )

    os.makedirs(config.MODELS_PATH, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.MODELS_PATH,
        filename="fast_resnet3d_best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    callbacks = [checkpoint_cb]
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                mode="max",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        default_root_dir=config.LOGS_PATH,
        log_every_n_steps=1,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model=model, datamodule=data_module)

    best_path = checkpoint_cb.best_model_path
    print(f"Best checkpoint saved to: {best_path}")

    return model


def test_fast_resnet(
    config: Configuration,
    batch_size: int = 2,
    num_workers: int = 4,
    checkpoint_path: str | None = None,
):
    """Evaluate a trained Lightning checkpoint on the test split."""
    import pytorch_lightning as L  # type: ignore[reportMissingImports]

    accelerator, devices, device_label = resolve_lightning_accelerator(config)
    print(f"[ResNet] device: {device_label}")

    data_module = UPennGBMDataModule(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if checkpoint_path is None:
        pattern = os.path.join(config.MODELS_PATH, "fast_resnet3d_best*.ckpt")
        candidates = glob(pattern)
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found matching: {pattern}. Train first or pass checkpoint_path explicitly."
            )
        checkpoint_path = max(candidates, key=os.path.getmtime)

    model = FastResNet3DLightningModule.load_from_checkpoint(checkpoint_path)

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=config.LOGS_PATH,
    )

    metrics = trainer.test(model=model, datamodule=data_module)
    print(f"Tested checkpoint: {checkpoint_path}")
    print(f"Test metrics: {metrics}")
    return metrics
