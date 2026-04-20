"""Downstream survival training entrypoint built on PyTorch Lightning."""


import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.config import Configuration
from src.data.lightning_datamodule import UPennGBMDataModule
from src.models import MultimodalSurvivalLightningModule
from src.training import resolve_lightning_accelerator
from torch.utils.data import DataLoader, Subset


def train_stage_survival(
    CONFIG: Configuration, ssl_checkpoint_path: str, 
    survival_dm: UPennGBMDataModule, train_loader: DataLoader, 
    val_loader: DataLoader, test_loader: DataLoader
):
    # ======================== Survival Module & Config ========================  
    survival_module = MultimodalSurvivalLightningModule(
        num_classes=len(CONFIG.labels),
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        in_channels=4,
        patch_size=16,
        vit_depth=4,
        vol_size=96,
        tabular_in=len(survival_dm.train_ds.tabular_cols),
        tabular_tokens=8,
        tabular_hidden=128,
        learning_rate=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
    )

    survival_module.model.load_pretrained_encoder(ssl_checkpoint_path, strict=False)
    survival_accelerator, survival_devices, survival_device_label = resolve_lightning_accelerator(CONFIG)
    print(f" - [Survival] device: {survival_device_label}")

    survival_checkpoint_cb = ModelCheckpoint(
        dirpath=CONFIG.MODELS_PATH,
        filename="survival_checkpoint_best",
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
    )

    survival_trainer = L.Trainer(
        max_epochs=CONFIG.survival_epochs,
        accelerator=survival_accelerator,
        devices=survival_devices,
        callbacks=[
            survival_checkpoint_cb,
            EarlyStopping(
                monitor="val_bacc",
                mode="max",
                patience=2,
                min_delta=1e-4,
            ),
        ],
        default_root_dir=CONFIG.LOGS_PATH,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    # ======================== TRAINING & TESTING ========================
    survival_trainer.fit(survival_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    survival_trainer.test(survival_module, dataloaders=test_loader)

    if not survival_checkpoint_cb.best_model_path:
        raise RuntimeError("No survival checkpoint was saved by ModelCheckpoint.")

    print(f" - Best Survival checkpoint: {survival_checkpoint_cb.best_model_path}")
    return survival_module