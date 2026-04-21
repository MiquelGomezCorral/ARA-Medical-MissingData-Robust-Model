"""SSL pretraining entrypoint built on PyTorch Lightning."""

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader


from .device import resolve_lightning_accelerator
from src.models import SSLPretrainingLightningModule
from src.config import Configuration


def _build_ssl_module(CONFIG: Configuration) -> SSLPretrainingLightningModule:
    return SSLPretrainingLightningModule(
        embed_dim=CONFIG.ssl_embed_dim,
        patch_size=CONFIG.ssl_patch_size,
        in_channels=4,
        proj_dim=CONFIG.ssl_proj_dim,
        learning_rate=CONFIG.ssl_learning_rate,
        weight_decay=CONFIG.ssl_weight_decay,
        temperature=CONFIG.ssl_temperature,
        vit_depth=CONFIG.ssl_vit_depth,
        num_heads=CONFIG.ssl_num_heads,
        dropout=CONFIG.ssl_dropout,
        vol_size=CONFIG.ssl_vol_size,
    )


def _run_ssl_stage(
    CONFIG: Configuration,
    ssl_train_loader: DataLoader,
    ssl_module: SSLPretrainingLightningModule,
    checkpoint_filename: str,
):
    ssl_accelerator, ssl_devices, ssl_device_label = resolve_lightning_accelerator(CONFIG)
    print(f" - [SSL] device: {ssl_device_label}")

    ssl_checkpoint_cb = ModelCheckpoint(
        dirpath=CONFIG.MODELS_PATH,
        filename=checkpoint_filename,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
    )

    ssl_trainer = L.Trainer(
        max_epochs=CONFIG.ssl_epochs,
        accelerator=ssl_accelerator,
        devices=ssl_devices,
        callbacks=[
            ssl_checkpoint_cb,
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                patience=2,
                min_delta=1e-4,
            ),
        ],
        default_root_dir=CONFIG.LOGS_PATH,
        log_every_n_steps=1,
        enable_progress_bar=True,
        profiler="simple",
    )

    ssl_trainer.fit(ssl_module, train_dataloaders=ssl_train_loader)

    if not ssl_checkpoint_cb.best_model_path:
        raise RuntimeError("No SSL checkpoint was saved by ModelCheckpoint.")

    print(f" - Best SSL checkpoint: {ssl_checkpoint_cb.best_model_path}")
    return ssl_checkpoint_cb.best_model_path


def train_stage_vit_pretraining(CONFIG: Configuration, ssl_train_loader: DataLoader):
    """Stage 0: pretrain ViT encoder before SSL and survival stages."""
    ssl_module = _build_ssl_module(CONFIG)
    return _run_ssl_stage(
        CONFIG,
        ssl_train_loader,
        ssl_module,
        checkpoint_filename="vit_pretraining_best",
    )



def train_stage_ssl(
    CONFIG: Configuration,
    ssl_train_loader: DataLoader,
    init_checkpoint_path: str | None = None,
):
    # ======================== SSL MODULE & CONFIG ========================
    ssl_module = _build_ssl_module(CONFIG)

    if init_checkpoint_path:
        ckpt = torch.load(init_checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        ssl_module.load_state_dict(state_dict, strict=False)
        print(f" - Loaded pretraining checkpoint for SSL stage: {init_checkpoint_path}")

    return _run_ssl_stage(
        CONFIG,
        ssl_train_loader,
        ssl_module,
        checkpoint_filename="ssl_pretraining_best",
    )