"""SSL pretraining entrypoint built on PyTorch Lightning."""

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader


from .device import resolve_lightning_accelerator
from src.models import SSLPretrainingLightningModule
from src.config import Configuration



def train_stage_ssl(CONFIG: Configuration, ssl_train_loader: DataLoader):
    # ======================== SSL MODULE & CONFIG ========================
    ssl_module = SSLPretrainingLightningModule(
        embed_dim=512,
        patch_size=16,
        in_channels=4,
        proj_dim=128,
        learning_rate=1e-4,
        weight_decay=1e-4,
        temperature=0.5,
        vit_depth=8,
        num_heads=8,
        dropout=0.1,
        vol_size=96,
    )
    ssl_accelerator, ssl_devices, ssl_device_label = resolve_lightning_accelerator(CONFIG)
    print(f" - [SSL] device: {ssl_device_label}")

    ssl_checkpoint_cb = ModelCheckpoint(
        dirpath=CONFIG.MODELS_PATH,
        filename="ssl_pretraining_best",
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

    # ======================== TRAINING ========================
    ssl_trainer.fit(ssl_module, train_dataloaders=ssl_train_loader)

    if not ssl_checkpoint_cb.best_model_path:
        raise RuntimeError("No SSL checkpoint was saved by ModelCheckpoint.")

    print(f" - Best SSL checkpoint: {ssl_checkpoint_cb.best_model_path}")
    return ssl_checkpoint_cb.best_model_path