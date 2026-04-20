"""Fast training script for UPenn GBM tensors with Lightning and 3D ResNet modules."""

import torch
from src.data import SSLDataModule
from torch.utils.data import DataLoader

from maikol_utils.print_utils import print_separator

from src.data import UPennGBMDataModule
from src.config import Configuration
from src.training import train_stage_ssl, train_stage_survival, test_model


def train_3d_vit(CONFIG: Configuration):
    print_separator("Preparing Data", sep_type='LONG')
    ssl_train_loader, survival_dm, train_loader, val_loader, test_loader = prepare_data(CONFIG)


    print_separator("Starting SSL Pretraining", sep_type='SUPER')
    ssl_checkpoint_path = train_stage_ssl(CONFIG,ssl_train_loader)


    print_separator("Starting Survival Pretraining", sep_type='SUPER')
    survival_module = train_stage_survival(
        CONFIG, ssl_checkpoint_path,
        survival_dm, train_loader, 
        val_loader, test_loader
    )

    print_separator("Starting Testing", sep_type='SUPER')
    test_model(CONFIG, survival_module, test_loader)


def prepare_data(CONFIG: Configuration):
    # ======================== SSL DataModule ========================
    ssl_dm = SSLDataModule(
        config=CONFIG,
        batch_size=16,
        num_workers=4,
        vol_size=96,
        noise_std=0.05,
        crop_scale=0.85,
    )
    ssl_dm.setup()

    ssl_train_loader = DataLoader(
        ssl_dm.train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers = True,
        prefetch_factor=4,
    )


    # ======================== Survival DataModule ========================
    survival_dm = UPennGBMDataModule(
        config=CONFIG,
        batch_size=16,
        num_workers=4,
    )
    survival_dm.setup("fit")
    survival_dm.setup("test")

    train_loader = DataLoader(
        survival_dm.train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers = True
    )
    val_loader = DataLoader(
        survival_dm.val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers = True
    )
    test_loader = DataLoader(
        survival_dm.test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers = True
    )


    # ======================== Log ========================
    print(f" - SSL Train samples: {len(ssl_dm.train_ds):>5}")
    print(f" - Survival Train samples: {len(survival_dm.train_ds):>5}")
    print(f" - Survival Val samples: {len(survival_dm.val_ds):>5}")
    print(f" - Survival Test samples: {len(survival_dm.test_ds):>5}")

    return ssl_train_loader, survival_dm, train_loader, val_loader, test_loader