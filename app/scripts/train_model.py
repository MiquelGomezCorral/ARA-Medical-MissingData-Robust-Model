import torch
from torch.utils.data import DataLoader

from maikol_utils.print_utils import print_separator

from src.data import BraTSSSLDataModule
from src.data import UPennGBMDataModule
from src.config import Configuration
from src.training import train_stage_vit_pretraining, train_stage_ssl, train_stage_survival, test_model
from src.utils import set_all_seeds


def train_3d_vit(CONFIG: Configuration):
    set_all_seeds(CONFIG.seed)
    print_separator("Preparing Data", sep_type='LONG')
    ssl_train_loader, survival_dm, train_loader, val_loader, test_loader = prepare_data(CONFIG)

    print_separator("Starting ViT Pretraining", sep_type='SUPER')
    vit_pretraining_checkpoint = train_stage_vit_pretraining(CONFIG, ssl_train_loader)

    print_separator("Starting SSL Pretraining", sep_type='SUPER')
    ssl_checkpoint_path = train_stage_ssl(
        CONFIG,
        ssl_train_loader,
        init_checkpoint_path=vit_pretraining_checkpoint,
    )


    print_separator("Starting Survival Pretraining", sep_type='SUPER')
    survival_module = train_stage_survival(
        CONFIG, ssl_checkpoint_path,
        survival_dm, train_loader, 
        val_loader, test_loader
    )

    print_separator("Starting Testing", sep_type='SUPER')
    results = test_model(CONFIG, survival_module, test_loader)


def train_ssl_pretraining(CONFIG: Configuration):
    set_all_seeds(CONFIG.seed)
    print_separator("Preparing SSL Data", sep_type='LONG')
    ssl_train_loader = prepare_ssl_data(CONFIG)

    print_separator("Starting ViT Pretraining", sep_type='SUPER')
    vit_pretraining_checkpoint = train_stage_vit_pretraining(CONFIG, ssl_train_loader)

    print_separator("Starting SSL Pretraining", sep_type='SUPER')
    ssl_checkpoint_path = train_stage_ssl(
        CONFIG,
        ssl_train_loader,
        init_checkpoint_path=vit_pretraining_checkpoint,
    )
    return ssl_checkpoint_path


def prepare_ssl_data(CONFIG: Configuration):
    ssl_dm = BraTSSSLDataModule(
        config=CONFIG,
        batch_size=CONFIG.ssl_batch_size,
        num_workers=CONFIG.ssl_num_workers,
        patch_size=CONFIG.ssl_aug_patch_size,
        cutout_min_ratio=CONFIG.ssl_cutout_min_ratio,
        cutout_max_ratio=CONFIG.ssl_cutout_max_ratio,
    )
    ssl_dm.setup()

    print(f" - SSL Train samples:      {len(ssl_dm.train_ds):>5}")
    return ssl_dm.train_dataloader()


def prepare_survival_data(CONFIG: Configuration):
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

    print(f" - Survival Train samples: {len(survival_dm.train_ds):>5}")
    print(f" - Survival Val samples:   {len(survival_dm.val_ds):>5}")
    print(f" - Survival Test samples:  {len(survival_dm.test_ds):>5}")

    return survival_dm, train_loader, val_loader, test_loader


def prepare_data(CONFIG: Configuration):
    ssl_train_loader = prepare_ssl_data(CONFIG)
    survival_dm, train_loader, val_loader, test_loader = prepare_survival_data(CONFIG)

    return ssl_train_loader, survival_dm, train_loader, val_loader, test_loader