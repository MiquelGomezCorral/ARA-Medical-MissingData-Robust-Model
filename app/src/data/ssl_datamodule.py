"""Lightning DataModule for SSL pretraining."""

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Configuration
from .UPennGBMDataset import UPennGBMDataset
from .augmentations_3d import MRIAugmentPipeline, TwoViewTransform


class SSLImageDataset(Dataset):
    """Return two augmented views of each MRI sample."""

    def __init__(self, base_dataset, two_view_transform: TwoViewTransform):
        self.ds = base_dataset
        self.transform = two_view_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        xi, xj = self.transform(sample["image"])
        return xi, xj


class SSLDataModule(L.LightningDataModule):
    """Simple data module that exposes a single train loader for SSL."""

    def __init__(
        self,
        config: Configuration,
        batch_size: int = 4,
        num_workers: int = 0,
        vol_size: int = 96,
        noise_std: float = 0.05,
        crop_scale: float = 0.85,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vol_size = vol_size
        self.noise_std = noise_std
        self.crop_scale = crop_scale

    def setup(self, stage: str | None = None) -> None:
        train_base = UPennGBMDataset(self.config, partition="train")
        val_base = UPennGBMDataset(self.config, partition="val")
        test_base = UPennGBMDataset(self.config, partition="test")
        combined = torch.utils.data.ConcatDataset([train_base, val_base, test_base])
        aug = MRIAugmentPipeline(vol_size=self.vol_size, noise_std=self.noise_std, crop_scale=self.crop_scale)
        self.train_ds = SSLImageDataset(combined, TwoViewTransform(aug))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )