"""Lightning DataModule for UPenn GBM tensor data."""

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from src.config import Configuration
from .UPennGBMDataset import UPennGBMDataset
from .augmentations_3d import  SurvivalAugmentPipeline, SurvivalInferencePipeline

class UPennGBMDataModule(L.LightningDataModule):
    """Create train/val/test dataloaders from UPennGBMDataset."""

    def __init__(self, config: Configuration, batch_size: int = 2, num_workers: int = 4):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds: UPennGBMDataset | None = None
        self.val_ds: UPennGBMDataset | None = None
        self.test_ds: UPennGBMDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = UPennGBMDataset(self.config, partition="train", transform=SurvivalAugmentPipeline())
            self.val_ds = UPennGBMDataset(self.config, partition="val", transform=SurvivalInferencePipeline())

        if stage in (None, "test"):
            self.test_ds = UPennGBMDataset(self.config, partition="test", transform=SurvivalInferencePipeline())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
