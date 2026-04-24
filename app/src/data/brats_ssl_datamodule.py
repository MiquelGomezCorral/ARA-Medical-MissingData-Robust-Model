"""Lightning DataModule for BraTS SSL pretraining."""

import json
import os

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Configuration
from src.data.augmentations_3d import BraTSSSLAugmentPipeline, TwoViewTransform


class BraTSTensorDataset(Dataset):
    """Load BraTS tensors from disk and apply per-channel normalization."""

    def __init__(self, config: Configuration, exclude_ids: set[str] | None = None):
        self.config = config
        self.tensor_dir = config.brats_tensors_96
        self.exclude_ids = exclude_ids or set()
        self.patient_ids = self._collect_ids()

    def _collect_ids(self) -> list[str]:
        tensor_ids = [
            filename.removesuffix(".pt")
            for filename in os.listdir(self.tensor_dir)
            if filename.endswith(".pt")
        ]
        tensor_ids = sorted(tensor_ids)
        return [patient_id for patient_id in tensor_ids if patient_id not in self.exclude_ids]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        image = torch.load(os.path.join(self.tensor_dir, f"{pid}.pt"), weights_only=True)

        for channel_idx in range(image.shape[0]):
            channel = image[channel_idx]
            mean = channel.mean()
            std = channel.std().clamp(min=1e-5)
            image[channel_idx] = (channel - mean) / std

        return {
            "image": image,
            "patient_id": pid,
        }


class BraTSSSLDataset(Dataset):
    """Return two corrupted views and the uncorrupted target volume."""

    def __init__(self, base_dataset: BraTSTensorDataset, two_view_transform: TwoViewTransform):
        self.ds = base_dataset
        self.transform = two_view_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        xi, xj = self.transform(sample["image"])
        return xi, xj, sample["image"]


class BraTSSSLDataModule(L.LightningDataModule):
    """Expose a single BraTS train loader for SSL pretraining."""

    def __init__(
        self,
        config: Configuration,
        batch_size: int = 4,
        num_workers: int = 0,
        patch_size: int = 12,
        cutout_min_ratio: float = 0.1,
        cutout_max_ratio: float = 0.25,
        overlap_ids_path: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.cutout_min_ratio = cutout_min_ratio
        self.cutout_max_ratio = cutout_max_ratio
        self.overlap_ids_path = overlap_ids_path or config.brats_overlap_ids_path

    def _load_excluded_ids(self) -> set[str]:
        if not os.path.exists(self.overlap_ids_path):
            return set()

        with open(self.overlap_ids_path, "r", encoding="utf-8") as file:
            overlap_ids = json.load(file) or []

        return set(overlap_ids)

    def setup(self, stage: str | None = None) -> None:
        excluded_ids = self._load_excluded_ids()
        base_dataset = BraTSTensorDataset(self.config, exclude_ids=excluded_ids)
        augment = BraTSSSLAugmentPipeline(
            patch_size=self.patch_size,
            cutout_min_ratio=self.cutout_min_ratio,
            cutout_max_ratio=self.cutout_max_ratio,
        )
        self.train_ds = BraTSSSLDataset(base_dataset, TwoViewTransform(augment))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )