"""
3-D MRI Augmentations for SSL pretraining.

`TwoViewTransform` wraps any augmentation pipeline and applies it twice
independently to the same volume, returning (view_i, view_j) — the
positive pair expected by SSLPretraining.

All transforms work on torch.Tensor with shape (C, H, W, D).
"""

import random
import torch
import torch.nn.functional as F


class RandomFlip3D:
    """Randomly flip the volume along one or more spatial axes."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for dim in (1, 2, 3):
            if random.random() < self.p:
                x = x.flip(dim)
        return x


class RandomRotate90_3D:
    """Randomly rotate 0 / 90 / 180 / 270 degrees in one of the three planes."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            plane = random.choice([(1, 2), (1, 3), (2, 3)])
            k = random.randint(1, 3)
            x = torch.rot90(x, k=k, dims=plane)
        return x


class RandomIntensityScaling:
    """Multiply all voxel intensities by a random scalar in [lo, hi]."""

    def __init__(self, lo: float = 0.9, hi: float = 1.1):
        self.lo = lo
        self.hi = hi

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(self.lo, self.hi)
        return x * scale


class RandomGaussianNoise:
    """Add zero-mean Gaussian noise with random std in [0, max_std]."""

    def __init__(self, max_std: float = 0.05):
        self.max_std = max_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = random.uniform(0.0, self.max_std)
        return x + torch.randn_like(x) * std


class RandomCrop3D:
    """
    Randomly crop a `crop_size`^3 sub-volume and resize back to `target_size`^3.
    Simulates zoom/scale augmentation.
    """

    def __init__(self, target_size: int = 96, min_scale: float = 0.8):
        self.target = target_size
        self.min_scale = min_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W, D = x.shape
        scale = random.uniform(self.min_scale, 1.0)
        ch = int(H * scale)
        cw = int(W * scale)
        cd = int(D * scale)

        dh = random.randint(0, H - ch)
        dw = random.randint(0, W - cw)
        dd = random.randint(0, D - cd)

        x = x[:, dh:dh + ch, dw:dw + cw, dd:dd + cd]
        x = F.interpolate(
            x.unsqueeze(0), size=(self.target, self.target, self.target),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
        return x


class MRIAugmentPipeline:
    """Applies a sequence of augmentations to a 3-D MRI tensor."""

    def __init__(
        self,
        vol_size:      int   = 96,
        flip_p:        float = 0.5,
        rotate_p:      float = 0.5,
        noise_std:     float = 0.05,
        intensity_lo:  float = 0.9,
        intensity_hi:  float = 1.1,
        crop_scale:    float = 0.85,
    ):
        self.transforms = [
            RandomFlip3D(p=flip_p),
            RandomRotate90_3D(p=rotate_p),
            RandomIntensityScaling(lo=intensity_lo, hi=intensity_hi),
            RandomGaussianNoise(max_std=noise_std),
            RandomCrop3D(target_size=vol_size, min_scale=crop_scale),
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class TwoViewTransform:
    """Apply the same augmentation pipeline twice to create a positive pair."""

    def __init__(self, base_transform: MRIAugmentPipeline):
        self.aug = base_transform

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.aug(x), self.aug(x)