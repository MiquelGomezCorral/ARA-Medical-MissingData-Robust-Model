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
import math

    
# ====================================================================
#                   SSL Augmentations
# ====================================================================
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
            # --- NORMALIZE ---
            RobustIntensityStandardization3D(nonzero=True),
            ZNormalization3D(nonzero=True),
            
            # --- AUGMENT ---
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
    

class RandomPatchSwap3D:
    """Swap a pair of random cubic patches inside the same volume."""

    def __init__(self, patch_size: int = 12, p: float = 0.5, swaps: int = 1):
        self.patch_size = patch_size
        self.p = p
        self.swaps = swaps

    def _sample_start(self, limit: int, patch_size: int) -> int:
        if limit <= patch_size:
            return 0
        return random.randint(0, limit - patch_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        x = x.clone()
        _, height, width, depth = x.shape
        patch_size = min(self.patch_size, height, width, depth)

        for _ in range(self.swaps):
            h1 = self._sample_start(height, patch_size)
            w1 = self._sample_start(width, patch_size)
            d1 = self._sample_start(depth, patch_size)
            h2 = self._sample_start(height, patch_size)
            w2 = self._sample_start(width, patch_size)
            d2 = self._sample_start(depth, patch_size)

            patch_1 = x[:, h1:h1 + patch_size, w1:w1 + patch_size, d1:d1 + patch_size].clone()
            patch_2 = x[:, h2:h2 + patch_size, w2:w2 + patch_size, d2:d2 + patch_size].clone()
            x[:, h1:h1 + patch_size, w1:w1 + patch_size, d1:d1 + patch_size] = patch_2
            x[:, h2:h2 + patch_size, w2:w2 + patch_size, d2:d2 + patch_size] = patch_1

        return x


class RandomCutout3D:
    """Mask a random cubic region of the input volume."""

    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 0.25, p: float = 0.5):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        x = x.clone()
        _, height, width, depth = x.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        cutout_size = max(1, int(min(height, width, depth) * ratio))

        h0 = 0 if height <= cutout_size else random.randint(0, height - cutout_size)
        w0 = 0 if width <= cutout_size else random.randint(0, width - cutout_size)
        d0 = 0 if depth <= cutout_size else random.randint(0, depth - cutout_size)

        x[:, h0:h0 + cutout_size, w0:w0 + cutout_size, d0:d0 + cutout_size] = 0
        return x


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


class RobustIntensityStandardization3D:
    """
    Acts as on-the-fly histogram standardization. Clips extreme outliers 
    (e.g., 1st and 99th percentiles) and scales intensities to [0, 1].
    """
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0, nonzero: bool = True):
        self.lower = lower_percentile / 100.0
        self.upper = upper_percentile / 100.0
        self.nonzero = nonzero

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for i in range(x.shape[0]):
            if self.nonzero:
                mask = x[i] > 0
                if not mask.any():
                    continue
                valid_voxels = x[i][mask]
            else:
                valid_voxels = x[i].flatten()

            p_low  = torch.quantile(valid_voxels, self.lower)
            p_high = torch.quantile(valid_voxels, self.upper)

            x[i] = torch.clamp(x[i], min=p_low.item(), max=p_high.item())

            if p_high > p_low:
                if self.nonzero:
                    x[i][mask] = (x[i][mask] - p_low) / (p_high - p_low)
                    x[i][~mask] = 0.0  # restore background to zero
                else:
                    x[i] = (x[i] - p_low) / (p_high - p_low)
        return x


class ZNormalization3D:
    """
    Applies Z-score normalization (zero mean, unit variance) per channel.
    Calculates statistics only on the brain tissue if nonzero=True.
    """
    def __init__(self, nonzero: bool = True, eps: float = 1e-6):
        self.nonzero = nonzero
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for i in range(x.shape[0]): # Iterate over channels
            if self.nonzero:
                mask = x[i] > 0
                if mask.any():
                    mean = x[i][mask].mean()
                    std = x[i][mask].std()
                    x[i][mask] = (x[i][mask] - mean) / (std + self.eps)
            else:
                mean = x[i].mean()
                std = x[i].std()
                x[i] = (x[i] - mean) / (std + self.eps)
        return x




# ====================================================================
#               BraTS-Specific Augmentations
# ====================================================================

class BraTSSSLAugmentPipeline:
    """Apply the BraTS-specific SSL corruptions used for pretraining views."""

    def __init__(self, patch_size: int = 12, cutout_min_ratio: float = 0.1, cutout_max_ratio: float = 0.25):
        self.transforms = [
            RandomPatchSwap3D(patch_size=patch_size, p=0.8, swaps=1),
            RandomCutout3D(min_ratio=cutout_min_ratio, max_ratio=cutout_max_ratio, p=0.8),
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


# ====================================================================
#                   Survival Augmentations
# ====================================================================


class RandomRotate3D:
    """
    Small random 3D rotations (±max_angle degrees) applied independently
    per axis. Uses trilinear interpolation — does NOT break spatial anatomy
    the way rot90 does. Safe for survival: tumor volume/shape preserved.
    """
    def __init__(self, max_angle: float = 10.0, p: float = 0.5):
        self.max_angle = max_angle
        self.p = p

    def _rot_matrix(self, rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=torch.float32)
        Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=torch.float32)
        Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=torch.float32)
        return Rz @ Ry @ Rx  # (3,3)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        max_rad = math.radians(self.max_angle)
        rx = random.uniform(-max_rad, max_rad)
        ry = random.uniform(-max_rad, max_rad)
        rz = random.uniform(-max_rad, max_rad)

        R = self._rot_matrix(rx, ry, rz)  # (3,3)

        # Build affine: [R | 0] as (1, 3, 4)
        affine = torch.zeros(1, 3, 4, dtype=torch.float32)
        affine[0, :3, :3] = R

        grid = F.affine_grid(affine, x.unsqueeze(0).shape, align_corners=False)
        x = F.grid_sample(
            x.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(0)
        return x


class RandomTranslate3D:
    """
    Small random translations (±max_shift fraction of volume size).
    Safe for survival: does not distort tumor morphology.
    """
    def __init__(self, max_shift: float = 0.05, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        tx = random.uniform(-self.max_shift, self.max_shift)
        ty = random.uniform(-self.max_shift, self.max_shift)
        tz = random.uniform(-self.max_shift, self.max_shift)

        affine = torch.zeros(1, 3, 4, dtype=torch.float32)
        affine[0, 0, 0] = 1.0
        affine[0, 1, 1] = 1.0
        affine[0, 2, 2] = 1.0
        affine[0, 0, 3] = tx
        affine[0, 1, 3] = ty
        affine[0, 2, 3] = tz

        grid = F.affine_grid(affine, x.unsqueeze(0).shape, align_corners=False)
        x = F.grid_sample(
            x.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(0)
        return x


class RandomGammaCorrection:
    """
    Applies random gamma correction per channel: out = in^gamma.
    Simulates scanner contrast variability. Only applied to nonzero voxels
    to avoid amplifying background noise.
    """
    def __init__(self, gamma_range: tuple = (0.7, 1.5), p: float = 0.5):
        self.gamma_min, self.gamma_max = gamma_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        x = x.clone()
        for i in range(x.shape[0]):
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            mask = x[i] > 0
            if mask.any():
                # Shift to [0,1] before gamma, then shift back
                vmin = x[i][mask].min()
                vmax = x[i][mask].max()
                if vmax > vmin:
                    x[i][mask] = ((x[i][mask] - vmin) / (vmax - vmin)) ** gamma
                    x[i][mask] = x[i][mask] * (vmax - vmin) + vmin
        return x


class RandomBiasField:
    """
    Simulates MRI bias field: a smooth low-frequency multiplicative field.
    Implemented as a random linear gradient per axis — cheap but effective
    at simulating inter-scanner intensity drift.
    """
    def __init__(self, max_strength: float = 0.3, p: float = 0.5):
        self.max_strength = max_strength
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        x = x.clone()
        C, H, W, D = x.shape

        # Build a smooth multiplicative field via linear gradients on each axis
        strength = random.uniform(0.0, self.max_strength)

        h_grad = torch.linspace(1 - strength, 1 + strength, H)
        w_grad = torch.linspace(1 - strength, 1 + strength, W)
        d_grad = torch.linspace(1 - strength, 1 + strength, D)

        # Random direction for each axis
        if random.random() < 0.5: h_grad = h_grad.flip(0)
        if random.random() < 0.5: w_grad = w_grad.flip(0)
        if random.random() < 0.5: d_grad = d_grad.flip(0)

        field = (
            h_grad.view(H, 1, 1) *
            w_grad.view(1, W, 1) *
            d_grad.view(1, 1, D)
        )  # (H, W, D)

        x = x * field.unsqueeze(0)  # broadcast over channels
        return x


class RandomGaussianBlur3D:
    """
    Applies mild Gaussian blur using depthwise 3D convolution.
    Simulates low-SNR acquisitions without altering morphology.
    """
    def __init__(self, max_sigma: float = 1.0, p: float = 0.3):
        self.max_sigma = max_sigma
        self.p = p

    def _gaussian_kernel(self, sigma: float, kernel_size: int = 5):
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g /= g.sum()
        # outer product for 3D
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        return kernel  # (k, k, k)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        sigma = random.uniform(0.3, self.max_sigma)
        kernel = self._gaussian_kernel(sigma)  # (k, k, k)
        k = kernel.shape[0]
        pad = k // 2

        C = x.shape[0]
        # Expand kernel to (C, 1, k, k, k) for depthwise conv
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, k, k, k)

        x_blurred = F.conv3d(
            x.unsqueeze(0),       # (1, C, H, W, D)
            kernel,
            padding=pad,
            groups=C
        ).squeeze(0)

        return x_blurred


# ====================================================================
#           REUSE: Already implemented (kept as-is)
# ====================================================================


class SurvivalSafeFlip3D:
    """
    Left-right flip only (dim=1 in C,H,W,D layout).
    Brain is largely symmetric along this axis — safe for survival.
    Top-bottom and anterior-posterior flips are excluded.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            x = x.flip(1)  # H axis only
        return x


# ====================================================================
#           Survival Pipelines: Train and Test
# ====================================================================

class SurvivalAugmentPipeline:
    """
    Full training pipeline for survival prediction.

    Order rationale:
      1. Normalize first so intensity augmentations operate in a
         consistent [0,1] / z-scored space.
      2. Spatial transforms (rotation, translation, flip) — geometry
         changes before any intensity corruption.
      3. Intensity augmentations (bias field, gamma, noise) — simulate
         scanner variability on already-normalized images.
      4. Mild blur last — applied after all other corruptions.

    What is NOT included vs SSL pipeline:
      - No RandomPatchSwap3D   (breaks local anatomy / label signal)
      - No RandomCutout3D      (destroys tumor regions)
      - No RandomCrop+resize   (changes apparent tumor size)
      - No RandomRotate90_3D   (90° rotations break anatomical orientation)
    """

    def __init__(
        self,
        # Spatial
        rotate_max_angle:   float = 10.0,
        rotate_p:           float = 0.5,
        translate_max:      float = 0.05,
        translate_p:        float = 0.5,
        flip_p:             float = 0.5,
        # Intensity
        bias_strength:      float = 0.3,
        bias_p:             float = 0.5,
        gamma_range:        tuple = (0.7, 1.5),
        gamma_p:            float = 0.5,
        # Noise / blur
        noise_std:          float = 0.05,
        noise_p:            float = 0.5,
        blur_sigma:         float = 1.0,
        blur_p:             float = 0.3,
    ):
        self.transforms = [
            # 1. Normalize
            RobustIntensityStandardization3D(nonzero=True),
            ZNormalization3D(nonzero=True),
            
            # 2. Spatial
            SurvivalSafeFlip3D(p=flip_p),
            RandomRotate3D(max_angle=rotate_max_angle, p=rotate_p),
            RandomTranslate3D(max_shift=translate_max, p=translate_p),
            # 3. Intensity
            RandomBiasField(max_strength=bias_strength, p=bias_p),
            RandomGammaCorrection(gamma_range=gamma_range, p=gamma_p),
            RandomIntensityScaling(lo=0.9, hi=1.1),
            # 4. Noise / blur
            RandomGaussianNoise(max_std=noise_std),
            RandomGaussianBlur3D(max_sigma=blur_sigma, p=blur_p),
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class SurvivalInferencePipeline:
    """
    Test/validation pipeline — normalization only, no augmentation.
    Must exactly mirror the normalization used in training.
    """

    def __init__(self):
        self.transforms = [
            RobustIntensityStandardization3D(nonzero=True),
            ZNormalization3D(nonzero=True),
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x