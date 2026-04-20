"""Utilities for choosing the Lightning execution device."""

import torch

from src.config import Configuration


def resolve_lightning_accelerator(config: Configuration):
    """Return the Lightning accelerator, devices value, and a human-readable label."""
    if torch.cuda.is_available():
        gpu_index = getattr(config, "gpu_index", None)
        visible_gpu_count = torch.cuda.device_count()

        if gpu_index is not None:
            gpu_index = int(gpu_index)
            if gpu_index < 0 or gpu_index >= visible_gpu_count:
                raise ValueError(
                    f"gpu_index={gpu_index} is out of range for {visible_gpu_count} visible CUDA device(s)."
                )

            device_name = torch.cuda.get_device_name(gpu_index)
            return "gpu", [gpu_index], f"gpu:{gpu_index} ({device_name})"

        current_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_index)
        return "gpu", 1, f"gpu:{current_index} ({device_name})"

    return "cpu", 1, "cpu"