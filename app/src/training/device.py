"""Utilities for choosing the Lightning execution device."""

import torch

from src.config import Configuration


def resolve_lightning_accelerator(CONFIG: Configuration):
    """Return the Lightning accelerator, devices value, and a human-readable label."""
    if torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_index)
        return "gpu", 1, f"gpu:{current_index} ({device_name})"

    return "cpu", 1, "cpu"