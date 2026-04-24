"""Training utilities for SSL and survival models."""

from .device import resolve_lightning_accelerator

from .train_ssl import train_stage_vit_pretraining
from .train_ssl import train_stage_ssl
from .train_survival import train_stage_survival
from .test import test_model
