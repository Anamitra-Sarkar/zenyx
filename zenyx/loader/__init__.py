"""Zenyx model loaders — GDS (GPU), TPU, CPU, and hardware-aware fast loader."""

from __future__ import annotations

from zenyx.loader.gds_loader import GDSModelLoader, estimate_load_time
from zenyx.loader.tpu_loader import TPUModelLoader
from zenyx.loader.loader import ModelLoader, load_model
from zenyx.loader.loader_config import LoaderConfig
from zenyx.loader.stats import LoaderStats

__all__ = [
    "GDSModelLoader",
    "LoaderConfig",
    "LoaderStats",
    "ModelLoader",
    "TPUModelLoader",
    "estimate_load_time",
    "load_model",
]
