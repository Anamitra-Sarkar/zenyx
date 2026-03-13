"""Distributed setup for Zenyx.

Detects the execution environment (single process, torchrun, SLURM,
Google Cloud TPU) and initializes torch.distributed appropriately.
The user never calls init_process_group() — Zenyx does it.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

__all__ = [
    "auto_init_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "cleanup",
]

logger = logging.getLogger(__name__)


def auto_init_distributed() -> bool:
    """Auto-detect and initialize torch.distributed.

    Detection order:
    1. If TORCHELASTIC_RESTART_COUNT env var set → torchrun → init with env://
    2. If SLURM_PROCID env var set → SLURM → init with MASTER_ADDR/PORT
    3. If TPU_WORKER_ID env var set → Google Cloud TPU → init with XLA backend
    4. If RANK and WORLD_SIZE env vars set → manual distributed → init with env://
    5. Otherwise → single process → skip init, return False

    Returns True if distributed was initialized, False if single-process.

    Sets:
    - os.environ["MASTER_ADDR"] if not set (default "localhost")
    - os.environ["MASTER_PORT"] if not set (default "29500")
    """
    # Already initialized — nothing to do
    if dist.is_available() and dist.is_initialized():
        logger.info("torch.distributed already initialized (rank=%d)", dist.get_rank())
        return True

    # 1. Torchrun / torchelastic
    if "TORCHELASTIC_RESTART_COUNT" in os.environ:
        _ensure_master_env()
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        logger.info(
            "Distributed init via torchrun: rank=%d, world_size=%d, backend=%s",
            dist.get_rank(),
            dist.get_world_size(),
            backend,
        )
        return True

    # 2. SLURM
    if "SLURM_PROCID" in os.environ:
        os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
        os.environ.setdefault(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", "1"),
        )
        os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
        _ensure_master_env()
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        logger.info(
            "Distributed init via SLURM: rank=%d, world_size=%d, backend=%s",
            dist.get_rank(),
            dist.get_world_size(),
            backend,
        )
        return True

    # 3. Google Cloud TPU
    if "TPU_WORKER_ID" in os.environ:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]
            import torch_xla.distributed.xla_backend  # type: ignore[import-untyped]  # noqa: F401

            dist.init_process_group(backend="xla", init_method="xla://")
            logger.info(
                "Distributed init via XLA/TPU: rank=%d, world_size=%d",
                xm.get_ordinal(),
                xm.xrt_world_size(),
            )
            return True
        except ImportError:
            logger.warning("TPU detected but torch_xla not available")
            return False

    # 4. Manual distributed (RANK + WORLD_SIZE set by user)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _ensure_master_env()
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        logger.info(
            "Distributed init via env vars: rank=%d, world_size=%d, backend=%s",
            dist.get_rank(),
            dist.get_world_size(),
            backend,
        )
        return True

    # 5. Single process
    logger.info("Single-process mode (no distributed init)")
    return False


def _ensure_master_env() -> None:
    """Set MASTER_ADDR and MASTER_PORT defaults if not already set."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")


def get_rank() -> int:
    """Return current rank (0 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return world size (1 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True only on rank 0."""
    return get_rank() == 0


def barrier() -> None:
    """Global barrier. No-op if not distributed."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    """Destroy process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")
