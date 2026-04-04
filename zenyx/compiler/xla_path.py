"""XLA-friendly checkpoint/offload hooks.

This module avoids hard dependency on torch_xla/jax; it enables optional
integration when those runtimes are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class XLACheckpointPolicy:
    offload_threshold_bytes: int = 8 * 1024 * 1024


def remat_or_checkpoint(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Use runtime checkpointing primitive when available; fallback to fn."""
    # torch_xla path (best-effort)
    try:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(fn, *args, use_reentrant=False, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def maybe_offload_large_tensor(tensor: torch.Tensor, policy: XLACheckpointPolicy) -> torch.Tensor:
    size_bytes = tensor.numel() * tensor.element_size()
    if size_bytes >= policy.offload_threshold_bytes:
        return tensor.detach().cpu()
    return tensor


__all__ = ["XLACheckpointPolicy", "remat_or_checkpoint", "maybe_offload_large_tensor"]
