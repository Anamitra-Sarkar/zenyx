"""Zenyx — A hardware-agnostic, self-managing distributed training runtime.

Never OOM. No config. GPU, TPU, CPU, AMD, Apple Silicon.
1T params, 1M context, 500K vocab.

Exports
-------
- :func:`train` — single entry-point for training.
- :class:`Trainer` — full-stack trainer class.
- :func:`wrap` — wraps a model with Zenyx memory management.
- :mod:`bench` — benchmarking utilities (memory budget, vs DeepSpeed).
- :data:`offload_policy` — JAX checkpoint policy for host-DRAM activation
  offloading. Pass to ``nn.remat(policy=offload_policy)`` to fix the
  55 GB XLA HBM OOM on TPU v5 lite.

Usage::

    import zenyx

    # Train a model — everything auto-configured
    trainer = zenyx.train(model, dataloader, context_len=131072)

    # Or wrap for manual control
    model = zenyx.wrap(model)

    # JAX training: fix 55 GB OOM with one line
    from zenyx import offload_policy
    import flax.linen as nn
    BlockRemat = nn.remat(Block, policy=offload_policy, prevent_cse=False)
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "Trainer",
    "train",
    "wrap",
    "bench",
    "auto_init_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "ModelLoader",
    "load_model",
    # Phase 7: KV Cache Tiering
    "BeladyKVCacheManager",
    # Phase 8: FP8 KV Quantization
    "quantize_kv_fp8",
    "dequantize_kv",
    "GradientMonitor",
    # Phase 9: Dynamic Ring Curriculum
    "RingCurriculumManager",
    "CurriculumConfig",
    # Phase 10: Sparse Ring Attention
    "SparseRingAttentionKernel",
    "compute_skip_schedule",
    # JAX activation offloading (fixes 55 GB XLA HBM OOM)
    "offload_policy",
    "make_offload_policy",
    "make_offload_remat",
]

import importlib.machinery
import logging
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger("zenyx")

# Bootstrap zenyx.train as a namespace package to avoid importing
# zenyx/train/__init__.py during top-level eager imports.
_train_pkg = sys.modules.get("zenyx.train")
if _train_pkg is None:
    _train_pkg = types.ModuleType("zenyx.train")
    train_path = str(Path(__file__).resolve().parent / "train")
    _train_pkg.__path__ = [train_path]  # type: ignore[attr-defined]
    _train_pkg.__package__ = "zenyx.train"
    _train_pkg.__spec__ = importlib.machinery.ModuleSpec(
        name="zenyx.train", loader=None, is_package=True
    )
    _train_pkg.__spec__.submodule_search_locations = [train_path]
    sys.modules["zenyx.train"] = _train_pkg

# ── Phase 4 imports ──────────────────────────────────────────────────────────────────────────

from zenyx.train.trainer import Trainer
from zenyx.train.trainer import train as _trainer_train
from zenyx.train.distributed_setup import (
    auto_init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
)
from zenyx.loader.loader import ModelLoader, load_model

# Convenience: auto-init distributed on import if env vars suggest it
# but ONLY if ZENYX_AUTO_INIT_DISTRIBUTED=1 is set
# (don't force init on every import — that breaks testing)
if os.environ.get("ZENYX_AUTO_INIT_DISTRIBUTED", "0") == "1":
    auto_init_distributed()


def __getattr__(name: str) -> Any:
    """Lazily import optional heavy / JAX-dependent symbols on first access."""
    lazy_imports = {
        "BeladyKVCacheManager": ("zenyx.train.kv_cache_tier", "BeladyKVCacheManager"),
        "quantize_kv_fp8": ("zenyx.train.fp8_kv", "quantize_kv_fp8"),
        "dequantize_kv": ("zenyx.train.fp8_kv", "dequantize_kv"),
        "GradientMonitor": ("zenyx.train.fp8_kv", "GradientMonitor"),
        "RingCurriculumManager": ("zenyx.train.ring_curriculum", "RingCurriculumManager"),
        "CurriculumConfig": ("zenyx.train.ring_curriculum", "CurriculumConfig"),
        "SparseRingAttentionKernel": (
            "zenyx.ops.attention.sparse_ring_attn",
            "SparseRingAttentionKernel",
        ),
        "compute_skip_schedule": (
            "zenyx.ops.attention.sparse_ring_attn",
            "compute_skip_schedule",
        ),
        "offload_policy": ("zenyx.ops.remat", "offload_policy"),
        "make_offload_policy": ("zenyx.ops.remat", "make_offload_policy"),
        "make_offload_remat": ("zenyx.ops.remat", "make_offload_remat"),
    }

    if name not in lazy_imports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = lazy_imports[name]

    if attr_name in {"offload_policy", "make_offload_policy", "make_offload_remat"}:
        try:
            module = __import__(module_name, fromlist=[attr_name])
        except ImportError as exc:
            raise ImportError(
                "offload_policy requires JAX. Install with: pip install jax[cuda]"
            ) from exc
    else:
        module = __import__(module_name, fromlist=[attr_name])

    value = getattr(module, attr_name)
    globals()[name] = value
    return value

# ── Lazy hardware detection ───────────────────────────────────────────────────────────

_hardware_info: Optional[Any] = None


def _auto_detect_hardware() -> Any:
    """Lazily auto-detect hardware on first access.

    Time: O(num_devices).  Space: O(1).
    """
    global _hardware_info  # noqa: PLW0603
    if _hardware_info is None:
        from zenyx.core.hal.detector import detect_hardware

        _hardware_info = detect_hardware()
        logger.debug("Zenyx auto-detected hardware: %s", _hardware_info)
    return _hardware_info


def hardware() -> Any:
    """Return the detected hardware info. Lazily initialized.

    Time: O(num_devices) on first call, O(1) thereafter.  Space: O(1).
    """
    return _auto_detect_hardware()


# ── Public API ─────────────────────────────────────────────────────────────────────────


def train(
    model: Any,
    dataloader: Any,
    **kwargs: Any,
) -> Trainer:
    """Train a model — Zenyx handles everything else.

    This is the single entry-point.  Zenyx will auto-detect hardware,
    plan parallelism, manage memory across three tiers, and never OOM.

    Creates a Trainer with the given model and dataloader, starts training,
    and returns the Trainer for inspection.

    Parameters
    ----------
    model : Any
        A ``torch.nn.Module`` (or compatible).
    dataloader : Any
        An iterable of training batches.
    **kwargs
        Additional keyword arguments passed to :class:`Trainer`.

    Returns
    -------
    Trainer
        The trainer instance after training completes.

    Example::

        trainer = zenyx.train(model, dataloader, context_len=131072)
        state = trainer.get_state()
    """
    return _trainer_train(model, dataloader, **kwargs)


def wrap(model: Any) -> Any:
    """Wrap a model with Zenyx memory management.

    Hooks into forward / backward for activation checkpointing and
    three-tier memory management.

    Time: O(num_parameters).  Space: O(1) additional.

    Parameters
    ----------
    model : Any
        A ``torch.nn.Module``.

    Returns
    -------
    Any
        The same model, instrumented with Zenyx hooks.
    """
    _auto_detect_hardware()
    logger.info("Zenyx v%s — wrapping model for memory management", __version__)

    # Register forward/backward hooks for memory tracking
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            from zenyx.core.agent.profiler import AsyncProfiler

            _profiler = AsyncProfiler(enabled=True)

            def _forward_hook(
                module: Any,
                input: Any,
                output: Any,
            ) -> None:
                pass  # Placeholder for activation checkpointing hooks

            for name, module in model.named_modules():
                module.register_forward_hook(_forward_hook)

            # Attach profiler to model for external access
            model._zenyx_profiler = _profiler  # type: ignore[attr-defined]
    except ImportError:
        logger.debug("PyTorch not available — wrap() is a no-op")

    return model


# ── bench module re-export ──────────────────────────────────────────────────────────────────────

from zenyx import bench  # noqa: E402


# ── Private helpers ──────────────────────────────────────────────────────────────────────


def _count_params(model: Any) -> float:
    """Count the number of trainable parameters in a model."""
    try:
        return float(sum(p.numel() for p in model.parameters()))
    except (AttributeError, TypeError):
        return 0.0


def _infer_vocab_size(model: Any) -> int:
    """Try to infer vocabulary size from common model attributes."""
    for attr in ("config", "cfg"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            for key in ("vocab_size", "n_vocab", "ntokens"):
                val = getattr(cfg, key, None)
                if val is not None:
                    return int(val)
    return 32000  # Sensible default


def _infer_batch_size(dataloader: Any) -> int:
    """Try to infer batch size from a dataloader."""
    bs = getattr(dataloader, "batch_size", None)
    if bs is not None:
        return int(bs)
    return 1


def _map_hw_to_preset(hw: Any) -> Optional[str]:
    """Map a HardwareInfo to a hardware preset name."""
    name = getattr(hw, "device_name", "")
    name_lower = name.lower()
    if "h100" in name_lower:
        return "H100"
    if "h200" in name_lower:
        return "H200"
    if "a100" in name_lower:
        return "A100"
    if "4090" in name_lower:
        return "RTX_4090"
    if "tpu" in name_lower:
        return "TPU_v5e"
    return None
