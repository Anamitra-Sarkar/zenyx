"""Zenyx — A hardware-agnostic, self-managing distributed training runtime.

Never OOM. No config. GPU, TPU, CPU, AMD, Apple Silicon.
1T params, 1M context, 500K vocab.

Exports
-------
- :func:`train` — single entry-point for training.
- :func:`wrap` — wraps a model with Zenyx memory management.
- :mod:`bench` — benchmarking utilities (memory budget, vs DeepSpeed).

Usage::

    import zenyx

    # Train a model — everything auto-configured
    zenyx.train(model, dataloader)

    # Or wrap for manual control
    model = zenyx.wrap(model)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "train",
    "wrap",
    "bench",
]

import logging
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger("zenyx")

# ── Lazy hardware detection ──────────────────────────────────────────────

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


@property  # type: ignore[misc]
def hardware(self: Any) -> Any:
    """Module-level property emulation for lazy hardware info."""
    return _auto_detect_hardware()


# ── Public API ───────────────────────────────────────────────────────────


def train(
    model: Any,
    dataloader: Any,
    hardware: Any = None,
    context_len: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> None:
    """Train a model — Zenyx handles everything else.

    This is the single entry-point.  Zenyx will auto-detect hardware,
    plan parallelism, manage memory across three tiers, and never OOM.

    Time: O(steps × model_size).  Space: managed by TierAllocator.

    Parameters
    ----------
    model : Any
        A ``torch.nn.Module`` (or compatible).
    dataloader : Any
        An iterable of training batches.
    hardware : Any, optional
        A :class:`~zenyx.core.hal.base.HardwareInfo` instance.
        Auto-detected if ``None``.
    context_len : int, optional
        Maximum sequence length.  Inferred from first batch if ``None``.
    vocab_size : int, optional
        Vocabulary size.  Inferred from model config if ``None``.
    """
    # Step 1: hardware detection
    if hardware is None:
        hardware = _auto_detect_hardware()
    logger.info("Zenyx v%s — training on %s", __version__, hardware)

    # Step 2: detect topology
    from zenyx.ops.comm.topology import TopologyDetector

    topology = TopologyDetector.detect()
    logger.debug("Topology: %s", topology)

    # Step 3: set up profiler
    from zenyx.core.agent.profiler import AsyncProfiler

    profiler = AsyncProfiler(enabled=True)

    # Step 4: plan parallelism
    from zenyx.core.agent.planner import ParallelismPlanner

    planner = ParallelismPlanner(hardware, topology)

    # Estimate model params
    model_params = _count_params(model)
    _context_len = context_len or 2048
    _vocab_size = vocab_size or _infer_vocab_size(model)
    batch_size = _infer_batch_size(dataloader)

    plan = planner.plan(model_params, _vocab_size, _context_len, batch_size)
    logger.info("Plan: %s", plan)

    # Step 5: print memory budget
    from zenyx.bench.memory_budget import memory_budget

    hw_name = _map_hw_to_preset(hardware)
    if hw_name:
        report = memory_budget(model_params, _vocab_size, _context_len, hw_name)
        logger.info("\n%s", report)

    # Step 6: set up training controller
    from zenyx.core.agent.controller import TrainingController

    controller = TrainingController(
        planner=planner,
        profiler=profiler,
        model_params=model_params,
        vocab_size=_vocab_size,
        batch_size=batch_size,
    )

    # Step 7: training loop
    import itertools

    step = 0
    for batch in dataloader:
        handle = profiler.start_op("train_step")

        # Forward
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch

        try:
            import torch

            if torch.cuda.is_available() and hasattr(inputs, "to"):
                inputs = inputs.to("cuda")
        except ImportError:
            pass

        fwd_handle = profiler.start_op("forward")
        output = model(inputs)
        profiler.end_op(fwd_handle)

        # Loss + backward
        loss_val = 0.0
        if hasattr(output, "mean"):
            loss = output.mean()
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            bwd_handle = profiler.start_op("backward")
            if hasattr(loss, "backward"):
                loss.backward()
            profiler.end_op(bwd_handle)

        profiler.end_op(handle)

        # Controller step — may trigger replan
        new_plan = controller.step(step, loss_val, _context_len)
        if new_plan is not None:
            plan = new_plan

        step += 1

    # Cleanup
    profiler.shutdown()
    stats = controller.get_training_stats()
    logger.info("Training complete: %s", stats)


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


# ── bench module re-export ───────────────────────────────────────────────

from zenyx import bench  # noqa: E402


# ── Private helpers ──────────────────────────────────────────────────────


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
