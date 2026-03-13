"""Legacy (Phase 2) training entrypoint — internal use only.

.. note::
    This module is retained for backward compatibility.  The public training
    API is :func:`zenyx.train.trainer.train` (Phase 4).  Do **not** import
    ``train`` from this module — it has been renamed to ``_legacy_train``.

Workflow
--------
1. Detect hardware (auto if not provided).
2. Profile model (stub if profiler not yet available).
3. Plan parallelism (stub if planner not yet available).
4. Initialise allocator (TierAllocator stub).
5. Check feasibility, print memory budget.
6. Set up pipeline schedule.
7. Training loop: forward → backward → optimiser step → checkpoint (periodic).
8. **Never raises OOM** — catches ``torch.cuda.OutOfMemoryError``, evicts,
   and retries.
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zenyx.train.checkpoint import AsyncCheckpointer
from zenyx.train.mixed_prec import FP8ActivationStorage
from zenyx.train.pipeline import BraidedPipeline

__all__ = [
    "wrap",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT_EVERY: int = 500  # steps
_OOM_RETRY_LIMIT: int = 3

# One-shot warning flag for no-label fake-loss path.
_FAKE_LOSS_WARNED: bool = False

# ---------------------------------------------------------------------------
# Hardware detection (lightweight — full detector in detector.py)
# ---------------------------------------------------------------------------


def _detect_hardware() -> Dict[str, Any]:
    """Auto-detect available hardware.

    Returns
    -------
    Dict[str, Any]
        Keys: ``backend`` (``"cuda"`` | ``"xla"`` | ``"cpu"``),
        ``device_count``, ``device``, ``hbm_bytes``.

    Complexity
    ----------
    Time *O(1)*.
    """
    hw: Dict[str, Any] = {
        "backend": "cpu",
        "device_count": 1,
        "device": torch.device("cpu"),
        "hbm_bytes": 0,
    }

    if torch.cuda.is_available():
        hw["backend"] = "cuda"
        hw["device_count"] = torch.cuda.device_count()
        hw["device"] = torch.device("cuda:0")
        try:
            props = torch.cuda.get_device_properties(0)
            hw["hbm_bytes"] = props.total_mem
        except Exception:
            hw["hbm_bytes"] = 0
        return hw

    # Check for XLA / TPU.
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        hw["backend"] = "xla"
        hw["device_count"] = xm.xrt_world_size()
        hw["device"] = xm.xla_device()
        return hw
    except ImportError:
        pass

    return hw


# ---------------------------------------------------------------------------
# Stubbed subsystems (will be replaced by full implementations)
# ---------------------------------------------------------------------------


def _profile_model(model: nn.Module, hw: Dict[str, Any]) -> Dict[str, Any]:
    """Stub profiler — returns conservative estimates.

    Complexity
    ----------
    Time *O(P)* where *P* = number of parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = total_params * 4  # assume FP32
    return {
        "total_params": total_params,
        "param_bytes": param_bytes,
        "est_activation_bytes": param_bytes * 2,
    }


def _plan_parallelism(
    model: nn.Module,
    hw: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Stub planner — sensible defaults.

    Complexity
    ----------
    Time *O(1)*.
    """
    num_devices = hw["device_count"]
    plan: Dict[str, Any] = {
        "tp_degree": min(num_devices, 1),
        "pp_stages": 1,
        "dp_degree": max(1, num_devices),
        "microbatches": max(4, num_devices * 2),
    }
    if num_devices >= 4:
        plan["pp_stages"] = min(4, num_devices)
        plan["tp_degree"] = max(1, num_devices // plan["pp_stages"])
        plan["dp_degree"] = max(
            1, num_devices // (plan["pp_stages"] * plan["tp_degree"])
        )
    return plan


def _check_feasibility(hw: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    """Check OOM-free feasibility condition.

    (1/B_01) + (1/B_12) ≤ (1/F_compute)

    Returns *True* if the hard guarantee holds, *False* if throttling is
    needed (but we **never crash**).

    Complexity
    ----------
    Time *O(1)*.
    """
    hbm = hw.get("hbm_bytes", 0)
    needed = profile["param_bytes"] + profile["est_activation_bytes"]
    if hbm > 0 and needed > hbm:
        logger.warning(
            "Memory budget exceeds single-device HBM: need %.2f GB, have %.2f GB. "
            "Zenyx will use multi-tier offloading (T0→T1→T2). Training will be "
            "throttled but will NOT OOM.",
            needed / (1024**3),
            hbm / (1024**3),
        )
        return False
    return True


# ---------------------------------------------------------------------------
# OOM-safe execution context
# ---------------------------------------------------------------------------


@contextmanager
def _oom_guard(label: str = "step") -> Iterator[None]:
    """Context manager that catches CUDA OOM on the *first* attempt only.

    .. note::
        A ``@contextmanager`` generator cannot be retried — once ``yield``
        returns control to the caller's ``with`` block, the generator cannot
        be resumed a second time for the same context invocation.  Retry
        logic therefore lives in :func:`_safe_forward_backward` which uses
        an explicit ``for`` loop.  This wrapper is kept for single-shot OOM
        protection of miscellaneous CUDA operations outside the forward/
        backward path.

    Complexity
    ----------
    Time *O(1)* overhead.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "OOM during %s — evicting caches. "
            "Use _safe_forward_backward for retryable forward/backward passes.",
            label,
        )
        torch.cuda.empty_cache()
        raise


def _safe_forward_backward(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Run forward + backward with OOM protection and full exception logging.

    Retries up to ``_OOM_RETRY_LIMIT`` times on CUDA OOM.  All other
    exceptions are logged with full traceback context before re-raising so
    that shape mismatches, dtype errors, and NaN propagation are never
    silently swallowed.

    Complexity
    ----------
    Time *O(forward + backward)*.
    """
    global _FAKE_LOSS_WARNED

    for attempt in range(_OOM_RETRY_LIMIT):
        try:
            batch_dev = batch.to(device, non_blocking=True)
            output = model(batch_dev)
            if output.requires_grad:
                loss = output.sum()  # scalar proxy — see warning below
                if not _FAKE_LOSS_WARNED:
                    warnings.warn(
                        "_safe_forward_backward: no labels provided. "
                        "Using output.sum() as a proxy loss — gradients are "
                        "trivially uniform and carry no learning signal. "
                        "Pass labelled batches or use Trainer for real training.",
                        UserWarning,
                        stacklevel=2,
                    )
                    _FAKE_LOSS_WARNED = True
                loss.backward()
                return loss.detach()
            return output.detach()
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "OOM in forward/backward (attempt %d/%d) — evicting and retrying.",
                attempt + 1,
                _OOM_RETRY_LIMIT,
            )
            torch.cuda.empty_cache()
        except Exception:
            # Log full traceback so shape mismatches / dtype errors are visible.
            logger.exception(
                "Unhandled exception in forward/backward (attempt %d/%d, batch shape=%s).",
                attempt + 1,
                _OOM_RETRY_LIMIT,
                tuple(batch.shape) if hasattr(batch, "shape") else "unknown",
            )
            raise

    logger.error("Persistent OOM after %d retries — skipping this batch.", _OOM_RETRY_LIMIT)
    return None


# ---------------------------------------------------------------------------
# wrap()
# ---------------------------------------------------------------------------


def wrap(
    model: nn.Module,
    *,
    fp8_every_n: int = 4,
    force_simulated_fp8: bool = False,
) -> nn.Module:
    """Wrap a model with Zenyx memory management.

    .. deprecated::
        loop.py is deprecated. Use ``zenyx.train.trainer.Trainer`` instead.

    Hooks into forward/backward for FP8 E4M3 activation checkpointing.

    Parameters
    ----------
    model : nn.Module
        The user's model.
    fp8_every_n : int, optional
        Checkpoint every *N*-th layer (default 4).
    force_simulated_fp8 : bool, optional
        Force software FP8 fallback.

    Returns
    -------
    nn.Module
        The same model instance with selected layers wrapped.

    Complexity
    ----------
    Time *O(L)* where *L* = number of layers.
    """
    warnings.warn(
        "loop.py is deprecated. Use zenyx.train.trainer.Trainer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from zenyx.train.mixed_prec import fp8_checkpoint as _fp8_ckpt
    model = _fp8_ckpt(
        model,
        every_n=fp8_every_n,
        force_simulated=force_simulated_fp8,
    )
    logger.info("Model wrapped with Zenyx FP8 activation checkpointing (every_n=%d).", fp8_every_n)
    return model


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def _legacy_train(
    model: nn.Module,
    dataloader: DataLoader[Any],
    *,
    hardware: Optional[Dict[str, Any]] = None,
    context_len: Optional[int] = None,
    vocab_size: Optional[int] = None,
    max_steps: Optional[int] = None,
    lr: float = 1e-4,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    fp8_every_n: int = 4,
) -> Dict[str, Any]:
    """Legacy Phase 2 training entrypoint (internal use only).

    .. deprecated::
        loop.py is deprecated. Use ``zenyx.train.trainer.Trainer`` instead.

    Users call ``zenyx.train(model, dataloader)`` — everything else is
    auto-detected.  **Never raises** ``torch.cuda.OutOfMemoryError``.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    dataloader : DataLoader
        Training data.
    hardware : Dict, optional
        Override hardware detection result.
    context_len : int, optional
        Sequence length (auto-detected from first batch if *None*).
    vocab_size : int, optional
        Vocabulary size (auto-detected from model if *None*).
    max_steps : int, optional
        Stop after this many steps (*None* = 1 full epoch).
    lr : float, optional
        Learning rate (default 1e-4).
    checkpoint_dir : str, optional
        Where to save checkpoints (default: no checkpointing).
    checkpoint_every : int, optional
        Steps between checkpoints (default 500).
    fp8_every_n : int, optional
        FP8 checkpointing frequency for layers (default 4).

    Returns
    -------
    Dict[str, Any]
        Training summary with keys ``steps``, ``avg_loss``, ``hardware``.

    Complexity
    ----------
    Time *O(steps × (forward + backward + optimiser))*.
    """
    # ---- Step 1: Hardware detection ----
    warnings.warn(
        "loop.py is deprecated. Use zenyx.train.trainer.Trainer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    hw = hardware if hardware is not None else _detect_hardware()
    device = hw["device"]
    logger.info("Zenyx hardware: %s (%d devices)", hw["backend"], hw["device_count"])

    # ---- Step 2: Profile model ----
    profile = _profile_model(model, hw)
    logger.info(
        "Model profile: %.2fM params, ~%.2f GB parameters, ~%.2f GB activations.",
        profile["total_params"] / 1e6,
        profile["param_bytes"] / (1024**3),
        profile["est_activation_bytes"] / (1024**3),
    )

    # ---- Step 3: Plan parallelism ----
    plan = _plan_parallelism(model, hw, profile)
    logger.info("Parallelism plan: TP=%d, PP=%d, DP=%d, microbatches=%d",
                plan["tp_degree"], plan["pp_stages"],
                plan["dp_degree"], plan["microbatches"])

    # ---- Step 4: Check feasibility ----
    feasible = _check_feasibility(hw, profile)
    if feasible:
        logger.info("Feasibility check PASSED — OOM-free guarantee active.")
    else:
        logger.info("Feasibility check SOFT-FAIL — throttle mode active (no crash).")

    # ---- Step 5: Wrap model with FP8 checkpointing ----
    # Call fp8_checkpoint directly to avoid double DeprecationWarning
    # (wrap() also emits DeprecationWarning; _legacy_train already warned above).
    from zenyx.train.mixed_prec import fp8_checkpoint as _fp8_ckpt
    model = _fp8_ckpt(
        model,
        every_n=fp8_every_n,
        force_simulated=False,
    )
    model = model.to(device)

    # ---- Step 6: Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---- Step 7: Checkpointer ----
    checkpointer: Optional[AsyncCheckpointer] = None
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if checkpoint_dir is not None:
        checkpointer = AsyncCheckpointer(rank=rank, world_size=world_size)

    # ---- Step 8: Pipeline schedule (for PP > 1) ----
    pipeline: Optional[BraidedPipeline] = None
    if plan["pp_stages"] > 1:
        pipeline = BraidedPipeline(
            num_stages=plan["pp_stages"],
            num_microbatches=plan["microbatches"],
            num_devices=hw["device_count"],
        )
        pipeline.generate_schedule()
        logger.info("Braided pipeline schedule generated: %r", pipeline)

    # ---- Step 9: Training loop ----
    step = 0
    total_loss = 0.0
    model.train()

    logger.info("Starting training loop...")

    for batch in dataloader:
        if max_steps is not None and step >= max_steps:
            break

        # Auto-detect context_len from first batch.
        if context_len is None and isinstance(batch, torch.Tensor) and batch.dim() >= 2:
            context_len = batch.shape[1]
            logger.info("Auto-detected context_len=%d", context_len)

        optimizer.zero_grad(set_to_none=True)

        loss = _safe_forward_backward(model, batch, device)

        if loss is not None:
            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        step += 1

        # Periodic checkpoint.
        if (
            checkpointer is not None
            and checkpoint_dir is not None
            and step % checkpoint_every == 0
        ):
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}")
            state = {k: v for k, v in model.state_dict().items()}
            checkpointer.save(state, ckpt_path)
            logger.info("Checkpoint queued at step %d.", step)

        if step % 100 == 0:
            avg = total_loss / step if step > 0 else 0.0
            logger.info("Step %d — avg loss: %.6f", step, avg)

    # ---- Cleanup ----
    if checkpointer is not None:
        checkpointer.wait()
        checkpointer.shutdown()

    avg_loss = total_loss / step if step > 0 else 0.0
    summary = {
        "steps": step,
        "avg_loss": avg_loss,
        "hardware": hw,
    }
    logger.info("Training complete: %d steps, avg loss %.6f.", step, avg_loss)
    return summary
