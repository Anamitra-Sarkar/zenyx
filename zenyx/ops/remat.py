"""Zenyx activation offloading via JAX checkpoint policy.

This module provides the core fix for the 55 GB XLA HBM OOM that occurs
when training deep transformers on TPU v5 lite chips (16 GiB HBM each).

Root cause
----------
When ``jax.jit`` compiles a training step, XLA's ``BufferAssignment`` pass
statically pins ALL intermediate activations in HBM from the moment they
are produced in the forward pass until they are consumed in the backward
pass. For a 20-layer model with SwiGLU FFN (D_FF=4096) and MICRO_BS=8,
SEQ_LEN=8192, each layer produces:

    bf16[8, 8192, 4096] = 8 * 8192 * 4096 * 2 bytes = 512 MB

20 layers × 2 Dense ops each = 40 × 512 MB = **20 GB** just in SwiGLU
intermediates, all simultaneously live in HBM. Total compiled program
size reaches 55 GB on a 15.75 GiB chip → deterministic OOM before the
first MXU instruction executes.

Fix: JAX Offloadable API
------------------------
JAX exposes a ``CheckpointPolicy`` hook that is called during abstract
evaluation (tracing) — before any HLO is compiled or any tensor is
materialized. For each primitive, the policy returns one of:

- ``Saveable``:   keep the activation in HBM (XLA default)
- ``Recompute``:  discard and recompute during backward (standard remat)
- ``Offloadable(src="device", dst="pinned_host")``:  DMA the buffer to
  pinned host DRAM immediately after the forward kernel completes, and
  inject a prefetch DMA back to device just before the backward kernel
  that consumes it.

The ``Offloadable`` path is strictly superior to ``Recompute`` for large
activations on TPU because:
  1. Host DRAM (512 GiB) is orders of magnitude larger than HBM (16 GiB).
  2. The TPU ICI fabric and PCIe DMA engine overlap the host transfer with
     the next layer's forward computation, hiding latency entirely for
     tensors > ~1 MB (DMA setup cost ~10 µs, saturates PCIe Gen4 at
     ~28 GB/s for contiguous BF16 blocks).
  3. No extra FLOPs are spent recomputing SwiGLU matmuls.

Usage in model code
-------------------
Replace::

    BlockRemat = nn.remat(Block, prevent_cse=False)

With::

    from zenyx.ops.remat import offload_policy
    BlockRemat = nn.remat(Block, policy=offload_policy, prevent_cse=False)

Or use the decorator form on any Flax module or plain function::

    from zenyx.ops.remat import make_offload_remat
    apply_block = make_offload_remat(threshold_mb=5.0)(my_block_fn)

Threshold selection
-------------------
The default threshold is 5 MB. This is calibrated to:

- **Offload**: SwiGLU intermediate bf16[B, T, D_FF] which is 512 MB at
  (B=8, T=8192, D_FF=4096) — massively above threshold.
- **Keep in HBM**: small tensors like RMSNorm scale parameters (6 KB),
  RoPE sin/cos tables (128 KB), attention bias scalars — for which the
  PCIe DMA setup latency (10 µs) exceeds the tensor transfer time.

References
----------
- JAX autodiff remat docs:
  https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html
- JAX host offloading:
  https://docs.jax.dev/en/latest/notebooks/host-offloading.html
- JAX Offloadable discussion #19063:
  https://github.com/google/jax/discussions/19063
- JAX issue #23869 (remat + scan + offload):
  https://github.com/jax-ml/jax/issues/23869
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("zenyx.ops.remat")

# ---------------------------------------------------------------------------
# Threshold constant
# ---------------------------------------------------------------------------

#: Default byte threshold above which activations are offloaded to host DRAM.
#: 5 MB is chosen so that:
#:   - SwiGLU/FFN intermediates (>=64 MB) are always offloaded.
#:   - Small tensors (norms, biases, RoPE tables) stay in HBM.
_DEFAULT_THRESHOLD_BYTES: int = 5 * 1024 * 1024  # 5 MB

# ---------------------------------------------------------------------------
# Lazy imports for JAX checkpoint types
# ---------------------------------------------------------------------------

def _get_jax():
    """Lazy import of jax — returns None when JAX is not installed."""
    try:
        import jax  # type: ignore[import-untyped]
        return jax
    except ImportError:
        return None


def _get_offloadable():
    """Lazy import of Offloadable — avoids hard dependency on internal JAX paths."""
    try:
        from jax.ad_checkpoint import Offloadable
        return Offloadable
    except ImportError:
        pass
    try:
        from jax._src.interpreters.ad import Offloadable  # type: ignore[no-redef]
        return Offloadable
    except ImportError:
        return None


def _get_saveable():
    """Lazy import of Saveable."""
    try:
        from jax.ad_checkpoint import Saveable
        return Saveable
    except ImportError:
        pass
    try:
        from jax._src.interpreters.ad import Saveable  # type: ignore[no-redef]
        return Saveable
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Core policy factory
# ---------------------------------------------------------------------------

def make_offload_policy(threshold_mb: float = 5.0) -> Callable:
    """Create a JAX checkpoint policy that offloads large activations to host DRAM.

    The returned callable is a valid ``CheckpointPolicy`` for use with
    ``jax.checkpoint(policy=...)`` or ``flax.linen.remat(policy=...)``.

    When JAX is not installed this returns a no-op callable so that the module
    can be imported on CPU-only environments without raising ``ImportError``.

    Args:
        threshold_mb: Activation size threshold in megabytes. Tensors larger
            than this are offloaded to pinned host DRAM via async DMA.
            Tensors smaller than this are kept in HBM (``Saveable``).
            Default: 5.0 MB.

    Returns:
        A ``CheckpointPolicy`` callable ``(prim, *avals, **params) -> directive``.

    Example::

        policy = make_offload_policy(threshold_mb=5.0)
        BlockRemat = nn.remat(Block, policy=policy, prevent_cse=False)

    Time complexity:  O(1) per primitive call during tracing.
    Space complexity: O(1).
    """
    jax = _get_jax()
    if jax is None:
        logger.debug(
            "JAX not installed — make_offload_policy returning no-op policy. "
            "Install jax to enable host offloading."
        )
        def _noop_policy(*args: Any, **kwargs: Any) -> None:
            return None
        return _noop_policy

    threshold_bytes = int(threshold_mb * 1024 * 1024)
    Offloadable = _get_offloadable()
    Saveable = _get_saveable()

    if Offloadable is None or Saveable is None:
        logger.warning(
            "JAX Offloadable/Saveable not available in this JAX version. "
            "Falling back to plain jax.checkpoint (Recompute). "
            "Upgrade to JAX >= 0.4.25 for host offloading support."
        )
        # Return a policy that always recomputes — still correct, just slower.
        def _recompute_policy(prim: Any, *avals: Any, **params: Any) -> Any:
            return jax.checkpoint_policies.nothing_saveable
        return _recompute_policy

    def _offload_policy(prim: Any, *avals: Any, **params: Any) -> Any:
        """Evaluate primitive output size and decide HBM vs host-DRAM placement.

        Called by JAX during abstract evaluation (tracing phase), before any
        HLO is compiled. ``avals`` are ``ShapedArray`` objects — they carry
        shape and dtype but no concrete data.

        Returns:
            ``Offloadable(src='device', dst='pinned_host')`` if total output
            bytes exceed threshold, else ``Saveable``.
        """
        try:
            # abstract_eval returns ShapedArray or sequence of ShapedArray.
            out_avals = prim.abstract_eval(*avals, **params)
            if not isinstance(out_avals, (list, tuple)):
                out_avals = [out_avals]

            total_bytes: int = 0
            for aval in out_avals:
                if hasattr(aval, "size") and hasattr(aval, "dtype"):
                    total_bytes += int(aval.size) * int(aval.dtype.itemsize)

            if total_bytes > threshold_bytes:
                logger.debug(
                    "Offloading primitive %s output (%.1f MB > %.1f MB threshold) "
                    "to pinned host DRAM.",
                    prim.name if hasattr(prim, "name") else str(prim),
                    total_bytes / (1024 * 1024),
                    threshold_mb,
                )
                return Offloadable(src="device", dst="pinned_host")

            return Saveable

        except Exception as exc:
            # abstract_eval can fail for custom-call/opaque primitives.
            # Default to Saveable (keep in HBM) — conservative and correct.
            logger.debug(
                "abstract_eval failed for primitive %s (%s). Defaulting to Saveable.",
                prim.name if hasattr(prim, "name") else str(prim),
                exc,
            )
            return Saveable

    return _offload_policy


# ---------------------------------------------------------------------------
# Module-level default policy instance
# ---------------------------------------------------------------------------

#: Ready-to-use default policy with 5 MB threshold.
#: Pass directly to ``nn.remat(policy=offload_policy)`` or
#: ``jax.checkpoint(policy=offload_policy)``.
offload_policy: Callable = make_offload_policy(threshold_mb=5.0)


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------

def make_offload_remat(threshold_mb: float = 5.0) -> Callable:
    """Return a ``jax.checkpoint`` decorator with the Zenyx offload policy.

    Equivalent to::

        jax.checkpoint(policy=make_offload_policy(threshold_mb), prevent_cse=False)

    but callable as a decorator directly::

        @make_offload_remat(threshold_mb=5.0)
        def my_layer(x):
            ...

    When JAX is not installed this returns an identity decorator so that the
    module can be imported in CPU-only environments without raising ``ImportError``.

    Args:
        threshold_mb: See :func:`make_offload_policy`.

    Returns:
        A decorator that wraps a function with ``jax.checkpoint`` using the
        size-based offload policy.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    jax = _get_jax()
    if jax is None:
        logger.debug("JAX not installed — make_offload_remat returning identity decorator.")
        def _identity_decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)
            return wrapped
        return _identity_decorator

    policy = make_offload_policy(threshold_mb)

    def _jax_checkpoint_decorator(fn: Callable) -> Callable:
        checkpointed_fn = jax.checkpoint(fn, policy=policy, prevent_cse=False)

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return checkpointed_fn(*args, **kwargs)
        return wrapped

    return _jax_checkpoint_decorator
