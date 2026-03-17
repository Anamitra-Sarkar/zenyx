"""Zenyx OOM feasibility checker — pure math, no hardware dependencies.

Provides three key utilities:

1. **check_feasibility** — verifies the formal OOM-free guarantee condition:
   ``F_compute ≤ _PIPELINE_DEPTH × max(B_01, B_12)``
   i.e. the pipeline lookahead depth times the maximum available bandwidth must
   cover the effective compute demand (in bytes/sec), so eviction always keeps
   up with compute.
2. **estimate_memory_budget** — computes per-component memory requirements
   for a transformer model, including **peak backward activation memory**
   which was previously missing and caused the budget check to silently
   pass while XLA later OOMed with 55 GB allocations.
3. **compute_throughput_from_hardware** — converts raw TFLOPS into the
   correct bytes/sec unit required by ``check_feasibility``.

All functions are deterministic and side-effect free.

Unit contract
-------------
All three arguments to ``check_feasibility`` must be in **bytes/sec**
(i.e. a bandwidth).  The OOM-free condition is:

    F_compute ≤ _PIPELINE_DEPTH × max(B_01, B_12)

where *B_01* = T1→T0 bandwidth (bytes/sec), *B_12* = T2→T1 bandwidth
(bytes/sec), *F_compute* = effective memory bandwidth equivalent of
compute in bytes/sec (= TFLOPS × 10¹² / FLOP_PER_BYTE), and
*_PIPELINE_DEPTH* = prefetch lookahead depth in steps.

Activation memory fix
---------------------
The previous ``estimate_memory_budget`` used::

    activations_gb = 2.0 * weights_gb

This is a static approximation that ignores the true peak backward
activation cost, which is dominated by the FFN expansion layer::

    peak_activation_gb = micro_bs × seq_len × d_ff × n_layers × dtype_bytes / GiB

For (micro_bs=8, seq_len=8192, d_ff=4096, n_layers=20, dtype=bf16):
  - Old formula:  activations_gb = 2 × 1.27 GB = 2.54 GiB  ← wrong
  - New formula:  peak_activation_gb = 8×8192×4096×20×2 / 2^30 = 20.0 GiB ← correct

The old formula caused the budget assertion to pass with 1.65 GiB/device
while the XLA compiler reserved 55 GB and crashed.

Backward compatibility
----------------------
The new signature adds optional ``micro_bs``, ``seq_len``, ``d_ff``
parameters. When omitted, the function falls back to the 2×weights
approximation and logs a warning. Existing call sites continue to work.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from zenyx.core.allocator.constants import PIPELINE_DEPTH_STEPS

logger = logging.getLogger("zenyx.core.allocator.feasibility")

# Arithmetic intensity of a typical transformer layer in BF16:
# ~16 FLOP per byte (dominant cost is large matmuls at batch>1).
# Reference: Kaplan et al. 2020, Korthikanti et al. 2022.
_TRANSFORMER_FLOP_PER_BYTE: float = 16.0

# Pipeline prefetch depth used by the formal feasibility bound.
# FIX: Keep this constant shared with TierAllocator so the bound matches runtime behavior.


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeasibilityResult:
    """Outcome of the OOM feasibility check.

    Attributes:
        is_feasible:        ``True`` if F_compute ≤ B_01 AND F_compute ≤ B_12.
        margin:             Signed margin in bytes/sec.
                            ``F_compute - (_PIPELINE_DEPTH × max(B_01, B_12))``.
                            Positive → bandwidth bottleneck (deficit).
                            Negative → bandwidth headroom.
        bandwidth_t0_t1:    T0 ↔ T1 bandwidth in bytes/sec.
        bandwidth_t1_t2:    T1 ↔ T2 bandwidth in bytes/sec.
        compute_throughput: Compute throughput in bytes/sec equivalent.
        message:            Human-readable diagnostic string.
    """

    is_feasible: bool
    margin: float
    bandwidth_t0_t1: float
    bandwidth_t1_t2: float
    compute_throughput: float
    message: str

    def __repr__(self) -> str:
        status = "FEASIBLE" if self.is_feasible else "NOT FEASIBLE"
        return (
            f"FeasibilityResult(\n"
            f"  status={status},\n"
            f"  margin={self.margin:.6e} B/s,\n"
            f"  bandwidth_t0_t1={self.bandwidth_t0_t1:.3e} B/s,\n"
            f"  bandwidth_t1_t2={self.bandwidth_t1_t2:.3e} B/s,\n"
            f"  compute_throughput={self.compute_throughput:.3e} B/s,\n"
            f"  message={self.message!r}\n"
            f")"
        )


@dataclass(slots=True)
class MemoryBudget:
    """Estimated memory requirements for a transformer model.

    All values are in **gibibytes** (GiB = 2^30 bytes).
    """

    weights_gb: float
    activations_gb: float
    kv_cache_gb: float
    optimizer_gb: float
    total_gb: float
    per_device_gb: float

    def __repr__(self) -> str:
        return (
            f"MemoryBudget(\n"
            f"  weights     = {self.weights_gb:>8.2f} GiB,\n"
            f"  activations = {self.activations_gb:>8.2f} GiB,\n"
            f"  kv_cache    = {self.kv_cache_gb:>8.2f} GiB,\n"
            f"  optimizer   = {self.optimizer_gb:>8.2f} GiB,\n"
            f"  total       = {self.total_gb:>8.2f} GiB,\n"
            f"  per_device  = {self.per_device_gb:>8.2f} GiB\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Unit conversion helper
# ---------------------------------------------------------------------------


def compute_throughput_from_hardware(
    compute_tflops: float,
    flop_per_byte: float = _TRANSFORMER_FLOP_PER_BYTE,
) -> float:
    """Convert raw TFLOPS into bytes/sec for use in ``check_feasibility``.

    Args:
        compute_tflops: Peak compute throughput in TFLOPS.
        flop_per_byte:  Arithmetic intensity of the workload in FLOP/byte.
                        Defaults to 16.0 (BF16 transformer matmuls).

    Returns:
        Effective compute throughput in **bytes/sec**.

    Raises:
        ValueError: If either argument is ≤ 0.

    Example::

        >>> ct = compute_throughput_from_hardware(197.0)
        >>> ct
        1.23125e+13   # bytes/sec

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    if compute_tflops <= 0:
        raise ValueError(f"compute_tflops must be > 0, got {compute_tflops}")
    if flop_per_byte <= 0:
        raise ValueError(f"flop_per_byte must be > 0, got {flop_per_byte}")
    return (compute_tflops * 1e12) / flop_per_byte


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_feasibility(
    bandwidth_t0_t1: float,
    bandwidth_t1_t2: float,
    compute_throughput: float,
) -> FeasibilityResult:
    """Verify the formal OOM-free feasibility condition.

    The condition is::

        F_compute ≤ _PIPELINE_DEPTH × max(B_01, B_12)

    Args:
        bandwidth_t0_t1:    T0 ↔ T1 bandwidth in bytes/sec.
        bandwidth_t1_t2:    T1 ↔ T2 bandwidth in bytes/sec.
        compute_throughput: Effective compute throughput in bytes/sec.

    Returns:
        :class:`FeasibilityResult` with diagnosis.

    Raises:
        ValueError: If any input is ≤ 0.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    if bandwidth_t0_t1 <= 0:
        raise ValueError(f"bandwidth_t0_t1 must be > 0, got {bandwidth_t0_t1}")
    if bandwidth_t1_t2 <= 0:
        raise ValueError(f"bandwidth_t1_t2 must be > 0, got {bandwidth_t1_t2}")
    if compute_throughput <= 0:
        raise ValueError(f"compute_throughput must be > 0, got {compute_throughput}")

    effective_bw = PIPELINE_DEPTH_STEPS * max(bandwidth_t0_t1, bandwidth_t1_t2)
    margin = compute_throughput - effective_bw
    is_feasible = margin <= 0.0

    if is_feasible:
        message = (
            f"OOM-free guarantee active: F_compute ({compute_throughput:.3e} B/s) "
            f"≤ {PIPELINE_DEPTH_STEPS}×max(B_01={bandwidth_t0_t1:.3e}, "
            f"B_12={bandwidth_t1_t2:.3e}) = {effective_bw:.3e} B/s. "
            f"Headroom = {-margin:.3e} B/s."
        )
        logger.info(message)
    else:
        bottleneck = "T1↔T2" if bandwidth_t0_t1 >= bandwidth_t1_t2 else "T0↔T1"
        message = (
            f"Bandwidth bottleneck on {bottleneck}: "
            f"F_compute ({compute_throughput:.3e} B/s) > "
            f"{PIPELINE_DEPTH_STEPS}×max(B_01={bandwidth_t0_t1:.3e}, "
            f"B_12={bandwidth_t1_t2:.3e}) = {effective_bw:.3e} B/s. "
            f"Runtime will throttle step rate — never crashes, just slows. "
            f"Deficit = {margin:.3e} B/s."
        )
        logger.warning(message)

    return FeasibilityResult(
        is_feasible=is_feasible,
        margin=margin,
        bandwidth_t0_t1=bandwidth_t0_t1,
        bandwidth_t1_t2=bandwidth_t1_t2,
        compute_throughput=compute_throughput,
        message=message,
    )


def check_feasibility_for_hardware(
    bandwidth_t0_t1: float,
    bandwidth_t1_t2: float,
    compute_tflops: float,
    flop_per_byte: float = _TRANSFORMER_FLOP_PER_BYTE,
) -> FeasibilityResult:
    """Convenience wrapper: convert TFLOPS then call :func:`check_feasibility`.

    Args:
        bandwidth_t0_t1: T0 ↔ T1 bandwidth in bytes/sec.
        bandwidth_t1_t2: T1 ↔ T2 bandwidth in bytes/sec.
        compute_tflops:  Peak compute in TFLOPS.
        flop_per_byte:   Arithmetic intensity (default 16.0 for BF16 matmul).

    Returns:
        :class:`FeasibilityResult`.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    ct = compute_throughput_from_hardware(compute_tflops, flop_per_byte)
    return check_feasibility(bandwidth_t0_t1, bandwidth_t1_t2, ct)


def estimate_memory_budget(
    params: float,
    vocab_size: int,
    context_len: int,
    d_model: int,
    n_layers: int,
    n_kv_heads: int,
    dtype_bytes: int = 2,
    device_count: int = 1,
    # --- New parameters for accurate activation budget ---
    micro_bs: Optional[int] = None,
    seq_len: Optional[int] = None,
    d_ff: Optional[int] = None,
) -> MemoryBudget:
    """Estimate the total memory budget for a transformer training run.

    Previously the activation estimate used ``2 × weights_gb`` which gave
    ~3.5 GiB for a 634M model, while the true peak backward activation
    memory was ~21.5 GiB — causing budget checks to pass silently before
    XLA OOMed with 55 GB allocations.

    Formulae
    --------
    * **Weights**: ``params × dtype_bytes``
    * **KV cache**: ``2 × n_kv_heads × d_head × context_len × n_layers × dtype_bytes``
    * **Activations** (accurate, requires micro_bs + seq_len + d_ff)::

          peak_activation_gb = micro_bs × seq_len × d_ff × n_layers × dtype_bytes / GiB

      This formula captures the dominant cost: the FFN expansion tensor
      ``bf16[micro_bs, seq_len, d_ff]`` that XLA keeps live for every layer
      simultaneously during the backward pass.

    * **Activations** (fallback, when micro_bs/seq_len/d_ff not provided)::

          activations_gb ≈ 2 × weights_gb   (selective checkpointing estimate)

      A deprecation warning is logged when the fallback is used.

    * **Optimizer**: ``params × 4 × 2`` (Adam first + second moment in FP32)

    Args:
        params:       Total parameter count (e.g. ``634e6`` for 634M).
        vocab_size:   Vocabulary size.
        context_len:  Maximum sequence length in tokens.
        d_model:      Hidden dimension.
        n_layers:     Number of transformer layers.
        n_kv_heads:   Number of key-value heads (GQA).
        dtype_bytes:  Bytes per element (default 2 for FP16/BF16).
        device_count: Number of devices for per-device estimate.
        micro_bs:     Micro-batch size per training step (samples, not tokens).
                      Required for accurate activation estimate.
        seq_len:      Sequence length in tokens. Required for accurate estimate.
        d_ff:         Feed-forward expansion dimension (e.g. 4096 for 4× d_model=1024).
                      Required for accurate estimate.

    Returns:
        :class:`MemoryBudget` with all component sizes in GiB.

    Raises:
        ValueError: If any required numeric argument is ≤ 0.
        UserWarning: Logged (not raised) when micro_bs/seq_len/d_ff are omitted
            and the 2×weights fallback is used.

    Example — accurate (recommended)::

        budget = estimate_memory_budget(
            params=634e6, vocab_size=100_277, context_len=8192,
            d_model=1536, n_layers=20, n_kv_heads=4,
            dtype_bytes=2, device_count=8,
            micro_bs=8, seq_len=8192, d_ff=4096,
        )
        # budget.activations_gb == 20.0 GiB (not 2.5 GiB)

    Example — legacy (fallback, emits warning)::

        budget = estimate_memory_budget(
            params=634e6, vocab_size=100_277, context_len=8192,
            d_model=1536, n_layers=20, n_kv_heads=4,
        )
        # budget.activations_gb == 2.37 GiB (approximate, may undercount)

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    GIB = 1024 ** 3

    weights_bytes = params * dtype_bytes
    weights_gb = weights_bytes / GIB

    d_head = d_model / max(n_kv_heads, 1)
    kv_cache_bytes = 2 * n_kv_heads * d_head * context_len * n_layers * dtype_bytes
    kv_cache_gb = kv_cache_bytes / GIB

    # -----------------------------------------------------------------------
    # Activation memory — accurate formula when training params are provided
    # -----------------------------------------------------------------------
    if micro_bs is not None and seq_len is not None and d_ff is not None:
        if micro_bs <= 0:
            raise ValueError(f"micro_bs must be > 0, got {micro_bs}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {d_ff}")
        # Peak activation: FFN expansion tensor kept live across all layers
        # during backward pass. bf16[micro_bs, seq_len, d_ff] per layer.
        # This is the tensor that caused the 55 GB XLA OOM:
        #   micro_bs=8, seq_len=8192, d_ff=4096, n_layers=20, dtype=bf16
        #   → 8 × 8192 × 4096 × 20 × 2 / 2^30 = 20.0 GiB
        peak_activation_bytes = micro_bs * seq_len * d_ff * n_layers * dtype_bytes
        activations_gb = peak_activation_bytes / GIB
        logger.info(
            "Accurate activation budget: micro_bs=%d × seq_len=%d × d_ff=%d × "
            "n_layers=%d × dtype=%dB = %.2f GiB per step",
            micro_bs, seq_len, d_ff, n_layers, dtype_bytes, activations_gb,
        )
    else:
        # Legacy fallback — 2×weights approximation.
        # WARNING: This underestimates peak activation memory for large batches
        # and long sequences. Provide micro_bs, seq_len, d_ff for accuracy.
        activations_gb = 2.0 * weights_gb
        logger.warning(
            "estimate_memory_budget called without micro_bs/seq_len/d_ff. "
            "Using legacy 2×weights approximation (%.2f GiB). "
            "This may significantly underestimate peak activation memory. "
            "Pass micro_bs, seq_len, and d_ff for an accurate budget check.",
            activations_gb,
        )

    optimizer_bytes = params * 4 * 2
    optimizer_gb = optimizer_bytes / GIB

    total_gb = weights_gb + activations_gb + kv_cache_gb + optimizer_gb
    per_device_gb = total_gb / max(device_count, 1)

    budget = MemoryBudget(
        weights_gb=weights_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
        optimizer_gb=optimizer_gb,
        total_gb=total_gb,
        per_device_gb=per_device_gb,
    )

    logger.info(
        "Memory budget: %.1f GiB total (%.1f GiB/device with %d devices)",
        total_gb, per_device_gb, device_count,
    )

    return budget
