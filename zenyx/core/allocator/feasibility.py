"""Zenyx OOM feasibility checker — pure math, no hardware dependencies.

Provides two key utilities:

1. **check_feasibility** — verifies the formal OOM-free guarantee condition:
   ``(1/B_01) + (1/B_12) ≤ (1/F_compute)``.
2. **estimate_memory_budget** — computes per-component memory requirements
   for a transformer model.
3. **compute_throughput_from_hardware** — converts raw TFLOPS into the
   correct bytes/sec unit required by ``check_feasibility``.

All functions are deterministic and side-effect free.

Unit contract
-------------
All three arguments to ``check_feasibility`` must be in **bytes/sec**
(i.e. a bandwidth).  The condition

    (1/B_01) + (1/B_12) ≤ (1/F_compute)

is a roofline arithmetic-intensity check: the left-hand side is the
total time (seconds) to move one byte through the tier hierarchy; the
right-hand side is the time (seconds) per byte of compute work.  When
the condition holds, bandwidth can always keep up with compute and
eviction never stalls training.

The common mistake is passing ``compute_tflops * 1e12`` directly as
``compute_throughput``.  TFLOPS is operations/sec, not bytes/sec.  Use
``compute_throughput_from_hardware()`` to convert correctly.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("zenyx.core.allocator.feasibility")

# Arithmetic intensity of a typical transformer layer in BF16:
# ~16 FLOP per byte (dominant cost is large matmuls at batch>1).
# This is a well-established empirical constant for LLM training.
# Reference: Kaplan et al. 2020, Korthikanti et al. 2022.
_TRANSFORMER_FLOP_PER_BYTE: float = 16.0


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeasibilityResult:
    """Outcome of the OOM feasibility check.

    Attributes:
        is_feasible:        ``True`` if the bandwidth condition holds.
        margin:             Signed margin — positive = bandwidth bottleneck (deficit),
                            negative = bandwidth headroom.
                            Computed as ``((1/B_01) + (1/B_12)) - (1/F_compute)``.
        bandwidth_t0_t1:    T0 ↔ T1 bandwidth in bytes/sec.
        bandwidth_t1_t2:    T1 ↔ T2 bandwidth in bytes/sec.
        compute_throughput: Compute throughput in bytes/sec equivalent.
        message:            Human-readable diagnostic string.

    Time complexity:  O(1) for construction.
    Space complexity: O(1).
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
            f"  margin={self.margin:.6e} sec,\n"
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

    Attributes:
        weights_gb:     Model parameter storage.
        activations_gb: Forward-pass activation memory (with selective checkpointing).
        kv_cache_gb:    Key-value cache for the full context window.
        optimizer_gb:   Optimizer state (Adam: 2× weights).
        total_gb:       Sum of all components.
        per_device_gb:  ``total_gb / device_count`` (set by caller).

    Time complexity:  O(1) for construction.
    Space complexity: O(1).
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

    ``check_feasibility`` requires all three inputs in **bytes/sec**.  Raw
    TFLOPS is operations/sec and must be divided by the arithmetic intensity
    (FLOP/byte) of the workload to obtain the equivalent memory bandwidth.

    For transformer training in BF16 the arithmetic intensity is ~16 FLOP/byte
    for the dominant matmul operations (Korthikanti et al. 2022).  This means
    the compute unit can "consume" one byte of data every
    ``1 / (tflops * 1e12 / flop_per_byte)`` seconds.

    Args:
        compute_tflops: Peak compute throughput in TFLOPS (e.g. 197.0 for
                        TPU v5 lite).
        flop_per_byte:  Arithmetic intensity of the workload in FLOP/byte.
                        Defaults to 16.0 (BF16 transformer matmuls).

    Returns:
        Effective compute throughput in **bytes/sec**.

    Raises:
        ValueError: If either argument is ≤ 0.

    Example::

        >>> ct = compute_throughput_from_hardware(197.0)
        >>> ct
        1.23125e+13   # bytes/sec — use this as compute_throughput

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    if compute_tflops <= 0:
        raise ValueError(f"compute_tflops must be > 0, got {compute_tflops}")
    if flop_per_byte <= 0:
        raise ValueError(f"flop_per_byte must be > 0, got {flop_per_byte}")
    # TFLOPS * 1e12 FLOP/sec  ÷  FLOP/byte  =  bytes/sec
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

        (1 / B_01) + (1 / B_12)  ≤  (1 / F_compute)

    where *B_01* = T1→T0 bandwidth (bytes/sec), *B_12* = T2→T1 bandwidth
    (bytes/sec), and *F_compute* = **effective memory bandwidth equivalent
    of compute** in bytes/sec.

    **Unit requirement**: all three arguments must be in **bytes/sec**.
    Do NOT pass raw TFLOPS here — use
    :func:`compute_throughput_from_hardware` to convert first.

    Args:
        bandwidth_t0_t1:    T0 ↔ T1 bandwidth in bytes/sec.
        bandwidth_t1_t2:    T1 ↔ T2 bandwidth in bytes/sec.
        compute_throughput: Effective compute throughput in bytes/sec.
                            Obtain via :func:`compute_throughput_from_hardware`.

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

    inv_bw = (1.0 / bandwidth_t0_t1) + (1.0 / bandwidth_t1_t2)
    inv_compute = 1.0 / compute_throughput
    margin = inv_bw - inv_compute  # positive = BW bottleneck, negative = headroom

    is_feasible = margin <= 0.0

    if is_feasible:
        message = (
            f"OOM-free guarantee active: bandwidth latency "
            f"({inv_bw:.3e}s) ≤ compute latency ({inv_compute:.3e}s). "
            f"Headroom = {-margin:.3e}s."
        )
        logger.info(message)
    else:
        message = (
            f"Bandwidth bottleneck: (1/B_01)+(1/B_12)={inv_bw:.3e}s > "
            f"(1/F_compute)={inv_compute:.3e}s. "
            f"Runtime will throttle step rate — never crashes, just slows. "
            f"Deficit = {margin:.3e}s."
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

    Use this instead of ``check_feasibility`` when you have raw TFLOPS from
    hardware detection (e.g. ``hw.compute_tflops``).  Internally calls
    :func:`compute_throughput_from_hardware` to convert to bytes/sec.

    Args:
        bandwidth_t0_t1: T0 ↔ T1 bandwidth in bytes/sec.
        bandwidth_t1_t2: T1 ↔ T2 bandwidth in bytes/sec.
        compute_tflops:  Peak compute in TFLOPS (e.g. ``hw.compute_tflops``).
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
) -> MemoryBudget:
    """Estimate the total memory budget for a transformer training run.

    Formulae
    --------
    * **Weights**: ``params × dtype_bytes``
    * **KV cache**: ``2 × n_kv_heads × d_head × context_len × n_layers × dtype_bytes``
      where ``d_head = d_model / n_kv_heads`` (assuming GQA head dimension).
    * **Activations**: ``≈ 2 × weights`` (training, with selective checkpointing
      every 4th layer — see Zenyx spec).
    * **Optimizer**: ``2 × weights`` (Adam maintains first & second moment).

    Args:
        params:       Total parameter count (e.g. ``70e9`` for 70 B).
        vocab_size:   Vocabulary size.
        context_len:  Maximum sequence / context length in tokens.
        d_model:      Hidden dimension.
        n_layers:     Number of transformer layers.
        n_kv_heads:   Number of key-value heads (GQA).
        dtype_bytes:  Bytes per element (default 2 for FP16/BF16).
        device_count: Number of devices for per-device estimate (default 1).

    Returns:
        :class:`MemoryBudget` with all component sizes.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    GIB = 1024 ** 3

    weights_bytes = params * dtype_bytes
    weights_gb = weights_bytes / GIB

    d_head = d_model / max(n_kv_heads, 1)
    kv_cache_bytes = 2 * n_kv_heads * d_head * context_len * n_layers * dtype_bytes
    kv_cache_gb = kv_cache_bytes / GIB

    # Selective checkpointing every 4th layer → ~2× weights
    activations_gb = 2.0 * weights_gb

    # Adam: fp32 first + second moment = 4 bytes × 2 per param
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
        total_gb,
        per_device_gb,
        device_count,
    )

    return budget
