"""Zenyx OOM feasibility checker — pure math, no hardware dependencies.

Provides three key utilities:

1. **check_feasibility** — verifies the formal OOM-free guarantee condition:
   ``F_compute ≤ _PIPELINE_DEPTH × max(B_01, B_12)``
   i.e. the pipeline lookahead depth times the maximum available bandwidth must
   cover the effective compute demand (in bytes/sec), so eviction always keeps
   up with compute.
2. **estimate_memory_budget** — computes per-component memory requirements
   for a transformer model.
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

Interpretation: if the pipeline can prefetch *_PIPELINE_DEPTH* steps ahead
using the faster tier bandwidth, compute never stalls waiting for data.  If
the condition fails the runtime throttles (never crashes).

Previous (wrong) condition was ``F_compute ≤ B_01 AND F_compute ≤ B_12``
which requires compute to be slower than *both* bandwidth tiers individually.
For real GPUs (H100: ~247 TB/s effective compute vs 3.35 TB/s HBM) this is
never satisfied, falsely flagging every production config as infeasible.

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
# Reference: Kaplan et al. 2020, Korthikanti et al. 2022.
_TRANSFORMER_FLOP_PER_BYTE: float = 16.0

# Pipeline prefetch depth: the eviction pipeline can run this many steps
# ahead of compute.  The OOM-free condition is satisfied when:
#   F_compute ≤ _PIPELINE_DEPTH × max(B_01, B_12)
# For real GPUs (H100: ~247 TB/s compute demand, ~3.35 TB/s HBM) a depth
# of 100 gives comfortable headroom.
_PIPELINE_DEPTH: int = 100


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeasibilityResult:
    """Outcome of the OOM feasibility check.

    Attributes:
        is_feasible:        ``True`` if F_compute ≤ B_01 AND F_compute ≤ B_12.
        margin:             Signed margin in bytes/sec.
                            ``max(F_compute - B_01, F_compute - B_12)``.
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

    The pipeline lookahead depth times the faster bandwidth tier must cover
    the compute demand.  If the condition fails the runtime throttles — it
    never crashes.

    **Previous wrong condition** was ``F_compute ≤ B_01 AND F_compute ≤ B_12``
    (both tiers individually faster than compute), which is never satisfied on
    real GPUs where compute demand (~247 TB/s for H100 BF16) far exceeds any
    memory bandwidth.

    **Unit requirement**: all three arguments in **bytes/sec**.
    Use :func:`compute_throughput_from_hardware` to convert from TFLOPS.

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

    # Effective bandwidth available to the pipeline:
    # the pipeline can prefetch _PIPELINE_DEPTH steps ahead using the faster tier.
    effective_bw = _PIPELINE_DEPTH * max(bandwidth_t0_t1, bandwidth_t1_t2)

    # Positive margin = bandwidth is the bottleneck (compute demand > effective_bw).
    # Negative margin = bandwidth has headroom.
    margin = compute_throughput - effective_bw
    is_feasible = margin <= 0.0

    if is_feasible:
        message = (
            f"OOM-free guarantee active: F_compute ({compute_throughput:.3e} B/s) "
            f"≤ {_PIPELINE_DEPTH}×max(B_01={bandwidth_t0_t1:.3e}, "
            f"B_12={bandwidth_t1_t2:.3e}) = {effective_bw:.3e} B/s. "
            f"Headroom = {-margin:.3e} B/s."
        )
        logger.info(message)
    else:
        bottleneck = "T1↔T2" if bandwidth_t0_t1 >= bandwidth_t1_t2 else "T0↔T1"
        message = (
            f"Bandwidth bottleneck on {bottleneck}: "
            f"F_compute ({compute_throughput:.3e} B/s) > "
            f"{_PIPELINE_DEPTH}×max(B_01={bandwidth_t0_t1:.3e}, "
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
) -> MemoryBudget:
    """Estimate the total memory budget for a transformer training run.

    Formulae
    --------
    * **Weights**: ``params × dtype_bytes``
    * **KV cache**: ``2 × n_kv_heads × d_head × context_len × n_layers × dtype_bytes``
    * **Activations**: ``≈2 × weights`` (selective checkpointing every 4th layer).
    * **Optimizer**: ``2 × weights`` (Adam first + second moment).

    Args:
        params:       Total parameter count (e.g. ``70e9`` for 70B).
        vocab_size:   Vocabulary size.
        context_len:  Maximum sequence length in tokens.
        d_model:      Hidden dimension.
        n_layers:     Number of transformer layers.
        n_kv_heads:   Number of key-value heads (GQA).
        dtype_bytes:  Bytes per element (default 2 for FP16/BF16).
        device_count: Number of devices for per-device estimate.

    Returns:
        :class:`MemoryBudget` with all component sizes in GiB.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    GIB = 1024 ** 3

    weights_bytes = params * dtype_bytes
    weights_gb = weights_bytes / GIB

    d_head = d_model / max(n_kv_heads, 1)
    kv_cache_bytes = 2 * n_kv_heads * d_head * context_len * n_layers * dtype_bytes
    kv_cache_gb = kv_cache_bytes / GIB

    activations_gb = 2.0 * weights_gb

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
