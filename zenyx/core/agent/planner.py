"""Zenyx agent — Parallelism planner.

Determines optimal TP / PP / DP / Ring degrees given model size, context
length, vocabulary size, and the detected hardware topology.  Supports
replanning from live profiling data at curriculum shifts.

Planning Heuristics (from spec)
-------------------------------
1. Compute total memory requirement.
2. Determine minimum devices needed for memory.
3. Maximise DP degree (best for throughput) while ensuring TP + PP can fit
   the model.
4. TP degree: prefer powers of 2, match NVLink domain size (typically 8 for
   DGX).
5. PP degree: minimise bubble fraction. Prefer *braided* if m >> P.
6. Ring attention degree: ≥ 40 for 1M context with 120B model.
7. Warn if hardware is insufficient.

Complexity
----------
``plan()`` is O(log D) where D = device count (binary-search-style fitting).
``replan()`` is O(K) where K = distinct profiled ops.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from zenyx.core.hal.base import HardwareInfo
from zenyx.ops.comm.topology import Topology

logger = logging.getLogger("zenyx.agent.planner")


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ParallelismPlan:
    """A complete parallelism configuration for a training run.

    Attributes
    ----------
    tp_degree : int
        Tensor-parallelism degree (intra-node, power of 2).
    pp_degree : int
        Pipeline-parallelism degree (stages).
    dp_degree : int
        Data-parallelism degree.
    ring_degree : int
        Ring-attention degree (sequence parallelism).
    schedule_type : str
        Pipeline schedule: ``"braided"``, ``"1f1b"``, or ``"gpipe"``.
    estimated_bubble_fraction : float
        Expected pipeline bubble as a fraction [0, 1].
    estimated_tokens_per_sec : float
        Projected throughput (tokens / second).
    memory_per_device_gb : float
        Estimated peak memory per device in GB.
    """

    tp_degree: int = 1
    pp_degree: int = 1
    dp_degree: int = 1
    ring_degree: int = 1
    schedule_type: str = "1f1b"
    estimated_bubble_fraction: float = 0.0
    estimated_tokens_per_sec: float = 0.0
    memory_per_device_gb: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ParallelismPlan(TP={self.tp_degree}, PP={self.pp_degree}, "
            f"DP={self.dp_degree}, Ring={self.ring_degree}, "
            f"schedule={self.schedule_type!r}, "
            f"bubble={self.estimated_bubble_fraction:.3f}, "
            f"tok/s={self.estimated_tokens_per_sec:,.0f}, "
            f"mem/dev={self.memory_per_device_gb:.2f}GB)"
        )


# ── Planner ──────────────────────────────────────────────────────────────


class ParallelismPlanner:
    """Plans TP / PP / DP / Ring degrees for a training run.

    Parameters
    ----------
    hardware : HardwareInfo
        Detected hardware capabilities.
    topology : Topology
        Inter-device topology graph.

    Time: O(1) construction.  Space: O(1).
    """

    # NVLink domain size for DGX-class nodes
    _NVLINK_DOMAIN_SIZE = 8
    # KV cache overhead factor (2 for K + V, dtype_bytes=2 for FP16)
    _DTYPE_BYTES = 2
    # Default microbatch count for braided schedule feasibility check
    _DEFAULT_MICROBATCHES = 64

    def __init__(self, hardware: HardwareInfo, topology: Topology) -> None:
        self._hw = hardware
        self._topo = topology
        self._last_plan: Optional[ParallelismPlan] = None

    # ── Public API ───────────────────────────────────────────────────────

    def plan(
        self,
        model_params: float,
        vocab_size: int,
        context_len: int,
        batch_size: int,
    ) -> ParallelismPlan:
        """Compute an optimal parallelism plan for the given workload.

        Time: O(log D) where D = device count.  Space: O(1).

        Parameters
        ----------
        model_params : float
            Number of model parameters (e.g. 7e9 for 7 B).
        vocab_size : int
            Vocabulary size.
        context_len : int
            Maximum sequence length.
        batch_size : int
            Global micro-batch size.

        Returns
        -------
        ParallelismPlan
            Recommended parallelism configuration.
        """
        total_devices = self._hw.device_count
        per_device_mem_gb = self._hw.per_device_memory_gb

        # ── Step 1: estimate total memory requirement ────────────────────
        weights_gb = self._estimate_weights_gb(model_params)
        optimizer_gb = weights_gb * 2.0  # Adam: 2× model states
        activations_gb = weights_gb * 2.0  # ~2× weights with checkpointing
        kv_cache_gb = self._estimate_kv_cache_gb(model_params, context_len, vocab_size)
        total_memory_gb = weights_gb + optimizer_gb + activations_gb + kv_cache_gb

        # ── Step 2: minimum devices for memory ──────────────────────────
        usable_mem_gb = per_device_mem_gb * 0.85  # 85% usable
        min_devices_mem = max(1, math.ceil(total_memory_gb / usable_mem_gb))

        if min_devices_mem > total_devices:
            logger.warning(
                "Hardware insufficient: need ≥%d devices for %.1fGB model, "
                "but only %d available. Will attempt with heavy offloading.",
                min_devices_mem,
                total_memory_gb,
                total_devices,
            )

        # ── Step 3: determine Ring degree ────────────────────────────────
        ring_degree = self._compute_ring_degree(
            model_params, context_len, total_devices,
        )

        # Devices available for TP × PP × DP after ring
        devices_after_ring = max(1, total_devices // ring_degree)

        # ── Step 4: determine TP degree ──────────────────────────────────
        tp_degree = self._compute_tp_degree(
            weights_gb, per_device_mem_gb, devices_after_ring,
        )

        # ── Step 5: determine PP degree ──────────────────────────────────
        devices_after_tp = max(1, devices_after_ring // tp_degree)
        pp_degree = self._compute_pp_degree(
            weights_gb, per_device_mem_gb, tp_degree, devices_after_tp,
        )

        # ── Step 6: maximise DP degree ───────────────────────────────────
        dp_degree = max(1, devices_after_ring // (tp_degree * pp_degree))

        # ── Step 7: pipeline schedule ────────────────────────────────────
        schedule_type, bubble_frac = self._choose_schedule(pp_degree)

        # ── Step 8: throughput estimate ──────────────────────────────────
        mem_per_device = total_memory_gb / (tp_degree * pp_degree * dp_degree * ring_degree)
        tokens_per_sec = self._estimate_throughput(
            model_params,
            batch_size,
            context_len,
            dp_degree,
            bubble_frac,
        )

        plan = ParallelismPlan(
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            dp_degree=dp_degree,
            ring_degree=ring_degree,
            schedule_type=schedule_type,
            estimated_bubble_fraction=bubble_frac,
            estimated_tokens_per_sec=tokens_per_sec,
            memory_per_device_gb=mem_per_device,
        )
        self._last_plan = plan
        logger.info("Parallelism plan: %s", plan)
        return plan

    def replan(
        self,
        profiling_data: Dict[str, Any],
        model_params: float = 0.0,
        vocab_size: int = 0,
        context_len: int = 0,
        batch_size: int = 0,
    ) -> ParallelismPlan:
        """Replan based on actual profiling data.

        Called only at curriculum shifts or every N steps. Adjusts degrees
        based on observed compute-vs-communication balance.

        Time: O(K) where K = number of distinct profiled ops.  Space: O(1).

        Parameters
        ----------
        profiling_data : Dict[str, Any]
            Mapping from op name to :class:`ProfileTiming` objects.
        model_params : float
            Updated model parameter count.
        vocab_size : int
            Updated vocabulary size.
        context_len : int
            Updated context length.
        batch_size : int
            Updated global batch size.

        Returns
        -------
        ParallelismPlan
            Updated parallelism configuration.
        """
        # Analyse profiling data for compute/comm imbalance
        total_compute_ms = 0.0
        total_comm_ms = 0.0
        for op_name, timing in profiling_data.items():
            avg = getattr(timing, "avg_ms", 0.0)
            if any(k in op_name.lower() for k in ("comm", "allreduce", "send", "recv", "ring")):
                total_comm_ms += avg
            else:
                total_compute_ms += avg

        # If we have a valid prior plan and new workload params, replan
        if model_params > 0 and context_len > 0:
            new_plan = self.plan(model_params, vocab_size, context_len, batch_size)
        elif self._last_plan is not None:
            new_plan = ParallelismPlan(
                tp_degree=self._last_plan.tp_degree,
                pp_degree=self._last_plan.pp_degree,
                dp_degree=self._last_plan.dp_degree,
                ring_degree=self._last_plan.ring_degree,
                schedule_type=self._last_plan.schedule_type,
                estimated_bubble_fraction=self._last_plan.estimated_bubble_fraction,
                estimated_tokens_per_sec=self._last_plan.estimated_tokens_per_sec,
                memory_per_device_gb=self._last_plan.memory_per_device_gb,
            )
        else:
            new_plan = ParallelismPlan()

        # Adjust TP/DP balance if comm overhead is too high
        if total_compute_ms > 0 and total_comm_ms > 0:
            comm_ratio = total_comm_ms / (total_compute_ms + total_comm_ms)
            if comm_ratio > 0.3 and new_plan.tp_degree > 1:
                # Communication-bound: reduce TP, increase DP
                new_plan.tp_degree = max(1, new_plan.tp_degree // 2)
                new_plan.dp_degree = new_plan.dp_degree * 2
                logger.info(
                    "Replan: comm overhead %.1f%% — reducing TP to %d, increasing DP to %d",
                    comm_ratio * 100,
                    new_plan.tp_degree,
                    new_plan.dp_degree,
                )
            elif comm_ratio < 0.1 and new_plan.dp_degree > 1:
                # Compute-bound with room: increase TP for faster per-op execution
                max_tp = min(
                    new_plan.tp_degree * 2,
                    self._NVLINK_DOMAIN_SIZE,
                    self._hw.device_count,
                )
                if max_tp > new_plan.tp_degree:
                    new_plan.dp_degree = max(1, new_plan.dp_degree // 2)
                    new_plan.tp_degree = max_tp
                    logger.info(
                        "Replan: compute-bound (comm %.1f%%) — increasing TP to %d",
                        comm_ratio * 100,
                        new_plan.tp_degree,
                    )

        self._last_plan = new_plan
        return new_plan

    # ── Private helpers ──────────────────────────────────────────────────

    def _estimate_weights_gb(self, model_params: float) -> float:
        """Estimate model weights size in GB (FP16 = 2 bytes/param).

        Time: O(1).
        """
        return model_params * self._DTYPE_BYTES / (1024 ** 3)

    def _estimate_kv_cache_gb(
        self,
        model_params: float,
        context_len: int,
        vocab_size: int,
    ) -> float:
        """Estimate KV cache size in GB.

        Uses the formula:
            2 × n_kv_heads × d_head × context_len × n_layers × dtype_bytes

        When architecture details are unknown, estimates from param count.

        Time: O(1).
        """
        # Estimate architecture from params
        d_model = int(math.sqrt(model_params / 12))
        n_layers = max(1, int(model_params / (12 * d_model * d_model)))
        n_kv_heads = max(1, d_model // 128)  # GQA heuristic
        d_head = d_model // max(1, n_kv_heads * 8)  # Assume n_heads = 8 × n_kv_heads

        kv_bytes = (
            2  # K + V
            * n_kv_heads
            * d_head
            * context_len
            * n_layers
            * self._DTYPE_BYTES
        )
        return kv_bytes / (1024 ** 3)

    def _compute_ring_degree(
        self,
        model_params: float,
        context_len: int,
        total_devices: int,
    ) -> int:
        """Determine ring-attention degree.

        Spec: ≥ 40 for 1M context with 120B model.

        Time: O(1).
        """
        if context_len <= 8192:
            return 1  # Short context — no ring needed

        # Scale ring degree with context length
        # At 1M context, 120B params → need ≥40
        # Linear scaling: ring ∝ context_len / 25_000
        target_ring = max(1, context_len // 25_000)

        # For very large models, increase ring to distribute KV cache
        if model_params >= 100e9:
            target_ring = max(target_ring, 40)
        elif model_params >= 10e9:
            target_ring = max(target_ring, context_len // 50_000)

        # Cap at available devices
        ring_degree = min(target_ring, total_devices)

        # Ensure we don't use more than 50% of devices for ring alone
        ring_degree = min(ring_degree, total_devices // 2) if total_devices > 2 else 1

        # At minimum 1
        return max(1, ring_degree)

    def _compute_tp_degree(
        self,
        weights_gb: float,
        per_device_mem_gb: float,
        available_devices: int,
    ) -> int:
        """Determine tensor-parallelism degree.

        Prefers powers of 2, capped at NVLink domain size (8).

        Time: O(1).
        """
        if available_devices <= 1:
            return 1

        usable = per_device_mem_gb * 0.85

        # TP needed to fit weights in memory (weights split across TP)
        min_tp_for_mem = max(1, math.ceil(weights_gb / usable))

        # Round up to next power of 2
        tp = 1
        while tp < min_tp_for_mem:
            tp *= 2

        # Cap at NVLink domain size and available devices
        tp = min(tp, self._NVLINK_DOMAIN_SIZE, available_devices)

        # Ensure power of 2
        tp = 2 ** int(math.log2(tp)) if tp > 0 else 1
        return max(1, tp)

    def _compute_pp_degree(
        self,
        weights_gb: float,
        per_device_mem_gb: float,
        tp_degree: int,
        available_devices: int,
    ) -> int:
        """Determine pipeline-parallelism degree.

        Minimises bubble fraction — avoid PP if model fits with TP alone.

        Time: O(1).
        """
        if available_devices <= 1:
            return 1

        usable = per_device_mem_gb * 0.85
        # With TP, each device holds weights/TP + optimizer/TP + activations
        mem_per_device_with_tp = (weights_gb * 3.0) / tp_degree  # weights + optimizer + activations

        if mem_per_device_with_tp <= usable:
            return 1  # Fits without PP

        # Need PP to further split
        pp = max(1, math.ceil(mem_per_device_with_tp / usable))
        pp = min(pp, available_devices)
        return pp

    def _choose_schedule(self, pp_degree: int) -> tuple[str, float]:
        """Choose the pipeline schedule and estimate bubble fraction.

        Time: O(1).

        Returns
        -------
        tuple[str, float]
            (schedule_name, estimated_bubble_fraction).
        """
        if pp_degree <= 1:
            return ("1f1b", 0.0)

        # Braided schedule: ~0 bubble if microbatches >> pipeline stages
        if self._DEFAULT_MICROBATCHES >= pp_degree * 4:
            return ("braided", 0.01)  # Near-zero bubble

        # 1F1B: bubble = 1/P
        bubble_1f1b = 1.0 / pp_degree

        # GPipe: bubble = (P-1)/P
        bubble_gpipe = (pp_degree - 1) / pp_degree

        if pp_degree <= 4:
            return ("1f1b", bubble_1f1b)

        return ("braided", 0.01)

    def _estimate_throughput(
        self,
        model_params: float,
        batch_size: int,
        context_len: int,
        dp_degree: int,
        bubble_frac: float,
    ) -> float:
        """Estimate tokens/sec throughput.

        Uses a simple roofline model: tokens/sec = (FLOPS / flops_per_token)
        × (1 - bubble) × dp_degree, with MFU ≈ 45%.

        Time: O(1).
        """
        # FLOPs per token ≈ 6 × params (forward + backward)
        flops_per_token = 6 * model_params

        # Total available FLOPS across all devices (only DP replicas contribute unique tokens)
        device_tflops = self._hw.compute_tflops
        total_flops = device_tflops * 1e12 * dp_degree

        # Model FLOPs utilisation
        mfu = 0.45 if self._hw.backend == "cuda" else 0.30

        tokens_per_sec = (total_flops * mfu * (1.0 - bubble_frac)) / flops_per_token
        return max(0.0, tokens_per_sec)

    def __repr__(self) -> str:
        plan_str = repr(self._last_plan) if self._last_plan else "None"
        return (
            f"ParallelismPlanner(hw={self._hw.device_name!r}, "
            f"devices={self._hw.device_count}, "
            f"last_plan={plan_str})"
        )
