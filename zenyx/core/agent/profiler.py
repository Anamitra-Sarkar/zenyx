"""Zenyx agent — Lightweight async CUDA event profiler.

Uses CUDA events (NOT ``MemorySnapshot``) for sub-microsecond profiling with
< 1 % overhead on training step time.  A background aggregation thread
periodically collects completed event pairs and updates running statistics.

Complexity
----------
- ``start_op`` / ``end_op``: O(1) — record a CUDA event.
- ``get_timings``: O(K) where K = number of distinct op names.
- ``get_memory_usage``: O(1) — queries driver counters.
- Background aggregation: O(N) per sweep where N = pending event pairs.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("zenyx.agent.profiler")


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ProfileHandle:
    """Opaque handle returned by :meth:`AsyncProfiler.start_op`.

    Attributes
    ----------
    op_name : str
        Name of the profiled operation.
    start_event : Any
        CUDA start event (or timestamp for CPU fallback).
    end_event : Any
        CUDA end event (populated by :meth:`end_op`).
    stream : Any
        CUDA stream on which the events were recorded.
    """

    op_name: str
    start_event: Any = None
    end_event: Any = None
    stream: Any = None

    def __repr__(self) -> str:
        return f"ProfileHandle(op={self.op_name!r})"


@dataclass
class ProfileTiming:
    """Aggregated timing statistics for one operation.

    Attributes
    ----------
    op_name : str
        Name of the profiled operation.
    total_ms : float
        Cumulative elapsed time in milliseconds.
    count : int
        Number of completed measurements.
    avg_ms : float
        Mean elapsed time in milliseconds.
    min_ms : float
        Minimum elapsed time in milliseconds.
    max_ms : float
        Maximum elapsed time in milliseconds.
    """

    op_name: str
    total_ms: float = 0.0
    count: int = 0
    avg_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ProfileTiming(op={self.op_name!r}, avg={self.avg_ms:.3f}ms, "
            f"count={self.count}, min={self.min_ms:.3f}ms, max={self.max_ms:.3f}ms)"
        )


@dataclass
class MemoryUsage:
    """Current memory usage across all three tiers.

    Values are in **gigabytes**.

    Attributes
    ----------
    t0_used_gb : float
        T0 (HBM / VRAM) used.
    t0_total_gb : float
        T0 total capacity.
    t1_used_gb : float
        T1 (CPU DRAM) used.
    t1_total_gb : float
        T1 total capacity.
    t2_used_gb : float
        T2 (NVMe) used.
    t2_total_gb : float
        T2 total capacity.
    """

    t0_used_gb: float = 0.0
    t0_total_gb: float = 0.0
    t1_used_gb: float = 0.0
    t1_total_gb: float = 0.0
    t2_used_gb: float = 0.0
    t2_total_gb: float = 0.0

    def __repr__(self) -> str:
        return (
            f"MemoryUsage("
            f"T0={self.t0_used_gb:.2f}/{self.t0_total_gb:.2f}GB, "
            f"T1={self.t1_used_gb:.2f}/{self.t1_total_gb:.2f}GB, "
            f"T2={self.t2_used_gb:.2f}/{self.t2_total_gb:.2f}GB)"
        )


# ── Profiler ─────────────────────────────────────────────────────────────


class AsyncProfiler:
    """Lightweight async CUDA event profiler with background aggregation.

    The profiler records CUDA events around operations and a daemon thread
    sweeps completed pairs to update per-op statistics.  When CUDA is
    unavailable the profiler transparently falls back to ``time.perf_counter``
    with negligible overhead.

    Parameters
    ----------
    enabled : bool
        If ``False``, all methods become no-ops.
    aggregation_interval_s : float
        How often the background thread sweeps pending events (seconds).

    Time: O(1) per start/end.  Space: O(P) where P = pending events.
    """

    def __init__(
        self,
        enabled: bool = True,
        aggregation_interval_s: float = 0.5,
    ) -> None:
        self._enabled = enabled
        self._aggregation_interval = aggregation_interval_s

        # Per-op running accumulators — updated by the background thread
        self._timings: Dict[str, ProfileTiming] = {}
        self._timings_lock = threading.Lock()

        # Pending (completed) event pairs awaiting aggregation
        self._pending: List[ProfileHandle] = []
        self._pending_lock = threading.Lock()

        # Detect CUDA availability once
        self._has_cuda = self._probe_cuda()

        # Background aggregation thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True,
            name="zenyx-profiler-agg",
        )
        if self._enabled:
            self._thread.start()

    # ── Public API ───────────────────────────────────────────────────────

    def start_op(self, op_name: str) -> ProfileHandle:
        """Record the start of an operation on the current CUDA stream.

        Time: O(1).  Space: O(1).

        Parameters
        ----------
        op_name : str
            Human-readable name for the operation (e.g. ``"fwd_attn"``).

        Returns
        -------
        ProfileHandle
            Handle to pass to :meth:`end_op`.
        """
        handle = ProfileHandle(op_name=op_name)
        if not self._enabled:
            return handle

        if self._has_cuda:
            import torch

            stream = torch.cuda.current_stream()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record(stream)
            handle.start_event = start_evt
            handle.end_event = end_evt
            handle.stream = stream
        else:
            handle.start_event = time.perf_counter()

        return handle

    def end_op(self, handle: ProfileHandle) -> None:
        """Record the end of an operation and enqueue for aggregation.

        Time: O(1).  Space: O(1).

        Parameters
        ----------
        handle : ProfileHandle
            Handle returned by :meth:`start_op`.
        """
        if not self._enabled:
            return

        if self._has_cuda:
            import torch

            if handle.end_event is not None:
                handle.end_event.record(handle.stream)
        else:
            handle.end_event = time.perf_counter()

        with self._pending_lock:
            self._pending.append(handle)

    def get_timings(self) -> Dict[str, ProfileTiming]:
        """Return accumulated timings per operation name.

        Time: O(K) where K = distinct op names.  Space: O(K).

        Returns
        -------
        Dict[str, ProfileTiming]
            Mapping from op name → aggregated timing stats.
        """
        # Flush pending before returning
        self._flush_pending()

        with self._timings_lock:
            return dict(self._timings)

    def get_memory_usage(self) -> MemoryUsage:
        """Return current GPU / CPU memory usage without heavy snapshots.

        Time: O(1).  Space: O(1).

        Returns
        -------
        MemoryUsage
            Current memory usage across all three tiers.
        """
        usage = MemoryUsage()

        if self._has_cuda:
            import torch

            try:
                t0_used = torch.cuda.memory_allocated()
                t0_total = torch.cuda.get_device_properties(0).total_mem
                usage.t0_used_gb = t0_used / (1024 ** 3)
                usage.t0_total_gb = t0_total / (1024 ** 3)
            except Exception:
                pass

        # T1 — CPU memory via /proc/meminfo (Linux) or psutil
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_info: Dict[str, int] = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    mem_info[key] = int(parts[1]) * 1024  # kB → bytes
            total = mem_info.get("MemTotal", 0)
            available = mem_info.get("MemAvailable", 0)
            usage.t1_total_gb = total / (1024 ** 3)
            usage.t1_used_gb = (total - available) / (1024 ** 3)
        except (FileNotFoundError, KeyError, ValueError):
            pass

        # T2 — NVMe / disk: not tracked at driver level; leave at 0
        return usage

    def shutdown(self) -> None:
        """Stop the background aggregation thread.

        Time: O(1).
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── Background aggregation ───────────────────────────────────────────

    def _aggregation_loop(self) -> None:
        """Daemon loop that periodically flushes pending event pairs."""
        while not self._stop_event.is_set():
            self._flush_pending()
            self._stop_event.wait(timeout=self._aggregation_interval)

    def _flush_pending(self) -> None:
        """Move completed event pairs into the timing accumulators.

        Time: O(N) where N = number of pending handles.
        """
        with self._pending_lock:
            batch = list(self._pending)
            self._pending.clear()

        if not batch:
            return

        for handle in batch:
            elapsed_ms = self._resolve_elapsed(handle)
            if elapsed_ms is None:
                # Not yet ready — put back
                with self._pending_lock:
                    self._pending.append(handle)
                continue

            with self._timings_lock:
                timing = self._timings.get(handle.op_name)
                if timing is None:
                    timing = ProfileTiming(op_name=handle.op_name)
                    self._timings[handle.op_name] = timing

                timing.total_ms += elapsed_ms
                timing.count += 1
                timing.avg_ms = timing.total_ms / timing.count
                timing.min_ms = min(timing.min_ms, elapsed_ms)
                timing.max_ms = max(timing.max_ms, elapsed_ms)

    def _resolve_elapsed(self, handle: ProfileHandle) -> Optional[float]:
        """Resolve a handle to elapsed milliseconds, or ``None`` if not ready.

        Time: O(1).
        """
        if self._has_cuda:
            import torch

            start_evt = handle.start_event
            end_evt = handle.end_event
            if start_evt is None or end_evt is None:
                return None
            # Check if the end event has completed
            if not end_evt.query():
                return None
            return start_evt.elapsed_time(end_evt)
        else:
            # CPU fallback — perf_counter timestamps in seconds
            start_t = handle.start_event
            end_t = handle.end_event
            if start_t is None or end_t is None:
                return None
            return (end_t - start_t) * 1000.0

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _probe_cuda() -> bool:
        """Return ``True`` if CUDA events are available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        n_ops = len(self._timings)
        n_pending: int
        with self._pending_lock:
            n_pending = len(self._pending)
        mode = "CUDA" if self._has_cuda else "CPU"
        return (
            f"AsyncProfiler(enabled={self._enabled}, mode={mode}, "
            f"ops={n_ops}, pending={n_pending})"
        )

    def __del__(self) -> None:
        self.shutdown()
