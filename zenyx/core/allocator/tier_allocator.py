"""Main allocator entry point orchestrating memory pool and reuse heap.

:class:`TierAllocator` is the public interface for Zenyx memory management.
It combines the pre-pinned :class:`~zenyx.core.allocator.mem_pool.MemoryPool`
with the Bélády-optimal :class:`~zenyx.core.allocator.reuse_heap.ReuseHeap`
to deliver an *eviction-as-throttle* guarantee: the runtime **never** raises
an OOM exception — it blocks (sleeps) until eviction frees sufficient space.

Async prefetching is driven by the compute graph: at each :meth:`step` call
the allocator looks ahead by :attr:`prefetch_window` operations and promotes
blocks from T1/T2 → T0 so they are resident before the kernel needs them.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from zenyx.core.allocator.reuse_heap import ComputeGraph, Op, ReuseHeap
from zenyx.core.hal.base import MemBlock, MemTier

# Lazy import to allow Phase 1 to land independently.  The actual
# ``MemoryPool`` class is resolved at first use rather than module load.
_MemoryPool: Optional[type] = None


def _get_memory_pool_class() -> type:
    """Lazily import :class:`MemoryPool` from the mem_pool module.

    Returns:
        The ``MemoryPool`` class.

    Raises:
        ImportError: If the mem_pool module is not yet available.
    """
    global _MemoryPool
    if _MemoryPool is None:
        from zenyx.core.allocator.mem_pool import MemoryPool

        _MemoryPool = MemoryPool
    return _MemoryPool


logger = logging.getLogger("zenyx.allocator.tier_allocator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BLOCK_SIZE_MB: int = 2
_MAX_BLOCK_SIZE_MB: int = 20
_DEFAULT_BLOCK_SIZE_MB: int = 4
_THROTTLE_POLL_INTERVAL: float = 0.05  # seconds
_MAX_EVICTION_RETRIES: int = 200  # ~10 seconds at 50 ms poll

# ---------------------------------------------------------------------------
# AllocatorStats
# ---------------------------------------------------------------------------


@dataclass
class AllocatorStats:
    """Snapshot of allocator health and throughput metrics.

    Attributes:
        t0_used_bytes: Bytes currently allocated in T0 (HBM/VRAM).
        t0_total_bytes: Total capacity of T0.
        t1_used_bytes: Bytes currently allocated in T1 (CPU DRAM pinned).
        t1_total_bytes: Total capacity of T1.
        t2_used_bytes: Bytes currently allocated in T2 (NVMe SSD).
        t2_total_bytes: Total capacity of T2.
        eviction_count: Total evictions performed since allocator creation.
        prefetch_requests: Total prefetch requests issued.
        prefetch_hits: Prefetch requests where the block was already resident
            in the target tier (no work needed).
        throttle_count: Number of times the allocator had to sleep-wait
            because all tiers were at capacity.
    """

    t0_used_bytes: int = 0
    t0_total_bytes: int = 0
    t1_used_bytes: int = 0
    t1_total_bytes: int = 0
    t2_used_bytes: int = 0
    t2_total_bytes: int = 0
    eviction_count: int = 0
    prefetch_requests: int = 0
    prefetch_hits: int = 0
    throttle_count: int = 0

    @property
    def prefetch_hit_rate(self) -> float:
        """Fraction of prefetch requests that were already resident.

        Returns:
            Hit rate in ``[0.0, 1.0]``, or ``0.0`` if no prefetches yet.
        """
        if self.prefetch_requests == 0:
            return 0.0
        return self.prefetch_hits / self.prefetch_requests

    def __repr__(self) -> str:
        def _fmt(used: int, total: int) -> str:
            if total == 0:
                return "0/0 MB"
            return f"{used / (1 << 20):.1f}/{total / (1 << 20):.1f} MB"

        return (
            f"AllocatorStats("
            f"T0={_fmt(self.t0_used_bytes, self.t0_total_bytes)}, "
            f"T1={_fmt(self.t1_used_bytes, self.t1_total_bytes)}, "
            f"T2={_fmt(self.t2_used_bytes, self.t2_total_bytes)}, "
            f"evictions={self.eviction_count}, "
            f"prefetch_hit_rate={self.prefetch_hit_rate:.1%}, "
            f"throttles={self.throttle_count})"
        )


# ---------------------------------------------------------------------------
# TierAllocator
# ---------------------------------------------------------------------------


class TierAllocator:
    """Orchestrates :class:`MemoryPool` and :class:`ReuseHeap`.

    The allocator implements two core invariants:

    1. **Never OOM** — eviction cascades through T0 → T1 → T2 and, as a
       last resort, the calling thread is **throttled** (``time.sleep``)
       until space becomes available.  No ``torch.cuda.OutOfMemoryError``
       is ever raised.
    2. **Prefetch ahead** — at each :meth:`step`, blocks required within
       the next :attr:`prefetch_window` operations are promoted to T0
       asynchronously so compute kernels never stall on memory transfers.

    Thread safety
    -------------
    All public methods are serialised by ``_lock`` (a ``threading.RLock``).
    Prefetch I/O is dispatched on a background daemon thread.

    Complexity (per call, *N* = tracked blocks, *W* = prefetch window)
    -------------------------------------------------------------------
    * :meth:`allocate` — *O(log N)* amortised (heap query + pool alloc).
    * :meth:`free`     — *O(1)*.
    * :meth:`step`     — *O(W · I + P · log N)* where *I* = avg inputs
      per op, *P* = prefetched blocks.

    Args:
        hardware: Hardware descriptor produced by ``detect_hardware()``.
        block_size_mb: Block granularity in MiB (clamped to 2–20).
    """

    def __init__(self, hardware: Any, block_size_mb: int = _DEFAULT_BLOCK_SIZE_MB) -> None:
        """Create a :class:`TierAllocator`.

        Instantiates the underlying :class:`MemoryPool` and
        :class:`ReuseHeap`, and validates the block size.

        Args:
            hardware: A ``HardwareInfo`` instance (from ``detect_hardware``).
            block_size_mb: Desired block size in MiB.  Clamped to
                ``[2, 20]`` per spec.

        Time:  O(1) plus pool pre-allocation time.
        Space: O(pool capacity).
        """
        # Clamp block size to spec range.
        if block_size_mb < _MIN_BLOCK_SIZE_MB:
            logger.warning(
                "block_size_mb=%d below minimum %d, clamping",
                block_size_mb,
                _MIN_BLOCK_SIZE_MB,
            )
            block_size_mb = _MIN_BLOCK_SIZE_MB
        elif block_size_mb > _MAX_BLOCK_SIZE_MB:
            logger.warning(
                "block_size_mb=%d above maximum %d, clamping",
                block_size_mb,
                _MAX_BLOCK_SIZE_MB,
            )
            block_size_mb = _MAX_BLOCK_SIZE_MB

        self._block_size_bytes: int = block_size_mb * (1 << 20)
        self._hardware = hardware

        # Core components
        PoolCls = _get_memory_pool_class()
        self._pool: Any = PoolCls(hardware, block_size_mb=block_size_mb)
        self._heap: ReuseHeap = ReuseHeap()

        # Compute graph tracking
        self._graph: Optional[ComputeGraph] = None
        self._current_op_idx: int = 0

        # Prefetch configuration
        self.prefetch_window: int = 3

        # Block registry: block_id → MemBlock currently allocated.
        self._blocks: dict[int, MemBlock] = {}

        # Statistics
        self._eviction_count: int = 0
        self._prefetch_requests: int = 0
        self._prefetch_hits: int = 0
        self._throttle_count: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Prefetch thread coordination
        self._prefetch_thread: Optional[threading.Thread] = None

        logger.info(
            "TierAllocator created: block_size=%d MiB, hardware=%s",
            block_size_mb,
            getattr(hardware, "device_name", "unknown"),
        )

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        size_bytes: int,
        tier: MemTier = MemTier.T0,
        block_id: Optional[int] = None,
    ) -> MemBlock:
        """Allocate a memory block, evicting if necessary.

        Attempts to allocate from the preferred *tier* via the underlying
        :class:`MemoryPool`.  If the tier is full the allocator uses the
        :class:`ReuseHeap` to select the best eviction candidate and
        demotes it to a lower tier, then retries.

        If **all** tiers are full the calling thread is throttled (sleeps)
        until space becomes available — eviction is a throughput throttle,
        **never** an exception.

        Args:
            size_bytes: Requested allocation size in bytes.
            tier: Preferred memory tier (default ``T0``).
            block_id: Optional explicit block identifier.  If ``None`` one
                is assigned automatically.

        Returns:
            A :class:`MemBlock` representing the allocated region.

        Time:  O(log N) amortised (eviction heap query).
        Space: O(1) beyond the allocation itself.
        """
        with self._lock:
            return self._allocate_inner(size_bytes, tier, block_id)

    def _allocate_inner(
        self,
        size_bytes: int,
        tier: MemTier,
        block_id: Optional[int],
    ) -> MemBlock:
        """Allocation logic without outer lock (caller holds ``_lock``)."""
        retries = 0

        while True:
            # --- Attempt allocation in the pool -------------------------
            try:
                block = self._pool.allocate(size_bytes, preferred_tier=tier)
                if block_id is not None:
                    block.block_id = block_id
                self._blocks[block.block_id] = block
                self._heap.update_access(block.block_id, self._current_op_idx)
                return block
            except Exception:
                # Pool could not satisfy the request — need eviction.
                pass

            # --- Eviction cascade: T0 → T1 → T2 -----------------------
            evicted = self._try_evict(tier)
            if evicted:
                retries = 0
                continue

            # --- All tiers full — throttle ------------------------------
            retries += 1
            self._throttle_count += 1
            if retries == 1:
                logger.warning(
                    "Zenyx throttling: all memory tiers at capacity, "
                    "waiting for eviction to complete"
                )
            if retries > _MAX_EVICTION_RETRIES:
                logger.error(
                    "Throttled for %d iterations (%.1f s) — "
                    "possible deadlock or insufficient total memory",
                    retries,
                    retries * _THROTTLE_POLL_INTERVAL,
                )
                # Still do not raise — keep trying.

            # Release lock while sleeping so other threads can free memory.
            self._lock.release()
            try:
                time.sleep(_THROTTLE_POLL_INTERVAL)
            finally:
                self._lock.acquire()

    def _try_evict(self, needed_tier: MemTier) -> bool:
        """Evict one block to make room in *needed_tier*.

        Selects the best candidate from the :class:`ReuseHeap` (Bélády
        or LRU fallback) and moves it to a lower tier.

        Returns:
            ``True`` if a block was successfully evicted, ``False``
            otherwise.

        Time:  O(log N)
        Space: O(1)
        """
        candidate_id = self._heap.get_eviction_candidate()
        if candidate_id is None:
            return False

        candidate = self._blocks.get(candidate_id)
        if candidate is None:
            # Stale entry — remove from heap and retry next time.
            self._heap.remove_block(candidate_id)
            return False

        # Determine destination tier for eviction.
        dest_tier = self._demote_tier(candidate.tier)
        if dest_tier is None:
            # Block is already in the lowest tier — nothing to evict to.
            return False

        try:
            evicted_block = self._pool.evict(candidate, dest_tier)
            self._blocks[candidate_id] = evicted_block
            self._eviction_count += 1
            logger.debug(
                "Evicted block %d: %s → %s",
                candidate_id,
                candidate.tier.name,
                dest_tier.name,
            )
            return True
        except Exception:
            logger.debug("Eviction of block %d failed", candidate_id, exc_info=True)
            return False

    @staticmethod
    def _demote_tier(current: MemTier) -> Optional[MemTier]:
        """Return the next lower tier, or ``None`` if already at T2.

        Time:  O(1)
        Space: O(1)
        """
        if current == MemTier.T0:
            return MemTier.T1
        if current == MemTier.T1:
            return MemTier.T2
        return None

    @staticmethod
    def _promote_tier(current: MemTier) -> Optional[MemTier]:
        """Return the next higher tier, or ``None`` if already at T0.

        Time:  O(1)
        Space: O(1)
        """
        if current == MemTier.T2:
            return MemTier.T1
        if current == MemTier.T1:
            return MemTier.T0
        return None

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self, block: MemBlock) -> None:
        """Release *block* and remove it from the eviction heap.

        Args:
            block: Previously allocated :class:`MemBlock`.

        Time:  O(1)
        Space: O(1)
        """
        with self._lock:
            bid = block.block_id
            self._blocks.pop(bid, None)
            self._heap.remove_block(bid)
            try:
                self._pool.free(block)
            except Exception:
                logger.debug("Pool free for block %d failed", bid, exc_info=True)
            logger.debug("Freed block %d", bid)

    # ------------------------------------------------------------------
    # Compute graph registration
    # ------------------------------------------------------------------

    def register_compute_graph(self, graph: ComputeGraph) -> None:
        """Register (or update) the compute graph for optimal eviction.

        If a graph was previously registered the heap is rebuilt
        asynchronously so that in-flight allocations are not stalled.
        The first registration is synchronous (blocking) to ensure the
        heap is ready before the first :meth:`step`.

        Args:
            graph: The transformer (or general) compute graph.

        Time:  O(G + N log N) for first registration, O(1) for
            subsequent (async rebuild).
        Space: O(G + N)
        """
        with self._lock:
            is_first = self._graph is None
            self._graph = graph
            self._current_op_idx = 0

            if is_first:
                self._heap.build_from_graph(graph)
                logger.info("Compute graph registered (sync): %r", graph)
            else:
                self._heap.rebuild_async(graph)
                logger.info("Compute graph updated (async rebuild): %r", graph)

    # ------------------------------------------------------------------
    # Step / prefetch
    # ------------------------------------------------------------------

    def step(self, op_idx: int) -> None:
        """Advance the allocator to operation *op_idx*.

        Updates access patterns in the reuse heap for every block
        consumed by the current operation and triggers asynchronous
        prefetching for blocks needed within the next
        :attr:`prefetch_window` operations.

        Args:
            op_idx: Index of the operation about to execute.

        Time:  O(I + W · I + P · log N)  — I = inputs of current op,
            W = prefetch window, P = blocks prefetched.
        Space: O(W · I)
        """
        with self._lock:
            self._current_op_idx = op_idx

            # Update heap for blocks consumed by the current op.
            if self._graph is not None and op_idx < len(self._graph.ops):
                current_op = self._graph.ops[op_idx]
                for bid in current_op.input_blocks:
                    self._heap.update_access(bid, op_idx)

            # Kick off prefetching in background.
            needed = self._heap.blocks_needed_in_window(
                op_idx + 1, self.prefetch_window
            )
            if needed:
                self._dispatch_prefetch(needed)

    def _dispatch_prefetch(self, block_ids: list[int]) -> None:
        """Schedule async prefetch of *block_ids* to T0.

        Blocks that are already in T0 are counted as prefetch hits.
        The actual data movement is performed on a daemon thread so the
        calling :meth:`step` returns immediately.

        Args:
            block_ids: Blocks to promote to T0.

        Time:  O(P)
        Space: O(P)
        """
        to_prefetch: list[tuple[int, MemBlock]] = []

        for bid in block_ids:
            self._prefetch_requests += 1
            block = self._blocks.get(bid)
            if block is None:
                continue
            if block.tier == MemTier.T0:
                self._prefetch_hits += 1
                continue
            to_prefetch.append((bid, block))

        if not to_prefetch:
            return

        # Fire-and-forget daemon thread for the actual data movement.
        def _prefetch_worker() -> None:
            for bid, blk in to_prefetch:
                try:
                    promoted = self._pool.prefetch(blk, MemTier.T0)
                    with self._lock:
                        self._blocks[bid] = promoted
                    logger.debug(
                        "Prefetched block %d: %s → T0", bid, blk.tier.name
                    )
                except Exception:
                    logger.debug(
                        "Prefetch of block %d failed", bid, exc_info=True
                    )

        thread = threading.Thread(
            target=_prefetch_worker, name="zenyx-prefetch", daemon=True
        )
        thread.start()
        self._prefetch_thread = thread

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> AllocatorStats:
        """Return a snapshot of allocator health metrics.

        The tier usage numbers are read from the underlying
        :class:`MemoryPool`.

        Returns:
            :class:`AllocatorStats` reflecting the current state.

        Time:  O(1)
        Space: O(1)
        """
        with self._lock:
            # Read tier capacities from pool (duck-typed — pool exposes
            # per-tier usage via attributes or a method).
            t0_used = getattr(self._pool, "t0_used_bytes", 0)
            t0_total = getattr(self._pool, "t0_total_bytes", 0)
            t1_used = getattr(self._pool, "t1_used_bytes", 0)
            t1_total = getattr(self._pool, "t1_total_bytes", 0)
            t2_used = getattr(self._pool, "t2_used_bytes", 0)
            t2_total = getattr(self._pool, "t2_total_bytes", 0)

            return AllocatorStats(
                t0_used_bytes=t0_used,
                t0_total_bytes=t0_total,
                t1_used_bytes=t1_used,
                t1_total_bytes=t1_total,
                t2_used_bytes=t2_used,
                t2_total_bytes=t2_total,
                eviction_count=self._eviction_count,
                prefetch_requests=self._prefetch_requests,
                prefetch_hits=self._prefetch_hits,
                throttle_count=self._throttle_count,
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            stats = self.get_stats()
            return (
                f"TierAllocator("
                f"T0={stats.t0_used_bytes / (1 << 20):.1f}/"
                f"{stats.t0_total_bytes / (1 << 20):.1f} MB, "
                f"T1={stats.t1_used_bytes / (1 << 20):.1f}/"
                f"{stats.t1_total_bytes / (1 << 20):.1f} MB, "
                f"T2={stats.t2_used_bytes / (1 << 20):.1f}/"
                f"{stats.t2_total_bytes / (1 << 20):.1f} MB, "
                f"evictions={self._eviction_count}, "
                f"prefetch_hit_rate={stats.prefetch_hit_rate:.1%})"
            )
