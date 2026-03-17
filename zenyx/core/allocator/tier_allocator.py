"""Main allocator entry point orchestrating memory pool and reuse heap.

:class:`TierAllocator` is the public interface for Zenyx memory management.
It combines the pre-pinned :class:`~zenyx.core.allocator.mem_pool.MemoryPool`
with the Bélády-optimal :class:`~zenyx.core.allocator.reuse_heap.ReuseHeap`
to deliver an *eviction-as-throttle* guarantee: the runtime **never** raises
an OOM exception — it blocks (sleeps) until eviction frees sufficient space.

Async prefetching is driven by the compute graph: at each :meth:`step` call
the allocator looks ahead by :attr:`prefetch_window` operations and promotes
blocks from T1/T2 → T0 so they are resident before the kernel needs them.

Fix notes
---------
* get_stats(): previously used getattr(self._pool, "t0_used_bytes", 0) etc.
  MemoryPool exposes no such top-level attributes — the data lives inside
  self._tiers[MemTier.Tx].allocated / .capacity.  All seven getattr calls
  resolved to 0, making every AllocatorStats snapshot all-zeros.  Fixed by
  calling self._pool.usage() which returns {MemTier: (allocated, capacity)}.

* _prefetch_worker deadlock: the worker closure previously acquired
  self._lock (RLock) to write self._blocks[bid].  If the main thread was
  blocked inside _allocate_inner on _space_available.wait() while still
  holding _lock, the worker would deadlock waiting for the same lock.
  Fixed by introducing a separate _prefetch_lock (plain threading.Lock)
  that guards only the _blocks dict writes from the prefetch path.
  The Condition / _lock is never acquired by any background thread.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from zenyx.core.allocator.reuse_heap import ComputeGraph, Op, ReuseHeap
from zenyx.core.hal.base import MemBlock, MemTier
from zenyx.core.allocator.constants import PIPELINE_DEPTH_STEPS

_MemoryPool: Optional[type] = None


def _get_memory_pool_class() -> type:
    global _MemoryPool
    if _MemoryPool is None:
        from zenyx.core.allocator.mem_pool import MemoryPool
        _MemoryPool = MemoryPool
    return _MemoryPool


logger = logging.getLogger("zenyx.allocator.tier_allocator")

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
    """Snapshot of allocator health and throughput metrics."""

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
    Public methods serialised by ``_lock`` (RLock + Condition).
    Prefetch workers use the separate ``_prefetch_lock`` (plain Lock) so
    they never contend with the allocator's Condition variable.

    Lock ordering (must always be acquired in this order to avoid deadlock):
        1. _lock  (main allocator state)
        2. _prefetch_lock  (blocks dict update from prefetch thread)
    Background prefetch threads acquire ONLY _prefetch_lock.
    """

    def __init__(self, hardware: Any, block_size_mb: int = _DEFAULT_BLOCK_SIZE_MB) -> None:
        if block_size_mb < _MIN_BLOCK_SIZE_MB:
            logger.warning(
                "block_size_mb=%d below minimum %d, clamping",
                block_size_mb, _MIN_BLOCK_SIZE_MB,
            )
            block_size_mb = _MIN_BLOCK_SIZE_MB
        elif block_size_mb > _MAX_BLOCK_SIZE_MB:
            logger.warning(
                "block_size_mb=%d above maximum %d, clamping",
                block_size_mb, _MAX_BLOCK_SIZE_MB,
            )
            block_size_mb = _MAX_BLOCK_SIZE_MB

        self._block_size_bytes: int = block_size_mb * (1 << 20)
        self._hardware = hardware

        PoolCls = _get_memory_pool_class()
        self._pool: Any = PoolCls(hardware, block_size_mb=block_size_mb)
        self._heap: ReuseHeap = ReuseHeap()

        self._graph: Optional[ComputeGraph] = None
        self._current_op_idx: int = 0
        # FIX: Share the pipeline prefetch horizon with feasibility checks.
        self.prefetch_window: int = PIPELINE_DEPTH_STEPS

        self._blocks: dict[int, MemBlock] = {}

        self._eviction_count: int = 0
        self._prefetch_requests: int = 0
        self._prefetch_hits: int = 0
        self._throttle_count: int = 0

        # Primary lock + condition for allocation/eviction/throttle.
        self._lock = threading.RLock()
        self._space_available = threading.Condition(self._lock)

        # Separate lock for _blocks dict writes from the prefetch thread.
        # Background workers ONLY acquire this lock — never _lock — so the
        # Condition variable cycle is impossible.
        self._prefetch_lock = threading.Lock()

        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="zenyx-prefetch",
        )

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
        """Allocate *size_bytes* from *tier*, evicting + throttling if needed.

        Never raises OOM — worst case the calling thread sleeps until
        eviction frees space.

        Time:  O(log N) amortised.
        Space: O(1) beyond the allocation.
        """
        with self._lock:
            return self._allocate_inner(size_bytes, tier, block_id)

    def _allocate_inner(
        self,
        size_bytes: int,
        tier: MemTier,
        block_id: Optional[int],
    ) -> MemBlock:
        """Inner allocation loop — caller must hold ``_lock``."""
        retries = 0

        while True:
            try:
                block = self._pool.allocate(size_bytes, preferred_tier=tier)
                if block_id is not None:
                    block.block_id = block_id
                # Also update the prefetch-side view of _blocks.
                with self._prefetch_lock:
                    self._blocks[block.block_id] = block
                self._heap.update_access(block.block_id, self._current_op_idx)
                return block
            except Exception:
                pass

            evicted = self._try_evict(tier)
            if evicted:
                retries = 0
                continue

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

            # wait() atomically releases _lock and blocks, so other threads
            # (e.g. free()) can acquire _lock, free a block, and notify_all.
            self._space_available.wait(timeout=_THROTTLE_POLL_INTERVAL)

    def _try_evict(self, needed_tier: MemTier) -> bool:
        """Evict one block into a lower tier. Caller holds ``_lock``.

        Time:  O(log N)
        Space: O(1)
        """
        candidate_id = self._heap.get_eviction_candidate()
        if candidate_id is None:
            return False

        with self._prefetch_lock:
            candidate = self._blocks.get(candidate_id)

        if candidate is None:
            self._heap.remove_block(candidate_id)
            return False

        dest_tier = self._demote_tier(candidate.tier)
        if dest_tier is None:
            return False

        try:
            evicted_block = self._pool.evict(candidate, dest_tier)
            with self._prefetch_lock:
                self._blocks[candidate_id] = evicted_block
            self._eviction_count += 1
            logger.debug(
                "Evicted block %d: %s → %s",
                candidate_id, candidate.tier.name, dest_tier.name,
            )
            return True
        except Exception:
            logger.debug("Eviction of block %d failed", candidate_id, exc_info=True)
            return False

    @staticmethod
    def _demote_tier(current: MemTier) -> Optional[MemTier]:
        if current == MemTier.T0:
            return MemTier.T1
        if current == MemTier.T1:
            return MemTier.T2
        return None

    @staticmethod
    def _promote_tier(current: MemTier) -> Optional[MemTier]:
        if current == MemTier.T2:
            return MemTier.T1
        if current == MemTier.T1:
            return MemTier.T0
        return None

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self, block: MemBlock) -> None:
        """Release *block* back to the pool.

        Acquires _lock first, then _prefetch_lock (lock-ordering rule)
        so a concurrent prefetch write to _blocks cannot race this free.

        Time:  O(1)
        Space: O(1)
        """
        with self._lock:
            bid = block.block_id
            self._heap.remove_block(bid)
            with self._prefetch_lock:
                self._blocks.pop(bid, None)
            try:
                self._pool.free(block)
                self._space_available.notify_all()
            except Exception:
                logger.debug("Pool free for block %d failed", bid, exc_info=True)
            logger.debug("Freed block %d", bid)

    # ------------------------------------------------------------------
    # Compute graph registration
    # ------------------------------------------------------------------

    def register_compute_graph(self, graph: ComputeGraph) -> None:
        """Register (or update) the compute graph for Bélády-optimal eviction.

        First registration is synchronous; subsequent updates rebuild the
        heap asynchronously so in-flight allocations are not stalled.

        Time:  O(G + N log N) first call; O(1) subsequent.
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
        """Advance the allocator to *op_idx* and schedule prefetches.

        Time:  O(I + W·I + P·log N)
        Space: O(W·I)
        """
        with self._lock:
            self._current_op_idx = op_idx

            if self._graph is not None and op_idx < len(self._graph.ops):
                current_op = self._graph.ops[op_idx]
                for bid in current_op.input_blocks:
                    self._heap.update_access(bid, op_idx)

            needed = self._heap.blocks_needed_in_window(
                op_idx + 1, self.prefetch_window
            )
            if needed:
                self._dispatch_prefetch(needed)

    def _dispatch_prefetch(self, block_ids: list[int]) -> None:
        """Schedule async promotion of *block_ids* to T0.

        Caller holds ``_lock``; snapshot of to_prefetch is taken under
        ``_prefetch_lock`` so the worker never needs ``_lock``.

        Time:  O(P)
        Space: O(P)
        """
        to_prefetch: list[tuple[int, MemBlock]] = []

        # Snapshot under _prefetch_lock — worker will also use _prefetch_lock.
        with self._prefetch_lock:
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

        pool_ref = self._pool  # captured outside the lock

        def _prefetch_worker() -> None:
            """Background worker — acquires ONLY _prefetch_lock, never _lock.

            This is the key invariant that breaks the deadlock cycle:
            the main thread can hold _lock + block on _space_available,
            and this worker will never attempt to acquire _lock.
            """
            for bid, blk in to_prefetch:
                try:
                    promoted = pool_ref.prefetch(blk, MemTier.T0)
                    # Write result back under _prefetch_lock only.
                    with self._prefetch_lock:
                        self._blocks[bid] = promoted
                    logger.debug("Prefetched block %d: %s → T0", bid, blk.tier.name)
                except Exception:
                    logger.debug("Prefetch of block %d failed", bid, exc_info=True)

        self._prefetch_executor.submit(_prefetch_worker)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> AllocatorStats:
        """Return a live snapshot of allocator metrics.

        Time:  O(1)
        Space: O(1)
        """
        with self._lock:
            pool_usage = {}
            if hasattr(self._pool, "usage"):
                try:
                    pool_usage = self._pool.usage()
                except Exception:
                    pass

            t0 = pool_usage.get(MemTier.T0, (0, 0))
            t1 = pool_usage.get(MemTier.T1, (0, 0))
            t2 = pool_usage.get(MemTier.T2, (0, 0))

            return AllocatorStats(
                t0_used_bytes=t0[0],
                t0_total_bytes=t0[1],
                t1_used_bytes=t1[0],
                t1_total_bytes=t1[1],
                t2_used_bytes=t2[0],
                t2_total_bytes=t2[1],
                eviction_count=self._eviction_count,
                prefetch_requests=self._prefetch_requests,
                prefetch_hits=self._prefetch_hits,
                throttle_count=self._throttle_count,
            )

    def shutdown(self) -> None:
        """Shut down background prefetch thread pool."""
        self._prefetch_executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

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
