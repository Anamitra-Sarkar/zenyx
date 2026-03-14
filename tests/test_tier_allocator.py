"""Tests for TierAllocator — concurrency, stats, and lock isolation.

All tests are CPU-only and use mocked MemoryPool / ReuseHeap so no
CUDA device or NVMe is required.

Run with:
    pytest tests/test_tier_allocator.py -v
"""
from __future__ import annotations

import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from zenyx.core.hal.base import MemBlock, MemTier


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

def _make_block(tier: MemTier, size: int = 4 * 1024 * 1024) -> MemBlock:
    import torch
    t = torch.zeros(size // 2, dtype=torch.float16)
    return MemBlock(
        data=t,
        tier=tier,
        size_bytes=size,
        address=t.data_ptr(),
        dtype="float16",
        shape=tuple(t.shape),
    )


class _FakePool:
    """Minimal MemoryPool stub that satisfies TierAllocator's calls."""

    def __init__(self):
        self._allocated = {MemTier.T0: 0, MemTier.T1: 0, MemTier.T2: 0}
        self._cap = {MemTier.T0: 512 * 1024 * 1024,
                     MemTier.T1: 1024 * 1024 * 1024,
                     MemTier.T2: 4 * 1024 * 1024 * 1024}
        self._lock = threading.Lock()

    def allocate(self, size_bytes, preferred_tier=MemTier.T0):
        with self._lock:
            if self._allocated[preferred_tier] + size_bytes > self._cap[preferred_tier]:
                raise MemoryError("fake pool full")
            self._allocated[preferred_tier] += size_bytes
            return _make_block(preferred_tier, size_bytes)

    def free(self, block):
        with self._lock:
            self._allocated[block.tier] = max(
                0, self._allocated[block.tier] - block.size_bytes
            )

    def evict(self, block, dest_tier):
        with self._lock:
            self._allocated[block.tier] = max(
                0, self._allocated[block.tier] - block.size_bytes
            )
            self._allocated[dest_tier] += block.size_bytes
            new_block = _make_block(dest_tier, block.size_bytes)
            new_block.block_id = block.block_id
            return new_block

    def prefetch(self, block, target_tier):
        return block

    def usage(self):
        with self._lock:
            return {
                t: (self._allocated[t], self._cap[t])
                for t in (MemTier.T0, MemTier.T1, MemTier.T2)
            }


class _FakeHeap:
    """Minimal ReuseHeap stub."""

    def build_from_graph(self, graph): pass
    def rebuild_async(self, graph): pass
    def update_access(self, bid, op_idx): pass
    def remove_block(self, bid): pass
    def get_eviction_candidate(self): return None
    def blocks_needed_in_window(self, start, window): return []


class _FakeHardware:
    device_name = "FakeCPU"


def _make_allocator():
    """Create a TierAllocator with faked pool and heap."""
    from zenyx.core.allocator.tier_allocator import TierAllocator

    alloc = TierAllocator.__new__(TierAllocator)
    # Manually init so we bypass MemoryPool / ReuseHeap construction.
    import concurrent.futures
    alloc._block_size_bytes = 4 * 1024 * 1024
    alloc._hardware = _FakeHardware()
    alloc._pool = _FakePool()
    alloc._heap = _FakeHeap()
    alloc._graph = None
    alloc._current_op_idx = 0
    alloc.prefetch_window = 3
    alloc._blocks = {}
    alloc._eviction_count = 0
    alloc._prefetch_requests = 0
    alloc._prefetch_hits = 0
    alloc._throttle_count = 0
    alloc._lock = threading.RLock()
    alloc._space_available = threading.Condition(alloc._lock)
    alloc._prefetch_lock = threading.Lock()
    alloc._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="zenyx-prefetch-test"
    )
    return alloc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrefetchLockIsolation(unittest.TestCase):
    """Verify _prefetch_lock is a plain Lock, not the RLock/Condition."""

    def test_prefetch_lock_is_plain_lock(self):
        alloc = _make_allocator()
        # Must be a plain threading.Lock, not RLock or Condition
        self.assertIsInstance(alloc._prefetch_lock, type(threading.Lock()))
        self.assertIsNot(alloc._prefetch_lock, alloc._lock)
        self.assertIsNot(alloc._prefetch_lock, alloc._space_available)
        alloc.shutdown()


class TestConcurrentAllocateFree(unittest.TestCase):
    """Two threads allocating while a third frees — must not deadlock."""

    def test_no_deadlock_under_concurrent_alloc_free(self):
        alloc = _make_allocator()
        errors = []
        blocks_allocated = []
        lock = threading.Lock()

        SIZE = 4 * 1024 * 1024  # 4 MiB

        def alloc_worker():
            for _ in range(20):
                try:
                    blk = alloc.allocate(SIZE, MemTier.T0)
                    with lock:
                        blocks_allocated.append(blk)
                except Exception as e:
                    with lock:
                        errors.append(e)

        def free_worker():
            for _ in range(30):
                with lock:
                    if blocks_allocated:
                        blk = blocks_allocated.pop()
                    else:
                        blk = None
                if blk is not None:
                    try:
                        alloc.free(blk)
                    except Exception as e:
                        with lock:
                            errors.append(e)
                time.sleep(0.001)

        t1 = threading.Thread(target=alloc_worker)
        t2 = threading.Thread(target=alloc_worker)
        t3 = threading.Thread(target=free_worker)

        t1.start(); t2.start(); t3.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        t3.join(timeout=10)

        self.assertFalse(t1.is_alive(), "alloc thread 1 timed out (deadlock?)")
        self.assertFalse(t2.is_alive(), "alloc thread 2 timed out (deadlock?)")
        self.assertFalse(t3.is_alive(), "free thread timed out (deadlock?)")
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

        for blk in blocks_allocated:
            alloc.free(blk)
        alloc.shutdown()


class TestStatsNonZero(unittest.TestCase):
    """get_stats() must return non-zero used bytes after an allocation.

    Regression for the old getattr(self._pool, 't0_used_bytes', 0) bug
    that always returned 0.
    """

    def test_stats_reflect_allocation(self):
        alloc = _make_allocator()
        SIZE = 4 * 1024 * 1024
        blk = alloc.allocate(SIZE, MemTier.T0)
        stats = alloc.get_stats()
        self.assertGreater(
            stats.t0_used_bytes, 0,
            "t0_used_bytes should be > 0 after allocating a T0 block",
        )
        alloc.free(blk)
        alloc.shutdown()


class TestFreeDecrementsStats(unittest.TestCase):
    """Freeing a block must reduce usage back to zero."""

    def test_free_decrements_usage(self):
        alloc = _make_allocator()
        SIZE = 4 * 1024 * 1024
        blk = alloc.allocate(SIZE, MemTier.T0)
        alloc.free(blk)
        stats = alloc.get_stats()
        self.assertEqual(stats.t0_used_bytes, 0)
        alloc.shutdown()


if __name__ == "__main__":
    unittest.main()
