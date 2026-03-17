"""Zenyx MemoryPool — pre-pinned, three-tier, never-OOM memory pool.

This is the core of Zenyx's "Never-OOM" guarantee.  On initialisation,
the pool pre-allocates capacity across all three memory tiers (T0/T1/T2),
checks the formal feasibility condition, and provides block-level
allocation with automatic eviction to lower tiers when a higher tier is
full.

Block granularity: 2 MiB – 20 MiB contiguous chunks, aligned to 2 MiB
boundaries for NVMe page-size compatibility.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zenyx.core.hal.base import HALBase, MemBlock, MemTier, _human_bytes
from zenyx.core.hal.detector import HardwareInfo, build_hal_for_hardware
from zenyx.core.allocator.feasibility import (
    check_feasibility,
    compute_throughput_from_hardware,
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = logging.getLogger("zenyx.core.allocator.mem_pool")

# Block size constraints
MIN_BLOCK_SIZE = 2 * 1024 * 1024    # 2 MiB
MAX_BLOCK_SIZE = 20 * 1024 * 1024   # 20 MiB
BLOCK_ALIGNMENT = 2 * 1024 * 1024   # 2 MiB


def _align_up(n: int, alignment: int = BLOCK_ALIGNMENT) -> int:
    """Round *n* up to the nearest multiple of *alignment*.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    return ((n + alignment - 1) // alignment) * alignment


def _clamp_block_size(size: int) -> int:
    """Clamp and align *size* to the valid block range [MIN, MAX].

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    aligned = _align_up(max(size, MIN_BLOCK_SIZE))
    return min(aligned, MAX_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Tier capacity tracker
# ---------------------------------------------------------------------------


@dataclass
class _TierState:
    """Per-tier allocation bookkeeping.

    Attributes:
        capacity:  Total pre-allocated capacity in bytes.
        allocated: Currently allocated bytes.
        blocks:    Ordered dict of block_id → MemBlock (insertion order = LRU).

    Time complexity:  O(1) for all operations.
    Space complexity: O(B) where B = number of live blocks.
    """

    capacity: int = 0
    allocated: int = 0
    blocks: OrderedDict[str, MemBlock] = field(default_factory=OrderedDict)

    @property
    def free(self) -> int:
        """Available bytes on this tier."""
        return max(0, self.capacity - self.allocated)

    def __repr__(self) -> str:
        return (
            f"_TierState(allocated={_human_bytes(self.allocated)} / "
            f"{_human_bytes(self.capacity)}, blocks={len(self.blocks)})"
        )


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------


class MemoryPool:
    """Three-tier, pre-pinned memory pool with never-OOM guarantee.

    On construction:
      1. Runs :func:`check_feasibility` and logs a warning if the bandwidth
         condition is not met (throttle mode — never crashes).
      2. Pre-allocates capacity for T0 (90 % of available VRAM), T1 (pinned
         CPU), and T2 (mmap-backed NVMe).

    Allocation strategy:
      * Tries the preferred tier first.
      * If that tier is full, evicts the **least-recently-used** block to the
        next lower tier, then retries.
      * If T2 is full, **blocks** (throttles) until space is available.
      * **Never** raises an OOM exception.

    Args:
        hw_info:          Hardware description from :func:`detect_hardware`.
        block_size_mb:    Block granularity in MiB (clamped 2–20).
        t0_reserve_frac:  Fraction of per-device memory to reserve as T0 (default 0.9).
        t1_capacity:      T1 pool size in bytes (default 8 GiB).
        t2_capacity:      T2 pool size in bytes (default 64 GiB).
        compute_throughput: Compute throughput in **bytes/sec** for feasibility check.
                            If ``None``, derived from ``hw_info.compute_tflops`` via
                            :func:`compute_throughput_from_hardware` (correct units).

    Time complexity:  O(1) for construction.
    Space complexity: O(T0 + T1 + T2 capacities).
    """

    def __init__(
        self,
        hw_info: HardwareInfo,
        block_size_mb: int = 4,
        t0_reserve_frac: float = 0.90,
        t1_capacity: int = 8 * (1024 ** 3),
        t2_capacity: int = 64 * (1024 ** 3),
        compute_throughput: Optional[float] = None,
    ) -> None:
        self._hal: HALBase = build_hal_for_hardware(hw_info)
        self._hw_info = hw_info
        self._block_size_bytes = _clamp_block_size(block_size_mb * (1 << 20))
        self._lock = threading.RLock()
        self._evict_depth: int = 0
        self._copy_in_progress: int = 0
        self._throttle_event = threading.Event()
        self._throttle_event.set()  # not throttling initially

        # ---- tier state ---------------------------------------------------
        t0_cap = int(hw_info.per_device_memory_bytes * t0_reserve_frac)
        self._tiers: Dict[MemTier, _TierState] = {
            MemTier.T0: _TierState(capacity=t0_cap),
            MemTier.T1: _TierState(capacity=t1_capacity),
            MemTier.T2: _TierState(capacity=t2_capacity),
        }

        # ---- feasibility check --------------------------------------------
        # FIX: compute_throughput must be in bytes/sec, not raw TFLOPS.
        # Previously this used `hw_info.compute_tflops * 1e12 * 2.0` which
        # is FLOP/sec — wrong units.  Now we use compute_throughput_from_hardware()
        # which divides by arithmetic intensity (16 FLOP/byte for BF16 matmul)
        # to get the correct bytes/sec equivalent.
        if compute_throughput is None:
            compute_throughput = compute_throughput_from_hardware(
                hw_info.compute_tflops
            )

        self._feasibility = check_feasibility(
            bandwidth_t0_t1=hw_info.bandwidth_t0_t1,
            bandwidth_t1_t2=hw_info.bandwidth_t1_t2,
            compute_throughput=compute_throughput,
        )
        if not self._feasibility.is_feasible:
            logger.warning(
                "Feasibility check FAILED — %s", self._feasibility.message
            )
        else:
            logger.info("Feasibility check passed — OOM-free guarantee active")

        # ---- torch.cuda.MemPool for CUDA Graph address stability ----------
        self._cuda_pool: Optional[Any] = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                self._cuda_pool = torch.cuda.MemPool()
                logger.info(
                    "torch.cuda.MemPool created — CUDA Graph address stability guaranteed"
                )
            except (AttributeError, RuntimeError):
                logger.debug("torch.cuda.MemPool not available in this PyTorch version")

    # ------------------------------------------------------------------
    # allocate
    # ------------------------------------------------------------------

    def allocate(
        self,
        size_bytes: int,
        preferred_tier: MemTier = MemTier.T0,
    ) -> MemBlock:
        """Allocate *size_bytes* from the pool, starting at *preferred_tier*.

        If the preferred tier is full, the pool evicts the least-recently-used
        block to a lower tier and retries.  Falls through T0 → T1 → T2.
        If T2 is full, **blocks** until space is freed.

        All blocks are aligned to 2 MiB boundaries.

        Args:
            size_bytes:     Requested allocation size.
            preferred_tier: Tier to try first (default :attr:`MemTier.T0`).

        Returns:
            A :class:`MemBlock` residing on the best available tier.

        Time complexity:  O(1) amortised (may trigger one eviction chain).
        Space complexity: O(aligned_size).
        """
        aligned = _align_up(max(size_bytes, MIN_BLOCK_SIZE))

        with self._lock:
            for tier in _tier_fallback_chain(preferred_tier):
                state = self._tiers[tier]

                if tier == MemTier.T0 and self._copy_in_progress > 0:
                    logger.debug(
                        "Delaying T0 allocation reuse while %d eviction copies in progress",
                        self._copy_in_progress,
                    )
                elif state.free >= aligned:
                    return self._do_alloc(aligned, tier)

                if tier != MemTier.T2:
                    freed = self._evict_until(tier, aligned)
                    if freed and state.free >= aligned:
                        return self._do_alloc(aligned, tier)

            logger.warning(
                "All tiers full — throttling until %s freed on T2",
                _human_bytes(aligned),
            )

        self._throttle_wait(aligned)
        return self.allocate(size_bytes, MemTier.T2)

    def _do_alloc(self, size: int, tier: MemTier) -> MemBlock:
        """Perform the actual allocation via the HAL.

        For T0 allocations, uses ``torch.cuda.use_mem_pool`` when available
        to guarantee address stability for ``torch.compile`` CUDA Graph capture.

        Caller must hold ``self._lock``.

        Time complexity:  O(1).
        Space complexity: O(size).
        """
        if tier == MemTier.T0 and self._cuda_pool is not None:
            with torch.cuda.use_mem_pool(self._cuda_pool):
                block = self._hal.alloc(size, tier)
        else:
            block = self._hal.alloc(size, tier)
        state = self._tiers[tier]
        state.allocated += block.size_bytes
        state.blocks[block.block_id] = block
        return block

    # ------------------------------------------------------------------
    # evict
    # ------------------------------------------------------------------

    def evict(self, block: MemBlock, target_tier: MemTier) -> MemBlock:
        """Move *block* to *target_tier* (must be a lower / slower tier).

        Args:
            block:       Block to evict.
            target_tier: Destination tier (must have ``target_tier > block.tier``).

        Returns:
            New :class:`MemBlock` on *target_tier*.

        Time complexity:  O(size / bandwidth).
        Space complexity: O(size) — temporary duplication during copy.
        """
        if target_tier.value <= block.tier.value:
            logger.error(
                "evict: target_tier %s must be lower than current tier %s",
                target_tier,
                block.tier,
            )
            return block

        with self._lock:
            new_block = self._hal.alloc(block.size_bytes, target_tier)
            new_state = self._tiers[target_tier]
            new_state.allocated += new_block.size_bytes
            new_state.blocks[new_block.block_id] = new_block

        self._hal.copy(block, new_block)

        with self._lock:
            old_state = self._tiers[block.tier]
            old_state.blocks.pop(block.block_id, None)
            old_state.allocated = max(0, old_state.allocated - block.size_bytes)

        self._hal.free(block)

        logger.debug(
            "Evicted block %s: %s → %s (%s)",
            new_block.block_id,
            block.tier,
            target_tier,
            _human_bytes(block.size_bytes),
        )
        return new_block

    def _evict_until(self, tier: MemTier, needed: int) -> bool:
        """Evict LRU blocks from *tier* until *needed* bytes are free.

        Caller must hold ``self._lock``.

        Time complexity:  O(E) where E = number of blocks evicted.
        Space complexity: O(1) per eviction.
        """
        self._evict_depth += 1
        try:
            state = self._tiers[tier]
            target_tier = MemTier(tier.value + 1) if tier.value < 2 else None

            if target_tier is None:
                return False

            target_state = self._tiers[target_tier]

            blocks_to_copy: List[tuple[MemBlock, MemBlock]] = []

            while state.free < needed and state.blocks:
                oldest_id, oldest_block = next(iter(state.blocks.items()))

                if target_state.free < oldest_block.size_bytes:
                    if target_tier != MemTier.T2:
                        self._evict_until(target_tier, oldest_block.size_bytes)
                    if target_state.free < oldest_block.size_bytes:
                        break

                new_block = self._hal.alloc(oldest_block.size_bytes, target_tier)
                target_state.allocated += new_block.size_bytes
                target_state.blocks[new_block.block_id] = new_block

                state.blocks.pop(oldest_id, None)
                state.allocated = max(0, state.allocated - oldest_block.size_bytes)

                blocks_to_copy.append((oldest_block, new_block))

            result = state.free >= needed

            if blocks_to_copy and self._evict_depth == 1:
                # NOTE: bookkeeping removes source blocks before copy to avoid
                # holding the allocator lock over HAL I/O. To minimize reuse
                # races, T0 allocations are temporarily delayed while copies are
                # in flight (guarded by _copy_in_progress).
                assert self._evict_depth == 1
                self._copy_in_progress += 1
                self._lock.release()
                try:
                    for src, dst in blocks_to_copy:
                        logger.debug(
                            "LRU evict copy start: src_block_id=%s, %s → %s (%s)",
                            src.block_id,
                            tier,
                            target_tier,
                            _human_bytes(src.size_bytes),
                        )
                        self._hal.copy(src, dst)
                        self._hal.free(src)
                        logger.debug(
                            "LRU evict: %s → %s (%s)",
                            tier,
                            target_tier,
                            _human_bytes(src.size_bytes),
                        )
                finally:
                    self._lock.acquire()
                    self._copy_in_progress = max(0, self._copy_in_progress - 1)
            elif blocks_to_copy:
                for src, dst in blocks_to_copy:
                    self._hal.copy(src, dst)
                    self._hal.free(src)

            return result
        finally:
            self._evict_depth -= 1

    # ------------------------------------------------------------------
    # prefetch
    # ------------------------------------------------------------------

    def prefetch(self, block: MemBlock, target_tier: MemTier) -> MemBlock:
        """Move *block* to a higher (faster) tier.

        Args:
            block:       Block to promote.
            target_tier: Destination tier (must have ``target_tier < block.tier``).

        Returns:
            New :class:`MemBlock` on *target_tier*.

        Time complexity:  O(size / bandwidth).
        Space complexity: O(size) — temporary duplication during copy.
        """
        if target_tier.value >= block.tier.value:
            logger.error(
                "prefetch: target_tier %s must be higher than current tier %s",
                target_tier,
                block.tier,
            )
            return block

        with self._lock:
            state = self._tiers[target_tier]

            if state.free < block.size_bytes:
                self._evict_until(target_tier, block.size_bytes)

            if state.free < block.size_bytes:
                logger.warning(
                    "prefetch: cannot free enough space on %s — keeping on %s",
                    target_tier,
                    block.tier,
                )
                return block

            new_block = self._hal.alloc(block.size_bytes, target_tier)
            state.allocated += new_block.size_bytes
            state.blocks[new_block.block_id] = new_block

        self._hal.copy(block, new_block)

        with self._lock:
            old_state = self._tiers[block.tier]
            old_state.blocks.pop(block.block_id, None)
            old_state.allocated = max(0, old_state.allocated - block.size_bytes)

        self._hal.free(block)

        logger.debug(
            "Prefetched block %s: %s → %s (%s)",
            new_block.block_id,
            block.tier,
            target_tier,
            _human_bytes(block.size_bytes),
        )
        return new_block

    # ------------------------------------------------------------------
    # free
    # ------------------------------------------------------------------

    def free(self, block: MemBlock) -> None:
        """Release *block* back to its tier pool.

        Args:
            block: Previously allocated :class:`MemBlock`.

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        with self._lock:
            state = self._tiers[block.tier]
            state.blocks.pop(block.block_id, None)
            state.allocated = max(0, state.allocated - block.size_bytes)

        self._hal.free(block)
        self._throttle_event.set()

    # ------------------------------------------------------------------
    # throttle
    # ------------------------------------------------------------------

    def _throttle_wait(self, needed: int, timeout: float = 30.0) -> None:
        """Block until *needed* bytes are available on T2.

        This implements the "never crash, just slow" guarantee.
        Has a hard ceiling of ``timeout`` seconds — if nothing has freed
        memory after that, raises ``RuntimeError`` with a diagnostic
        message so the user knows to call ``allocator.free()``.

        Args:
            needed:  Bytes required.
            timeout: Maximum total wait in seconds before giving up.

        Raises:
            RuntimeError: If ``timeout`` elapses with no space freed.
                          This is NOT an OOM — it means the caller never
                          freed previously allocated blocks.

        Time complexity:  O(1) per wait cycle.
        Space complexity: O(1).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._throttle_event.clear()
            with self._lock:
                if self._tiers[MemTier.T2].free >= needed:
                    return
            self._throttle_event.wait(timeout=1.0)

        # Build diagnostic showing what is occupying each tier
        with self._lock:
            t0 = self._tiers[MemTier.T0]
            t1 = self._tiers[MemTier.T1]
            t2 = self._tiers[MemTier.T2]
            diag = (
                f"T0: {_human_bytes(t0.allocated)}/{_human_bytes(t0.capacity)} "
                f"({len(t0.blocks)} blocks), "
                f"T1: {_human_bytes(t1.allocated)}/{_human_bytes(t1.capacity)} "
                f"({len(t1.blocks)} blocks), "
                f"T2: {_human_bytes(t2.allocated)}/{_human_bytes(t2.capacity)} "
                f"({len(t2.blocks)} blocks)"
            )

        message = (
            f"Zenyx TierAllocator: all three memory tiers remain full after "
            f"{timeout:.0f}s throttle wait — {_human_bytes(needed)} needed on T2.\n"
            f"Tier usage: {diag}\n"
            f"Fix: call allocator.free(block) on blocks that are no longer "
            f"needed before allocating new ones. In a training loop, free "
            f"KV cache blocks at the end of each step."
        )
        logger.warning(message)
        raise RuntimeError(message)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def usage(self) -> Dict[MemTier, tuple[int, int]]:
        """Return ``{tier: (allocated, capacity)}`` snapshot.

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        with self._lock:
            return {
                tier: (state.allocated, state.capacity)
                for tier, state in self._tiers.items()
            }

    def get_cuda_pool(self) -> Optional[Any]:
        """Return the ``torch.cuda.MemPool`` for CUDA Graph capture.

        Returns:
            The ``torch.cuda.MemPool`` instance, or ``None`` if CUDA is
            not available.

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        return self._cuda_pool

    def __repr__(self) -> str:
        with self._lock:
            lines = [f"MemoryPool(feasible={self._feasibility.is_feasible})"]
            for tier in (MemTier.T0, MemTier.T1, MemTier.T2):
                state = self._tiers[tier]
                pct = (state.allocated / state.capacity * 100) if state.capacity > 0 else 0
                lines.append(
                    f"  {tier.name}: {_human_bytes(state.allocated)} / "
                    f"{_human_bytes(state.capacity)} ({pct:.1f}%) "
                    f"[{len(state.blocks)} blocks]"
                )
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tier_fallback_chain(preferred: MemTier) -> List[MemTier]:
    """Return tier fallback order starting from *preferred*."""
    tiers = [MemTier.T0, MemTier.T1, MemTier.T2]
    start = tiers.index(preferred)
    return tiers[start:]
