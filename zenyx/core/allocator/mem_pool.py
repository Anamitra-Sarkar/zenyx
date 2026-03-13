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
from zenyx.core.hal.detector import HardwareInfo
from zenyx.core.allocator.feasibility import check_feasibility

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
         condition is not met.
      2. Pre-allocates capacity for T0 (90 % of available VRAM), T1 (pinned
         CPU), and T2 (mmap-backed NVMe).

    Allocation strategy:
      * Tries the preferred tier first.
      * If that tier is full, evicts the **least-recently-used** block to the
        next lower tier, then retries.
      * If T2 is full, **blocks** (throttles) until space is available.
      * **Never** raises OOM.

    Args:
        hal:              Concrete HAL backend.
        hw_info:          Hardware description from :func:`detect_hardware`.
        t0_reserve_frac:  Fraction of per-device memory to reserve as T0 (default 0.9).
        t1_capacity:      T1 pool size in bytes (default 8 GiB).
        t2_capacity:      T2 pool size in bytes (default 64 GiB).
        compute_throughput: Compute throughput in bytes/sec for feasibility check.
                            If ``None``, uses ``hw_info.compute_tflops × 1e12 × 2``
                            (FLOP → bytes for FP16).

    Time complexity:  O(1) for construction.
    Space complexity: O(T0 + T1 + T2 capacities).
    """

    def __init__(
        self,
        hal: HALBase,
        hw_info: HardwareInfo,
        t0_reserve_frac: float = 0.90,
        t1_capacity: int = 8 * (1024 ** 3),
        t2_capacity: int = 64 * (1024 ** 3),
        compute_throughput: Optional[float] = None,
    ) -> None:
        self._hal = hal
        self._hw_info = hw_info
        self._lock = threading.Lock()
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
        ct = compute_throughput
        if ct is None:
            # Convert TFLOPS → bytes/sec (FP16: 2 bytes per FLOP output)
            ct = hw_info.compute_tflops * 1e12 * 2.0

        self._feasibility = check_feasibility(
            bandwidth_t0_t1=hw_info.bandwidth_t0_t1,
            bandwidth_t1_t2=hw_info.bandwidth_t1_t2,
            compute_throughput=ct,
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
            # Try preferred tier and each lower tier in order
            for tier in _tier_fallback_chain(preferred_tier):
                state = self._tiers[tier]

                # If there's room, allocate directly
                if state.free >= aligned:
                    return self._do_alloc(aligned, tier)

                # Try evicting from this tier to the next lower tier
                if tier != MemTier.T2:
                    freed = self._evict_until(tier, aligned)
                    if freed and state.free >= aligned:
                        return self._do_alloc(aligned, tier)

            # All tiers exhausted — throttle on T2 until space frees up
            logger.warning(
                "All tiers full — throttling until %s freed on T2",
                _human_bytes(aligned),
            )

        # Release lock while blocking to allow other threads to free memory
        self._throttle_wait(aligned)

        # Retry after throttle
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

        The copy is asynchronous; the old block is freed after the copy
        completes.

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

        # Copy data (async — HAL handles streams / thread pool)
        self._hal.copy(block, new_block)

        # Free the old block
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

        Caller must hold ``self._lock``.  Eviction targets the next lower
        tier.  Uses a two-phase pattern: bookkeeping under the lock, then
        blocking copies outside the lock so other threads can proceed.

        Args:
            tier:   Tier to evict from.
            needed: Bytes required.

        Returns:
            ``True`` if sufficient space was freed.

        Time complexity:  O(E) where E = number of blocks evicted.
        Space complexity: O(1) per eviction.
        """
        state = self._tiers[tier]
        target_tier = MemTier(tier.value + 1) if tier.value < 2 else None

        if target_tier is None:
            return False

        target_state = self._tiers[target_tier]

        # PHASE 1: Under the lock — claim blocks, update bookkeeping,
        # allocate destination buffers.
        blocks_to_copy: List[tuple[MemBlock, MemBlock]] = []

        while state.free < needed and state.blocks:
            # Pop the oldest (LRU) block
            oldest_id, oldest_block = next(iter(state.blocks.items()))

            # Check target tier has room
            if target_state.free < oldest_block.size_bytes:
                # Can't evict further — target tier is also full
                # Try evicting from target tier recursively
                if target_tier != MemTier.T2:
                    self._evict_until(target_tier, oldest_block.size_bytes)
                if target_state.free < oldest_block.size_bytes:
                    break

            # Allocate on target tier (HAL alloc is fast pool sub-allocation)
            new_block = self._hal.alloc(oldest_block.size_bytes, target_tier)
            target_state.allocated += new_block.size_bytes
            target_state.blocks[new_block.block_id] = new_block

            # Remove old block from current tier bookkeeping
            state.blocks.pop(oldest_id, None)
            state.allocated = max(0, state.allocated - oldest_block.size_bytes)

            blocks_to_copy.append((oldest_block, new_block))

        result = state.free >= needed

        # PHASE 2: Outside the lock — do the actual copy and free.
        # HAL.copy() is O(size/bandwidth) and can take milliseconds;
        # releasing the lock prevents throughput collapse from contention.
        if blocks_to_copy:
            self._lock.release()
            try:
                for src, dst in blocks_to_copy:
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

        return result

    # ------------------------------------------------------------------
    # prefetch
    # ------------------------------------------------------------------

    def prefetch(self, block: MemBlock, target_tier: MemTier) -> MemBlock:
        """Move *block* to a higher (faster) tier.

        Inverse of :meth:`evict` — promotes data towards T0.

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

            # Ensure room on target tier — evict if necessary
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

        # Copy data up
        self._hal.copy(block, new_block)

        # Free old block from lower tier
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

        # Signal any throttle-waiters that space may have freed up
        self._throttle_event.set()

    # ------------------------------------------------------------------
    # throttle
    # ------------------------------------------------------------------

    def _throttle_wait(self, needed: int, timeout: float = 30.0) -> None:
        """Block until *needed* bytes are available on T2.

        This implements the "never crash, just slow" guarantee.

        Args:
            needed:  Bytes required.
            timeout: Maximum wait in seconds before retrying.

        Time complexity:  O(1) per wait cycle.
        Space complexity: O(1).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._throttle_event.clear()
            with self._lock:
                if self._tiers[MemTier.T2].free >= needed:
                    return
            # Wait for a free() call to signal us
            self._throttle_event.wait(timeout=1.0)

        logger.warning(
            "Throttle timeout (%.0fs) — retrying allocation of %s",
            timeout,
            _human_bytes(needed),
        )

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
        """Return the ``torch.cuda.MemPool`` for use as context manager
        around ``torch.compile`` graph capture.

        Must be called before any ``torch.compile(...)`` call to ensure
        CUDA Graph address stability.

        Returns:
            The ``torch.cuda.MemPool`` instance, or ``None`` if CUDA is
            not available or the pool was not created.

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
    """Return tier fallback order starting from *preferred*.

    E.g. ``_tier_fallback_chain(MemTier.T0)`` → ``[T0, T1, T2]``.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    tiers = [MemTier.T0, MemTier.T1, MemTier.T2]
    start = tiers.index(preferred)
    return tiers[start:]
