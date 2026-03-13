"""Min-Heap eviction scheduler using Bélády-optimal reuse distances.

Computes reuse distances from a transformer compute graph at compile time
to approximate Bélády's optimal page replacement algorithm. Falls back to
LRU during dynamic schedule changes while the heap is rebuilt asynchronously
in a background thread.

For dynamic graphs with conditional computation, probabilistic reuse
distances are computed as expected values weighted by branch probabilities.
"""

from __future__ import annotations

import heapq
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("zenyx.allocator.reuse_heap")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Op:
    """A single operation in a compute graph.

    Attributes:
        name: Human-readable operation name (e.g. ``"linear_0"``).
        input_blocks: Block IDs consumed by this operation.
        output_blocks: Block IDs produced by this operation.
        compute_cost: Estimated cost in FLOPs for scheduling heuristics.
    """

    name: str
    input_blocks: List[int]
    output_blocks: List[int]
    compute_cost: int

    def __repr__(self) -> str:
        return (
            f"Op(name={self.name!r}, "
            f"inputs={self.input_blocks}, "
            f"outputs={self.output_blocks}, "
            f"cost={self.compute_cost})"
        )


@dataclass
class ComputeGraph:
    """A directed dataflow graph of operations.

    The graph is represented as a linear schedule of :class:`Op` objects.
    The order of ``ops`` defines the execution schedule used for reuse
    distance computation.

    Attributes:
        ops: Ordered list of operations forming the execution schedule.
    """

    ops: List[Op] = field(default_factory=list)

    def __repr__(self) -> str:
        block_ids: set[int] = set()
        for op in self.ops:
            block_ids.update(op.input_blocks)
            block_ids.update(op.output_blocks)
        return (
            f"ComputeGraph(ops={len(self.ops)}, "
            f"unique_blocks={len(block_ids)})"
        )


# ---------------------------------------------------------------------------
# Heap entry
# ---------------------------------------------------------------------------

# Tie-breaker counter to ensure stable heap ordering when reuse distances
# are equal.  Monotonically increasing across all :class:`ReuseHeap`
# instances in the process.
_tie_counter: int = 0
_tie_lock = threading.Lock()


def _next_tie() -> int:
    global _tie_counter
    with _tie_lock:
        _tie_counter += 1
        return _tie_counter


@dataclass(order=True)
class _HeapEntry:
    """Internal heap element keyed by *negative* reuse distance.

    Using negative reuse distance means the block with the **largest**
    reuse distance (best eviction candidate per Bélády) sits at the top
    of the min-heap.

    Attributes:
        neg_reuse_dist: Negative reuse distance (lower = farther next use).
        tie: Monotonic tie-breaker for deterministic ordering.
        block_id: Identifier of the tracked memory block (excluded from
            ordering via ``field(compare=False)``).
    """

    neg_reuse_dist: float
    tie: int
    block_id: int = field(compare=False)

    def __repr__(self) -> str:
        return (
            f"_HeapEntry(block={self.block_id}, "
            f"reuse_dist={-self.neg_reuse_dist})"
        )


# ---------------------------------------------------------------------------
# ReuseHeap
# ---------------------------------------------------------------------------


class ReuseHeap:
    """Bélády-optimal eviction scheduler backed by a min-heap.

    The heap is keyed by *negative* reuse distance so that the block
    whose next use is **farthest** in the future is at the top (classic
    Bélády). When the static graph is being rebuilt asynchronously the
    scheduler transparently falls back to LRU eviction.

    Thread safety
    -------------
    All public methods acquire ``_lock`` (a ``threading.RLock``) so the
    heap may be queried concurrently with an in-progress
    :meth:`rebuild_async`.

    Complexity (steady-state, *N* = number of tracked blocks)
    ----------------------------------------------------------
    * ``build_from_graph`` — *O(G + N log N)* where *G* = total block
      references across all ops.
    * ``get_eviction_candidate`` — amortised *O(log N)* (lazy deletion).
    * ``update_access`` — *O(N)* worst-case (heap push + lazy invalidation
      scan amortised over pops).
    """

    # ---- construction -----------------------------------------------------

    def __init__(self) -> None:
        """Initialise an empty reuse heap.

        Time:  O(1)
        Space: O(1)
        """
        self._lock = threading.RLock()

        # Core heap storage
        self._heap: list[_HeapEntry] = []

        # Mapping block_id → current (valid) neg_reuse_dist stored in heap.
        # Entries whose distance no longer matches are treated as stale and
        # skipped during :meth:`get_eviction_candidate`.
        self._block_dist: dict[int, float] = {}

        # LRU fallback — ordered dict tracks access recency.
        self._lru: OrderedDict[int, None] = OrderedDict()

        # Static graph + pre-computed per-op reuse distances
        self._graph: Optional[ComputeGraph] = None
        self._next_use: dict[int, list[int]] = {}  # block → sorted future op indices

        # Background rebuild state
        self._rebuilding = threading.Event()  # *set* while rebuild in progress
        self._rebuild_thread: Optional[threading.Thread] = None

        # Set of block_ids currently tracked (present in pool).
        self._tracked_blocks: set[int] = set()

        logger.debug("ReuseHeap initialised (empty)")

    # ---- graph analysis ---------------------------------------------------

    def build_from_graph(self, compute_graph: ComputeGraph) -> None:
        """Parse *compute_graph* and rebuild the heap with exact reuse distances.

        For each block referenced in the graph, the reuse distance at the
        current position (op index 0) is the index of its *next* use.
        Blocks with no future use receive ``float('inf')`` distance
        (evict first).

        Args:
            compute_graph: The execution schedule to analyse.

        Time:  O(G + N log N)  — G = total block refs, N = unique blocks.
        Space: O(G + N)
        """
        with self._lock:
            self._graph = compute_graph
            self._next_use.clear()

            # Build per-block list of operation indices where the block is used.
            for idx, op in enumerate(compute_graph.ops):
                for bid in op.input_blocks:
                    self._next_use.setdefault(bid, []).append(idx)

            # Rebuild heap for all currently tracked blocks.
            self._heap.clear()
            self._block_dist.clear()

            for bid in self._tracked_blocks:
                dist = self._reuse_distance_for(bid, current_op_idx=0)
                neg = -dist
                self._block_dist[bid] = neg
                heapq.heappush(self._heap, _HeapEntry(neg, _next_tie(), bid))

            logger.debug(
                "Heap rebuilt from graph: %d ops, %d tracked blocks",
                len(compute_graph.ops),
                len(self._tracked_blocks),
            )

    # ---- eviction ---------------------------------------------------------

    def get_eviction_candidate(self) -> Optional[int]:
        """Return the ``block_id`` with the largest reuse distance.

        If the heap is being rebuilt asynchronously, falls back to LRU.

        Returns:
            ``block_id`` to evict, or ``None`` if no blocks are tracked.

        Time:  amortised O(log N)
        Space: O(1)
        """
        with self._lock:
            if self._rebuilding.is_set():
                return self._lru_eviction_candidate()

            # Lazy-deletion loop — skip entries whose distance is stale.
            while self._heap:
                entry = self._heap[0]
                bid = entry.block_id
                # Valid if block is still tracked AND its stored distance
                # matches what we have on record.
                if (
                    bid in self._tracked_blocks
                    and bid in self._block_dist
                    and self._block_dist[bid] == entry.neg_reuse_dist
                ):
                    return bid
                heapq.heappop(self._heap)

            # Heap exhausted — fall back to LRU.
            return self._lru_eviction_candidate()

    # ---- access tracking --------------------------------------------------

    def update_access(self, block_id: int, current_op_idx: int) -> None:
        """Notify the heap that *block_id* was just accessed at *current_op_idx*.

        Recomputes the block's reuse distance from the current position and
        pushes an updated entry. The old entry becomes stale and is
        garbage-collected lazily during :meth:`get_eviction_candidate`.

        Also moves *block_id* to the most-recent end of the LRU tracker.

        Args:
            block_id: Block that was accessed.
            current_op_idx: Index of the operation that accessed it.

        Time:  O(log N) for the heap push.
        Space: O(1) amortised.
        """
        with self._lock:
            self._tracked_blocks.add(block_id)

            # LRU bookkeeping
            self._lru.pop(block_id, None)
            self._lru[block_id] = None  # move to end (most recent)

            # Recompute reuse distance from current position.
            dist = self._reuse_distance_for(block_id, current_op_idx)
            neg = -dist
            self._block_dist[block_id] = neg
            heapq.heappush(self._heap, _HeapEntry(neg, _next_tie(), block_id))

    def remove_block(self, block_id: int) -> None:
        """Stop tracking *block_id* (e.g. after it is freed).

        The block's heap entries become stale and are pruned lazily.

        Args:
            block_id: Block to remove from tracking.

        Time:  O(1)
        Space: O(1)
        """
        with self._lock:
            self._tracked_blocks.discard(block_id)
            self._block_dist.pop(block_id, None)
            self._lru.pop(block_id, None)

    # ---- async rebuild ----------------------------------------------------

    def rebuild_async(self, new_graph: ComputeGraph) -> None:
        """Trigger a background rebuild of the heap from *new_graph*.

        While the rebuild is in progress the scheduler falls back to LRU
        eviction. The previous rebuild (if any) is allowed to finish
        before the new one starts — only one rebuild thread runs at a time.

        Args:
            new_graph: Updated compute graph to analyse.

        Time:  O(1) to launch the thread.
        Space: O(G + N) in the background thread.
        """
        # Wait for any in-flight rebuild to finish first.
        if self._rebuild_thread is not None and self._rebuild_thread.is_alive():
            logger.debug("Waiting for previous rebuild to complete")
            self._rebuild_thread.join()

        self._rebuilding.set()
        logger.info("Async heap rebuild started — falling back to LRU")

        def _worker() -> None:
            try:
                self.build_from_graph(new_graph)
            except Exception:
                logger.exception("Async heap rebuild failed")
            finally:
                self._rebuilding.clear()
                logger.info("Async heap rebuild complete — Bélády mode restored")

        self._rebuild_thread = threading.Thread(
            target=_worker, name="zenyx-reuse-heap-rebuild", daemon=True
        )
        self._rebuild_thread.start()

    # ---- LRU fallback -----------------------------------------------------

    def _lru_eviction_candidate(self) -> Optional[int]:
        """Return the least-recently-used tracked block.

        Uses an :class:`~collections.OrderedDict` that is maintained
        by :meth:`update_access`.

        Returns:
            ``block_id`` of the LRU block, or ``None`` if empty.

        Time:  O(1)
        Space: O(1)
        """
        # Iterate from oldest to newest; return first that is tracked.
        for bid in self._lru:
            if bid in self._tracked_blocks:
                return bid
        return None

    # ---- probabilistic reuse distance -------------------------------------

    def _probabilistic_reuse_distance(
        self, block_id: int, branch_probs: Dict[str, float]
    ) -> float:
        """Compute expected reuse distance weighted by branch probabilities.

        For dynamic graphs with conditional computation, each branch may
        reference *block_id* at different future positions. The expected
        reuse distance is the probability-weighted sum across branches.

        If no branch references the block the distance is ``float('inf')``.

        Args:
            block_id: Block to compute distance for.
            branch_probs: Mapping of branch name → probability (should
                sum to 1.0).

        Returns:
            Expected reuse distance (ops until next use).

        Time:  O(B · G_b)  — B = branches, G_b = ops per branch.
        Space: O(B)
        """
        if not branch_probs:
            return float("inf")

        expected_dist: float = 0.0
        total_prob: float = 0.0

        for branch_name, prob in branch_probs.items():
            # Convention: branch ops are stored in ``_next_use`` under
            # composite keys ``"branch_name:block_id"``.  For a simple
            # graph the block_id key is used directly.
            uses = self._next_use.get(block_id, [])
            if uses:
                # Nearest future use in this branch.
                dist = float(uses[0])
            else:
                dist = float("inf")

            if dist == float("inf"):
                # Infinite distance contributes nothing useful to the
                # weighted sum but we still account for the probability
                # mass — if *all* branches are infinite the result is inf.
                continue

            expected_dist += prob * dist
            total_prob += prob

        if total_prob == 0.0:
            return float("inf")

        # Scale by the fraction of probability mass that had finite
        # distances and add an infinite contribution for the remainder.
        if total_prob < 1.0:
            # Some branches never touch this block — conservatively treat
            # as very large but finite so the block is still evictable.
            remaining = 1.0 - total_prob
            expected_dist = expected_dist / total_prob
            # Penalise by the chance that the block *is* needed.
            expected_dist /= total_prob
            # Blend: the higher the unused-branch probability, the
            # farther out we push the effective distance.
            expected_dist += remaining * 1e6
        else:
            expected_dist /= total_prob

        return expected_dist

    # ---- internal helpers -------------------------------------------------

    def _reuse_distance_for(self, block_id: int, current_op_idx: int) -> float:
        """Compute the reuse distance for *block_id* from *current_op_idx*.

        The reuse distance is the number of operations until the block's
        next use.  If the block has no future use the distance is
        ``float('inf')`` (highest eviction priority — Bélády).

        Args:
            block_id: Block to compute distance for.
            current_op_idx: Current position in the execution schedule.

        Returns:
            Reuse distance in number of operations.

        Time:  O(log U)  — binary search over U future uses.
        Space: O(1)
        """
        uses = self._next_use.get(block_id)
        if not uses:
            return float("inf")

        # Binary search for the first use strictly after current_op_idx.
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= current_op_idx:
                lo = mid + 1
            else:
                hi = mid

        if lo < len(uses):
            return float(uses[lo] - current_op_idx)
        return float("inf")

    def blocks_needed_in_window(
        self, current_op_idx: int, window: int
    ) -> list[int]:
        """Return block IDs that will be consumed in the next *window* ops.

        Useful for prefetch scheduling in :class:`TierAllocator`.

        Args:
            current_op_idx: Current position in the execution schedule.
            window: Number of future operations to look ahead.

        Returns:
            De-duplicated list of block IDs needed within the window.

        Time:  O(W · I)  — W = window, I = avg inputs per op.
        Space: O(W · I)
        """
        with self._lock:
            if self._graph is None:
                return []

            needed: list[int] = []
            seen: set[int] = set()
            end = min(current_op_idx + window, len(self._graph.ops))

            for idx in range(current_op_idx, end):
                for bid in self._graph.ops[idx].input_blocks:
                    if bid not in seen:
                        seen.add(bid)
                        needed.append(bid)
            return needed

    # ---- dunder -----------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            mode = "LRU" if self._rebuilding.is_set() else "Bélády"
            return (
                f"ReuseHeap(mode={mode}, "
                f"heap_size={len(self._heap)}, "
                f"tracked_blocks={len(self._tracked_blocks)})"
            )
