"""Zenyx HAL (Hardware Abstraction Layer) — abstract base class and shared types.

This module defines the five HAL primitives (alloc, free, copy, matmul, reduce),
the three-tier memory model (T0/T1/T2), and supporting data structures that every
backend must implement.

Typical usage::

    class MyBackend(HALBase):
        def alloc(self, size_bytes, tier): ...
        ...

"""
from __future__ import annotations

import abc
import enum
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

logger = logging.getLogger("zenyx.core.hal.base")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemTier(enum.IntEnum):
    """Three-tier memory hierarchy.

    * **T0** — Device HBM / VRAM (fastest, smallest).
    * **T1** — CPU DRAM, pinned (medium speed, medium capacity).
    * **T2** — NVMe SSD, memory-mapped (slowest, largest).
    """

    T0 = 0  # HBM / VRAM
    T1 = 1  # CPU DRAM (pinned)
    T2 = 2  # NVMe SSD (mmap)

    def __repr__(self) -> str:
        return f"MemTier.{self.name}"


class ReduceOp(enum.Enum):
    """Supported collective-reduce operations."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"
    AVG = "avg"

    def __repr__(self) -> str:
        return f"ReduceOp.{self.name}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemBlock:
    """Handle to a contiguous memory block managed by the HAL.

    Attributes:
        block_id:   Unique identifier for this block.
        data:       Underlying storage object (torch.Tensor, mmap buffer, etc.).
        tier:       Which memory tier this block resides in.
        size_bytes: Allocated size in bytes.
        address:    Virtual or physical address (backend-specific).
        dtype:      Element data-type string (e.g. ``"float16"``).
        shape:      Tensor shape, or ``()`` for raw buffers.

    Time complexity:  O(1) for creation.
    Space complexity: O(size_bytes) — managed externally by the allocator.
    """

    data: Any
    tier: MemTier
    size_bytes: int
    address: int = 0
    dtype: str = "float16"
    shape: Tuple[int, ...] = ()
    block_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __repr__(self) -> str:
        size_human = _human_bytes(self.size_bytes)
        return (
            f"MemBlock(id={self.block_id!r}, tier={self.tier!r}, "
            f"size={size_human}, dtype={self.dtype!r}, shape={self.shape})"
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class HALBase(abc.ABC):
    """Abstract base class for all Zenyx hardware backends.

    Every concrete backend (CUDA, ROCm, XLA, CPU, Metal) must implement the
    five primitives below.  Backends are **stateful** — they own their memory
    pools and stream handles.

    The five HAL primitives
    ~~~~~~~~~~~~~~~~~~~~~~
    1. **alloc**   — allocate a contiguous memory block on a given tier.
    2. **free**    — release a previously allocated block.
    3. **copy**    — asynchronously transfer data between tiers.
    4. **matmul**  — matrix multiplication (delegates to vendor BLAS).
    5. **reduce**  — collective all-reduce across a device group.
    """

    # ---- allocation -------------------------------------------------------

    @abc.abstractmethod
    def alloc(self, size_bytes: int, tier: MemTier) -> MemBlock:
        """Allocate *size_bytes* of contiguous memory on *tier*.

        Args:
            size_bytes: Requested allocation size (will be rounded up to the
                        backend's alignment requirement).
            tier:       Target memory tier.

        Returns:
            A :class:`MemBlock` handle.

        Raises:
            Never raises OOM — implementations must evict or throttle instead.

        Time complexity:  O(1) amortised (pool allocation).
        Space complexity: O(size_bytes).
        """
        ...

    @abc.abstractmethod
    def free(self, block: MemBlock) -> None:
        """Release *block* and return its memory to the tier pool.

        Args:
            block: Previously allocated :class:`MemBlock`.

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        ...

    # ---- data movement ----------------------------------------------------

    @abc.abstractmethod
    def copy(
        self,
        src: MemBlock,
        dst: MemBlock,
        stream: Optional[Any] = None,
    ) -> None:
        """Asynchronously copy data from *src* to *dst*.

        The copy may span tiers (e.g. T0 → T1).  If *stream* is provided, the
        operation is enqueued on that stream/queue; otherwise a default stream
        is used.

        Args:
            src:    Source block.
            dst:    Destination block (must be ≥ src.size_bytes).
            stream: Optional backend-specific stream handle.

        Time complexity:  O(size / bandwidth).
        Space complexity: O(1) (in-place copy).
        """
        ...

    # ---- compute ----------------------------------------------------------

    @abc.abstractmethod
    def matmul(
        self,
        a: MemBlock,
        b: MemBlock,
        out: Optional[MemBlock] = None,
    ) -> MemBlock:
        """Matrix multiplication *a @ b*.

        If *out* is ``None``, a new block is allocated for the result.
        Delegates to vendor BLAS (cuBLAS, hipBLAS, …).

        Args:
            a:   Left operand (2-D tensor in a MemBlock).
            b:   Right operand (2-D tensor in a MemBlock).
            out: Optional pre-allocated output block.

        Returns:
            MemBlock containing the result.

        Time complexity:  O(M × N × K) — standard matrix-multiply.
        Space complexity: O(M × N) for the output.
        """
        ...

    @abc.abstractmethod
    def reduce(
        self,
        tensor: MemBlock,
        op: ReduceOp,
        group: Optional[Any] = None,
    ) -> MemBlock:
        """Collective all-reduce of *tensor* using operation *op*.

        When *group* is ``None`` the default process group is used.

        Args:
            tensor: Input block.
            op:     Reduction operation (:class:`ReduceOp`).
            group:  Optional communication group handle.

        Returns:
            MemBlock with the reduced result (may be in-place).

        Time complexity:  O(size × log(world_size)) — ring all-reduce.
        Space complexity: O(size) for the result.
        """
        ...

    # ---- introspection ----------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    """Return a human-readable byte string (e.g. ``'1.23 GiB'``).

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n:.2f} PiB"
