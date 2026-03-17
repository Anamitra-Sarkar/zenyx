"""Zenyx HAL — ROCm / HIP backend implementation.

ROCm shares the CUDA programming model via HIP, so this backend re-uses
much of the CUDA path with ROCm-specific adjustments:

* Uses ``hipBLASLt`` (via ``torch.matmul`` on ROCm builds) for GEMM.
* Uses ``MIOpen`` for fused attention (when available).
* Warns about the **37-45 % MFU** ceiling compared to H100's 45-55 %
  (kernel-gap documented in Zenyx arch spec).
* Collective reduce uses ``RCCL`` (ROCm's NCCL fork) through
  ``torch.distributed``.

Typical usage::

    from zenyx.core.hal.rocm_hal import RocmHAL
    hal = RocmHAL(device_index=0)
    blk = hal.alloc(1 << 20, MemTier.T0)
"""
from __future__ import annotations

import logging
import mmap
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import torch

from zenyx.core.hal.base import HALBase, MemBlock, MemTier, ReduceOp, _human_bytes

logger = logging.getLogger("zenyx.core.hal.rocm_hal")

# ── ROCm detection ────────────────────────────────────────────────────────

_ROCM_AVAILABLE: bool = False
try:
    if torch.cuda.is_available() and "rocm" in (torch.version.hip or ""):
        _ROCM_AVAILABLE = True
except Exception:
    pass

_REDUCE_OP_MAP = {
    ReduceOp.SUM: torch.distributed.ReduceOp.SUM if torch.distributed.is_available() else None,
    ReduceOp.MAX: torch.distributed.ReduceOp.MAX if torch.distributed.is_available() else None,
    ReduceOp.MIN: torch.distributed.ReduceOp.MIN if torch.distributed.is_available() else None,
}


class RocmHAL(HALBase):
    """ROCm/HIP backend implementing the five HAL primitives.

    Mirrors :class:`CudaHAL` logic but emits a performance warning at init
    and uses ROCm-appropriate defaults.

    Parameters
    ----------
    device_index : int
        HIP device ordinal (default ``0``).
    t1_capacity_bytes : int | None
        Pinned CPU pool size.  ``None`` → auto (50 % of available RAM,
        capped at 64 GiB).
    t2_dir : str | None
        Directory for NVMe-backed mmap files.  ``None`` → system temp.

    Time complexity:  O(1) for init (pool pre-allocation is amortised).
    Space complexity: O(t1_capacity_bytes + t2 file size).
    """

    # ── construction ──────────────────────────────────────────────────

    def __init__(
        self,
        device_index: int = 0,
        t1_capacity_bytes: int | None = None,
        t2_dir: str | None = None,
    ) -> None:
        if not _ROCM_AVAILABLE:
            raise RuntimeError(
                "ROCm/HIP not available — torch.version.hip is unset or CUDA unavailable. "
                "Install a ROCm-enabled PyTorch build."
            )

        self._device = torch.device("cuda", device_index)
        torch.cuda.set_device(self._device)

        logger.warning(
            "ROCm detected: expect 37-45%% MFU vs H100's 45-55%% due to kernel gap. "
            "Zenyx will use hipBLASLt for GEMM and RCCL for collectives."
        )

        # T0 — HBM via HIP allocator (normal torch.cuda path on ROCm)
        self._t0_total = torch.cuda.get_device_properties(self._device).total_mem
        self._t0_used: int = 0

        # T1 — pinned CPU
        if t1_capacity_bytes is None:
            try:
                import psutil
                t1_capacity_bytes = min(psutil.virtual_memory().available // 2, 64 * (1 << 30))
            except ImportError:
                t1_capacity_bytes = 16 * (1 << 30)
        self._t1_capacity = t1_capacity_bytes
        self._t1_used: int = 0

        # T2 — NVMe mmap
        self._t2_dir = t2_dir or tempfile.gettempdir()
        self._t2_used: int = 0
        self._t2_files: Dict[str, Tuple[mmap.mmap, str]] = {}

        # Async I/O executor for T2
        self._io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="zenyx-rocm-io")

        # Block registry
        self._blocks: Dict[str, MemBlock] = {}

        logger.info(
            "RocmHAL initialised — device=%s  T0=%.2f GiB  T1=%.2f GiB",
            self._device,
            self._t0_total / (1 << 30),
            self._t1_capacity / (1 << 30),
        )

    # ── alloc ─────────────────────────────────────────────────────────

    def alloc(self, size_bytes: int, tier: MemTier) -> MemBlock:
        """Allocate *size_bytes* on *tier*.

        Time complexity:  O(1) amortised.
        Space complexity: O(size_bytes).
        """
        if tier == MemTier.T0:
            return self._alloc_t0(size_bytes)
        elif tier == MemTier.T1:
            return self._alloc_t1(size_bytes)
        else:
            return self._alloc_t2(size_bytes)

    def _alloc_t0(self, size_bytes: int) -> MemBlock:
        """Allocate from HIP device memory (T0 / HBM).

        Raises ``RuntimeError`` on OOM rather than silently falling back
        to T1.  The caller (MemoryPool eviction loop) is responsible for
        evicting blocks and retrying, matching the CudaHAL contract.
        """
        try:
            t = torch.empty(size_bytes, dtype=torch.uint8, device=self._device)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"RocmHAL: T0 OOM — could not allocate {_human_bytes(size_bytes)} on "
                f"{self._device}. Evict blocks from T0 before retrying."
            ) from exc
        self._t0_used += size_bytes
        blk = MemBlock(data=t, tier=MemTier.T0, size_bytes=size_bytes,
                        address=t.data_ptr(), dtype="uint8", shape=t.shape)
        self._blocks[blk.block_id] = blk
        return blk

    def _alloc_t1(self, size_bytes: int) -> MemBlock:
        """Allocate pinned CPU memory (T1).

        pin_memory requires a CUDA/ROCm runtime. Guard with
        ``torch.cuda.is_available()`` to avoid
        ``RuntimeError: Cannot pin memory without CUDA``
        on systems where ROCm is not active (e.g. CPU-only CI).
        """
        if self._t1_used + size_bytes > self._t1_capacity:
            logger.warning("RocmHAL: T1 capacity exceeded — falling back to T2")
            return self._alloc_t2(size_bytes)
        _pin = torch.cuda.is_available()
        t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=_pin)
        self._t1_used += size_bytes
        blk = MemBlock(data=t, tier=MemTier.T1, size_bytes=size_bytes,
                        address=t.data_ptr(), dtype="uint8", shape=t.shape)
        self._blocks[blk.block_id] = blk
        return blk

    def _alloc_t2(self, size_bytes: int) -> MemBlock:
        path = os.path.join(self._t2_dir, f"zenyx_rocm_{os.getpid()}_{id(self)}_{len(self._t2_files)}.mmap")
        with open(path, "wb") as f:
            f.seek(size_bytes - 1)
            f.write(b"\x00")
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, size_bytes)
        os.close(fd)
        self._t2_used += size_bytes
        blk = MemBlock(data=mm, tier=MemTier.T2, size_bytes=size_bytes,
                        address=0, dtype="uint8", shape=(size_bytes,))
        self._t2_files[blk.block_id] = (mm, path)
        self._blocks[blk.block_id] = blk
        return blk

    # ── free ──────────────────────────────────────────────────────────

    def free(self, block: MemBlock) -> None:
        """Release *block* and return memory to the pool.

        Uses ``block.data = None`` rather than ``del block.data`` so that
        the tensor's Python refcount is decremented correctly.  ``del``
        removes the attribute but does not guarantee the refcount reaches
        zero if other references exist (e.g. a live HIP stream op);
        assigning ``None`` is the correct idiom for nulling a dataclass slot.

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        bid = block.block_id
        if bid not in self._blocks:
            logger.warning("RocmHAL.free: block %s not found", bid)
            return
        if block.tier == MemTier.T0:
            self._t0_used = max(0, self._t0_used - block.size_bytes)
            block.data = None
        elif block.tier == MemTier.T1:
            self._t1_used = max(0, self._t1_used - block.size_bytes)
            block.data = None
        elif block.tier == MemTier.T2:
            self._t2_used = max(0, self._t2_used - block.size_bytes)
            if bid in self._t2_files:
                mm, path = self._t2_files.pop(bid)
                mm.close()
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self._blocks.pop(bid, None)

    # ── copy ──────────────────────────────────────────────────────────

    def copy(self, src: MemBlock, dst: MemBlock, stream: Any | None = None) -> None:
        """Async copy between tiers.

        Time complexity:  O(size / bandwidth).
        Space complexity: O(1).
        """
        if isinstance(src.data, torch.Tensor) and isinstance(dst.data, torch.Tensor):
            s = stream or torch.cuda.current_stream(self._device)
            with torch.cuda.stream(s):
                dst.data[:src.size_bytes].copy_(src.data[:src.size_bytes], non_blocking=True)
        elif isinstance(src.data, mmap.mmap) and isinstance(dst.data, torch.Tensor):
            raw = src.data[:src.size_bytes]
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            dst.data[:src.size_bytes].copy_(t)
        elif isinstance(src.data, torch.Tensor) and isinstance(dst.data, mmap.mmap):
            raw = src.data[:src.size_bytes].cpu().numpy().tobytes()
            dst.data[:src.size_bytes] = raw
        else:
            dst.data[:src.size_bytes] = src.data[:src.size_bytes]

    # ── matmul ────────────────────────────────────────────────────────

    def matmul(self, a: MemBlock, b: MemBlock, out: MemBlock | None = None) -> MemBlock:
        """Matrix multiply via hipBLASLt (torch.matmul on ROCm).

        Time complexity:  O(M × N × K).
        Space complexity: O(M × N).
        """
        ta = a.data if isinstance(a.data, torch.Tensor) else torch.frombuffer(bytearray(a.data[:a.size_bytes]), dtype=torch.float16)
        tb = b.data if isinstance(b.data, torch.Tensor) else torch.frombuffer(bytearray(b.data[:b.size_bytes]), dtype=torch.float16)

        if out is not None and isinstance(out.data, torch.Tensor):
            torch.matmul(ta, tb, out=out.data)
            return out

        result = torch.matmul(ta, tb)
        blk = MemBlock(data=result, tier=MemTier.T0, size_bytes=result.nelement() * result.element_size(),
                        address=result.data_ptr(), dtype=str(result.dtype), shape=tuple(result.shape))
        self._blocks[blk.block_id] = blk
        return blk

    # ── reduce ────────────────────────────────────────────────────────

    def reduce(self, tensor: MemBlock, op: ReduceOp, group: Any | None = None) -> MemBlock:
        """All-reduce via RCCL (through torch.distributed).

        Time complexity:  O(size × log(world_size)).
        Space complexity: O(size).
        """
        if not torch.distributed.is_initialized():
            logger.debug("RocmHAL.reduce: distributed not initialised — returning input unchanged")
            return tensor

        t = tensor.data
        dist_op = _REDUCE_OP_MAP.get(op)
        if dist_op is None:
            if op == ReduceOp.AVG:
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=group)
                t.div_(torch.distributed.get_world_size(group))
            else:
                raise ValueError(f"Unsupported reduce op: {op}")
        else:
            torch.distributed.all_reduce(t, op=dist_op, group=group)
        return tensor

    # ── repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RocmHAL(device={self._device}, "
            f"T0={_human_bytes(self._t0_used)}/{_human_bytes(self._t0_total)}, "
            f"T1={_human_bytes(self._t1_used)}/{_human_bytes(self._t1_capacity)}, "
            f"T2={_human_bytes(self._t2_used)})"
        )
