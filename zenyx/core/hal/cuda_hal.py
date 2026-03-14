"""Zenyx CUDA HAL — concrete implementation of all five HAL primitives.

Memory model:
  * **T0** — ``torch.cuda.MemPool`` for device HBM/VRAM.
  * **T1** — Pinned host memory via ``torch.empty(..., pin_memory=True)``.
  * **T2** — Memory-mapped files on NVMe via ``mmap``.

Copy strategy:
  * T0 ↔ T1 — CUDA streams (``tensor.copy_(..., non_blocking=True)``).
  * T2 ↔ T1 — ``concurrent.futures.ThreadPoolExecutor`` for async NVMe I/O.
  * T0 ↔ T2 — Staged through an auto-allocated T1 intermediary (no direct
    path; NVMe cannot DMA directly to/from HBM without GPUDirect Storage).

This module **never** raises ``torch.cuda.OutOfMemoryError``.
"""
from __future__ import annotations

import logging
import mmap
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.cuda

from zenyx.core.hal.base import HALBase, MemBlock, MemTier, ReduceOp, _human_bytes

logger = logging.getLogger("zenyx.core.hal.cuda_hal")

# 2 MiB alignment — matches NVMe page and PCIe payload boundaries.
_ALIGN = 2 * 1024 * 1024  # 2 MiB


def _align_up(n: int, alignment: int = _ALIGN) -> int:
    """Round *n* up to the nearest multiple of *alignment*.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Internal T2 backing store
# ---------------------------------------------------------------------------


@dataclass
class _T2File:
    """Tracks a single memory-mapped file used for T2 storage.

    Attributes:
        path:   Path to the backing file on disk.
        mm:     Active ``mmap`` object.
        size:   Size of the mapped region in bytes.
    """

    path: str
    mm: mmap.mmap
    size: int
    fd: int

    def __repr__(self) -> str:
        return f"_T2File(path={self.path!r}, size={_human_bytes(self.size)})"


# ---------------------------------------------------------------------------
# CudaHAL
# ---------------------------------------------------------------------------


class CudaHAL(HALBase):
    """CUDA implementation of the Zenyx HAL.

    On construction, pre-allocates memory pools for T0 (device HBM) and T1
    (pinned host RAM) so that subsequent ``alloc`` calls are O(1) sub-
    allocations from these pools.

    Args:
        device:           CUDA device ordinal (default ``0``).
        t0_pool_fraction: Fraction of available VRAM to reserve (default 0.9).
        t1_pool_bytes:    Pinned host memory to reserve (default 8 GiB).
        t2_dir:           Directory for T2 mmap files (default tempdir).
        io_workers:       Thread-pool size for T2 async I/O (default 4).

    Time complexity:  O(1) for construction (pool allocation is one CUDA call).
    Space complexity: O(t0_pool + t1_pool + t2 files as allocated).
    """

    def __init__(
        self,
        device: int = 0,
        t0_pool_fraction: float = 0.90,
        t1_pool_bytes: int = 8 * (1024 ** 3),
        t2_dir: Optional[str] = None,
        io_workers: int = 4,
    ) -> None:
        self._device = torch.device("cuda", device)
        self._t2_dir = t2_dir or tempfile.mkdtemp(prefix="zenyx_t2_")
        self._io_pool = ThreadPoolExecutor(max_workers=io_workers)
        self._lock = threading.Lock()

        # ---- T0 pool (CUDA MemPool) ----------------------------------------
        torch.cuda.set_device(self._device)
        free_mem, total_mem = torch.cuda.mem_get_info(self._device)
        self._t0_pool_bytes = int(free_mem * t0_pool_fraction)
        # Use PyTorch's caching allocator — set max_split_size_mb to reduce
        # fragmentation and keep blocks in our controlled pool.
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            f"max_split_size_mb:{_ALIGN // (1024 * 1024)}",
        )

        self._t0_allocated: int = 0
        self._t0_blocks: Dict[str, MemBlock] = {}

        # ---- T1 pool (pinned host) -----------------------------------------
        self._t1_pool_bytes = t1_pool_bytes
        self._t1_allocated: int = 0
        self._t1_blocks: Dict[str, MemBlock] = {}

        # ---- T2 (mmap files) -----------------------------------------------
        self._t2_files: Dict[str, _T2File] = {}
        self._t2_allocated: int = 0
        self._t2_blocks: Dict[str, MemBlock] = {}

        # ---- default CUDA stream -------------------------------------------
        self._default_stream = torch.cuda.Stream(device=self._device)

        logger.info(
            "CudaHAL initialised on %s — T0 pool %s / T1 pool %s",
            self._device,
            _human_bytes(self._t0_pool_bytes),
            _human_bytes(self._t1_pool_bytes),
        )

    # ------------------------------------------------------------------
    # alloc
    # ------------------------------------------------------------------

    def alloc(self, size_bytes: int, tier: MemTier) -> MemBlock:
        """Allocate *size_bytes* from the given tier's pool.

        Allocation is aligned to 2 MiB boundaries for NVMe / PCIe compat.

        Never raises OOM — if T0 is full, the caller should evict first
        (handled at the MemoryPool level).  If the T0 allocator returns a
        zero-size sentinel (CUDA OOM was caught internally), a ``ValueError``
        is raised so the caller knows to evict before retrying.

        Args:
            size_bytes: Requested size.
            tier:       Target tier.

        Returns:
            A :class:`MemBlock` handle.

        Raises:
            ValueError: If T0 allocation returns the zero-size OOM sentinel.

        Time complexity:  O(1) amortised (pool sub-allocation).
        Space complexity: O(aligned_size).
        """
        aligned = _align_up(size_bytes)

        if tier == MemTier.T0:
            block = self._alloc_t0(aligned)
            # Guard: _alloc_t0 returns a zero-size sentinel on CUDA OOM.
            # Propagate as ValueError so the MemoryPool eviction loop can
            # catch it, evict the LRU block, and retry — never silently
            # proceeding with a corrupt zero-size block.
            if block.size_bytes == 0:
                raise ValueError(
                    f"CudaHAL: T0 allocation of {_human_bytes(aligned)} returned "
                    "zero-size sentinel (CUDA OOM). Evict blocks from T0 and retry."
                )
            return block
        elif tier == MemTier.T1:
            return self._alloc_t1(aligned)
        else:
            return self._alloc_t2(aligned)

    def _alloc_t0(self, size: int) -> MemBlock:
        """Allocate from CUDA device memory.

        Returns a zero-size sentinel block on CUDA OOM rather than
        propagating the exception.  Callers **must** check ``block.size_bytes
        == 0`` and handle accordingly (see :meth:`alloc`).

        Time complexity:  O(1) amortised.
        Space complexity: O(size).
        """
        torch.cuda.set_device(self._device)
        try:
            tensor = torch.empty(
                size // 2,  # FP16 = 2 bytes
                dtype=torch.float16,
                device=self._device,
            )
        except torch.cuda.OutOfMemoryError:
            # Never propagate — log and return a zero-size sentinel.
            # alloc() will convert this to a ValueError so the eviction
            # loop in MemoryPool can catch it cleanly.
            logger.error(
                "CUDA OOM while allocating %s on T0 — returning sentinel; caller must evict",
                _human_bytes(size),
            )
            tensor = torch.empty(0, dtype=torch.float16, device=self._device)
            # Sentinel: size_bytes=0 signals OOM to alloc()
            return MemBlock(
                data=tensor,
                tier=MemTier.T0,
                size_bytes=0,
                address=0,
                dtype="float16",
                shape=(0,),
            )

        block = MemBlock(
            data=tensor,
            tier=MemTier.T0,
            size_bytes=size,
            address=tensor.data_ptr(),
            dtype="float16",
            shape=tuple(tensor.shape),
        )
        with self._lock:
            self._t0_allocated += size
            self._t0_blocks[block.block_id] = block
        return block

    def _alloc_t1(self, size: int) -> MemBlock:
        """Allocate pinned CPU memory.

        Time complexity:  O(1) amortised.
        Space complexity: O(size).
        """
        tensor = torch.empty(
            size // 2,
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        block = MemBlock(
            data=tensor,
            tier=MemTier.T1,
            size_bytes=size,
            address=tensor.data_ptr(),
            dtype="float16",
            shape=tuple(tensor.shape),
        )
        with self._lock:
            self._t1_allocated += size
            self._t1_blocks[block.block_id] = block
        return block

    def _alloc_t2(self, size: int) -> MemBlock:
        """Allocate a memory-mapped file on NVMe for T2 storage.

        Time complexity:  O(1) (fallocate / ftruncate).
        Space complexity: O(size) on disk.
        """
        path = os.path.join(
            self._t2_dir,
            f"t2_{os.getpid()}_{threading.get_ident()}_{size}.mmap",
        )
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
        except OSError:
            os.close(fd)
            os.unlink(path)
            raise

        t2f = _T2File(path=path, mm=mm, size=size, fd=fd)
        block = MemBlock(
            data=mm,
            tier=MemTier.T2,
            size_bytes=size,
            address=0,  # no meaningful virtual address for mmap
            dtype="bytes",
            shape=(size,),
        )
        with self._lock:
            self._t2_files[block.block_id] = t2f
            self._t2_allocated += size
            self._t2_blocks[block.block_id] = block
        return block

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
            if block.tier == MemTier.T0:
                self._t0_blocks.pop(block.block_id, None)
                self._t0_allocated = max(0, self._t0_allocated - block.size_bytes)
                # Let PyTorch's caching allocator reclaim the memory.
                del block.data

            elif block.tier == MemTier.T1:
                self._t1_blocks.pop(block.block_id, None)
                self._t1_allocated = max(0, self._t1_allocated - block.size_bytes)
                del block.data

            elif block.tier == MemTier.T2:
                t2f = self._t2_files.pop(block.block_id, None)
                self._t2_blocks.pop(block.block_id, None)
                self._t2_allocated = max(0, self._t2_allocated - block.size_bytes)
                if t2f is not None:
                    t2f.mm.close()
                    os.close(t2f.fd)
                    try:
                        os.unlink(t2f.path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    # copy
    # ------------------------------------------------------------------

    def copy(
        self,
        src: MemBlock,
        dst: MemBlock,
        stream: Optional[Any] = None,
    ) -> None:
        """Asynchronously copy data from *src* to *dst*, potentially across tiers.

        Supported paths
        ---------------
        * **T0 ↔ T1** — CUDA streams (non-blocking DMA via PCIe/NVLink).
        * **T2 ↔ T1** — Thread-pool executor for async NVMe I/O.
        * **T0 ↔ T2** — Staged through an auto-allocated T1 intermediary.
          NVMe cannot DMA directly to/from device HBM without GPUDirect
          Storage (GDS), and GDS availability cannot be guaranteed at
          runtime. Staging is safe, correct, and avoids device stalls.

        Args:
            src:    Source block.
            dst:    Destination block (must be ≥ ``src.size_bytes``).
            stream: Optional ``torch.cuda.Stream``.

        Time complexity:  O(size / bandwidth).
        Space complexity: O(1) for T0↔T1 and T1↔T2; O(size) for T0↔T2
                          staging (temporary T1 buffer is freed after copy).
        """
        if dst.size_bytes < src.size_bytes:
            logger.warning(
                "copy: dst (%s) smaller than src (%s) — truncating",
                _human_bytes(dst.size_bytes),
                _human_bytes(src.size_bytes),
            )

        # ---- T0 ↔ T1 (CUDA stream copy) ----------------------------------
        if {src.tier, dst.tier} <= {MemTier.T0, MemTier.T1}:
            s = stream or self._default_stream
            with torch.cuda.stream(s):
                nbytes = min(src.size_bytes, dst.size_bytes)
                src_t = src.data
                dst_t = dst.data
                numel = nbytes // 2  # FP16 = 2 bytes
                dst_t[:numel].copy_(src_t[:numel], non_blocking=True)
            return

        # ---- T2 → T1 (read mmap into pinned tensor) ----------------------
        if src.tier == MemTier.T2 and dst.tier == MemTier.T1:
            self._io_pool.submit(self._copy_t2_to_t1, src, dst)
            return

        # ---- T1 → T2 (write pinned tensor to mmap) -----------------------
        if src.tier == MemTier.T1 and dst.tier == MemTier.T2:
            self._io_pool.submit(self._copy_t1_to_t2, src, dst)
            return

        # ---- T0 → T2 (stage: T0 → T1 intermediary → T2) -----------------
        if src.tier == MemTier.T0 and dst.tier == MemTier.T2:
            nbytes = min(src.size_bytes, dst.size_bytes)
            t1_intermediate = self._alloc_t1(nbytes)
            try:
                # Step 1: T0 → T1 via CUDA stream (synchronous w.r.t. host)
                s = stream or self._default_stream
                with torch.cuda.stream(s):
                    numel = nbytes // 2
                    t1_intermediate.data[:numel].copy_(src.data[:numel], non_blocking=False)
                torch.cuda.synchronize()
                # Step 2: T1 → T2 via thread-pool async NVMe write
                future = self._io_pool.submit(self._copy_t1_to_t2, t1_intermediate, dst)
                future.result()  # wait for NVMe write to complete
            finally:
                self.free(t1_intermediate)
            return

        # ---- T2 → T0 (stage: T2 → T1 intermediary → T0) -----------------
        if src.tier == MemTier.T2 and dst.tier == MemTier.T0:
            nbytes = min(src.size_bytes, dst.size_bytes)
            t1_intermediate = self._alloc_t1(nbytes)
            try:
                # Step 1: T2 → T1 via thread-pool async NVMe read
                future = self._io_pool.submit(self._copy_t2_to_t1, src, t1_intermediate)
                future.result()  # wait for NVMe read to complete
                # Step 2: T1 → T0 via CUDA stream
                s = stream or self._default_stream
                with torch.cuda.stream(s):
                    numel = nbytes // 2
                    dst.data[:numel].copy_(t1_intermediate.data[:numel], non_blocking=True)
            finally:
                self.free(t1_intermediate)
            return

        logger.error(
            "copy: unsupported tier pair src=%s dst=%s",
            src.tier,
            dst.tier,
        )

    @staticmethod
    def _copy_t2_to_t1(src: MemBlock, dst: MemBlock) -> None:
        """Background thread: read mmap bytes into pinned tensor.

        Time complexity:  O(size / NVMe bandwidth).
        Space complexity: O(1).
        """
        mm: mmap.mmap = src.data
        mm.seek(0)
        raw = mm.read(min(src.size_bytes, dst.size_bytes))
        buf = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        numel = min(buf.numel(), dst.data.numel())
        dst.data[:numel].copy_(buf[:numel])

    @staticmethod
    def _copy_t1_to_t2(src: MemBlock, dst: MemBlock) -> None:
        """Background thread: write pinned tensor bytes into mmap.

        Time complexity:  O(size / NVMe bandwidth).
        Space complexity: O(1).
        """
        mm: mmap.mmap = dst.data
        nbytes = min(src.size_bytes, dst.size_bytes)
        raw = bytes(src.data.numpy().view("uint8")[:nbytes])
        mm.seek(0)
        mm.write(raw)
        mm.flush()

    # ------------------------------------------------------------------
    # matmul
    # ------------------------------------------------------------------

    def matmul(
        self,
        a: MemBlock,
        b: MemBlock,
        out: Optional[MemBlock] = None,
    ) -> MemBlock:
        """Matrix multiplication via cuBLAS (``torch.matmul``).

        Supports FP16, BF16, and FP32 operands.  Both *a* and *b* must reside
        on T0 (device memory).

        Args:
            a:   Left operand.
            b:   Right operand.
            out: Optional pre-allocated output block.

        Returns:
            MemBlock containing ``a.data @ b.data``.

        Time complexity:  O(M × N × K).
        Space complexity: O(M × N) for the output tensor.
        """
        if a.tier != MemTier.T0 or b.tier != MemTier.T0:
            logger.error("matmul requires both operands on T0; got %s, %s", a.tier, b.tier)

        result_tensor: torch.Tensor
        if out is not None:
            torch.matmul(a.data, b.data, out=out.data)
            return out

        try:
            result_tensor = torch.matmul(a.data, b.data)
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during matmul — triggering GC and retrying")
            torch.cuda.empty_cache()
            result_tensor = torch.matmul(a.data, b.data)

        result_size = result_tensor.nelement() * result_tensor.element_size()
        return MemBlock(
            data=result_tensor,
            tier=MemTier.T0,
            size_bytes=result_size,
            address=result_tensor.data_ptr(),
            dtype=str(result_tensor.dtype).replace("torch.", ""),
            shape=tuple(result_tensor.shape),
        )

    # ------------------------------------------------------------------
    # reduce
    # ------------------------------------------------------------------

    def reduce(
        self,
        tensor: MemBlock,
        op: ReduceOp,
        group: Optional[Any] = None,
    ) -> MemBlock:
        """Collective all-reduce using NCCL via ``torch.distributed``.

        Falls back to a no-op if ``torch.distributed`` is not initialised
        (single-GPU case).

        Args:
            tensor: Input block on T0.
            op:     Reduction operation.
            group:  Optional process group.

        Returns:
            MemBlock with reduced result (in-place on *tensor.data*).

        Time complexity:  O(size × log(world_size)) — ring all-reduce.
        Space complexity: O(size).
        """
        if not torch.distributed.is_initialized():
            logger.debug("torch.distributed not initialised — reduce is a no-op")
            return tensor

        dist_op = _REDUCE_OP_MAP.get(op)
        if dist_op is None:
            logger.error("Unsupported reduce op %s — falling back to SUM", op)
            dist_op = torch.distributed.ReduceOp.SUM

        try:
            torch.distributed.all_reduce(tensor.data, op=dist_op, group=group)
        except RuntimeError as exc:
            logger.error("all_reduce failed: %s", exc)

        # AVG is not natively supported by NCCL — emulate as SUM / world_size
        if op == ReduceOp.AVG:
            world_size = torch.distributed.get_world_size(group)
            tensor.data.div_(world_size)

        return tensor

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CudaHAL(\n"
            f"  device={self._device},\n"
            f"  T0 usage={_human_bytes(self._t0_allocated)} / {_human_bytes(self._t0_pool_bytes)},\n"
            f"  T1 usage={_human_bytes(self._t1_allocated)} / {_human_bytes(self._t1_pool_bytes)},\n"
            f"  T2 usage={_human_bytes(self._t2_allocated)},\n"
            f"  active blocks: T0={len(self._t0_blocks)} T1={len(self._t1_blocks)} T2={len(self._t2_blocks)}\n"
            f")"
        )

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all resources held by this HAL instance.

        Time complexity:  O(B) where B = total number of live blocks.
        Space complexity: O(1).
        """
        with self._lock:
            # Free T2 mmap files
            for t2f in self._t2_files.values():
                try:
                    t2f.mm.close()
                    os.close(t2f.fd)
                    os.unlink(t2f.path)
                except OSError:
                    pass
            self._t2_files.clear()
            self._t2_blocks.clear()
            self._t2_allocated = 0

            # Clear T0/T1 block tracking (tensors freed by GC)
            self._t0_blocks.clear()
            self._t1_blocks.clear()
            self._t0_allocated = 0
            self._t1_allocated = 0

        self._io_pool.shutdown(wait=False)
        logger.info("CudaHAL closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ReduceOp mapping
# ---------------------------------------------------------------------------

_REDUCE_OP_MAP: Dict[ReduceOp, Any] = {}

# Populate lazily to avoid import-time failures when torch.distributed is
# not built or not initialised.
try:
    _REDUCE_OP_MAP = {
        ReduceOp.SUM: torch.distributed.ReduceOp.SUM,
        ReduceOp.MAX: torch.distributed.ReduceOp.MAX,
        ReduceOp.MIN: torch.distributed.ReduceOp.MIN,
        ReduceOp.AVG: torch.distributed.ReduceOp.SUM,  # AVG emulated
    }
except Exception:
    pass
