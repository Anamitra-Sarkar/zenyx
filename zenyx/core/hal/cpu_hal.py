"""Zenyx HAL — CPU / OpenBLAS backend implementation.

Pure-CPU fallback for machines without a GPU.  Uses **OpenBLAS**
(via ``torch.matmul`` or ``numpy``) and **AVX-512** auto-vectorisation
when the hardware supports it.

The three-tier model collapses on CPU:
* **T0** — main RAM (acts as "fast" tier).
* **T1** — also main RAM (same pool; pinned memory is a no-op on CPU).
* **T2** — NVMe-backed mmap files (slow spill).

Typical usage::

    from zenyx.core.hal.cpu_hal import CpuHAL
    hal = CpuHAL()
    blk = hal.alloc(1 << 20, MemTier.T0)
"""
from __future__ import annotations

import logging
import mmap
import os
import platform
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

from zenyx.core.hal.base import HALBase, MemBlock, MemTier, ReduceOp, _human_bytes

logger = logging.getLogger("zenyx.core.hal.cpu_hal")

# ── Optional imports ──────────────────────────────────────────────────────

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_NUMPY_AVAILABLE = False
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    pass


def _detect_avx512() -> bool:
    """Check if the CPU supports AVX-512 instructions.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            return "avx512" in cpuinfo.lower()
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True, timeout=5,
            )
            return "avx512" in result.stdout.lower()
    except Exception:
        pass
    return False


class CpuHAL(HALBase):
    """CPU backend implementing the five HAL primitives.

    Parameters
    ----------
    t0_capacity_bytes : int | None
        Main RAM budget for T0/T1.  ``None`` → 50 % of system RAM,
        capped at 128 GiB.
    t2_dir : str | None
        Spill directory for mmap files.

    Time complexity:  O(1) for init.
    Space complexity: O(t0_capacity_bytes + t2 spill).
    """

    def __init__(
        self,
        t0_capacity_bytes: int | None = None,
        t2_dir: str | None = None,
    ) -> None:
        self._avx512 = _detect_avx512()
        if self._avx512:
            logger.info("CpuHAL: AVX-512 detected — OpenBLAS will use wide vector units")
        else:
            logger.info("CpuHAL: AVX-512 not detected — using default SIMD width")

        # Capacity
        if t0_capacity_bytes is None:
            try:
                import psutil
                t0_capacity_bytes = min(psutil.virtual_memory().available // 2, 128 * (1 << 30))
            except ImportError:
                t0_capacity_bytes = 32 * (1 << 30)

        self._t0_capacity = t0_capacity_bytes
        self._t0_used: int = 0
        self._t1_used: int = 0  # shares T0 pool on CPU

        self._t2_dir = t2_dir or tempfile.gettempdir()
        self._t2_used: int = 0
        self._t2_files: Dict[str, Tuple[mmap.mmap, str]] = {}

        self._io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="zenyx-cpu-io")
        self._blocks: Dict[str, MemBlock] = {}

        logger.info(
            "CpuHAL initialised — RAM budget=%.2f GiB  AVX-512=%s",
            self._t0_capacity / (1 << 30),
            self._avx512,
        )

    # ── alloc ─────────────────────────────────────────────────────────

    def alloc(self, size_bytes: int, tier: MemTier) -> MemBlock:
        """Allocate *size_bytes* on *tier*.

        On CPU, T0 and T1 share the same RAM pool.  T2 spills to NVMe
        via memory-mapped files.

        Time complexity:  O(1) amortised.
        Space complexity: O(size_bytes).
        """
        if tier in (MemTier.T0, MemTier.T1):
            return self._alloc_ram(size_bytes, tier)
        else:
            return self._alloc_t2(size_bytes)

    def _alloc_ram(self, size_bytes: int, tier: MemTier) -> MemBlock:
        total_used = self._t0_used + self._t1_used
        if total_used + size_bytes > self._t0_capacity:
            logger.warning("CpuHAL: RAM budget exceeded — spilling to T2 (NVMe)")
            return self._alloc_t2(size_bytes)

        if _TORCH_AVAILABLE:
            data = torch.empty(size_bytes, dtype=torch.uint8)
            address = data.data_ptr()
        elif _NUMPY_AVAILABLE:
            data = np.empty(size_bytes, dtype=np.uint8)
            address = data.ctypes.data
        else:
            data = bytearray(size_bytes)
            address = 0

        if tier == MemTier.T0:
            self._t0_used += size_bytes
        else:
            self._t1_used += size_bytes

        blk = MemBlock(data=data, tier=tier, size_bytes=size_bytes,
                        address=address, dtype="uint8", shape=(size_bytes,))
        self._blocks[blk.block_id] = blk
        return blk

    def _alloc_t2(self, size_bytes: int) -> MemBlock:
        path = os.path.join(self._t2_dir, f"zenyx_cpu_{os.getpid()}_{len(self._t2_files)}.mmap")
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
        """Release *block*.

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        bid = block.block_id
        if bid not in self._blocks:
            return
        if block.tier == MemTier.T0:
            self._t0_used = max(0, self._t0_used - block.size_bytes)
        elif block.tier == MemTier.T1:
            self._t1_used = max(0, self._t1_used - block.size_bytes)
        elif block.tier == MemTier.T2:
            self._t2_used = max(0, self._t2_used - block.size_bytes)
            if bid in self._t2_files:
                mm, path = self._t2_files.pop(bid)
                mm.close()
                try:
                    os.unlink(path)
                except OSError:
                    pass
        block.data = None
        self._blocks.pop(bid, None)

    # ── copy ──────────────────────────────────────────────────────────

    def copy(self, src: MemBlock, dst: MemBlock, stream: Any | None = None) -> None:
        """Copy data between blocks (synchronous on CPU).

        Time complexity:  O(size / memory bandwidth).
        Space complexity: O(1).
        """
        size = min(src.size_bytes, dst.size_bytes)

        if _TORCH_AVAILABLE and isinstance(src.data, torch.Tensor) and isinstance(dst.data, torch.Tensor):
            dst.data[:size].copy_(src.data[:size])
        elif _NUMPY_AVAILABLE and isinstance(src.data, np.ndarray) and isinstance(dst.data, np.ndarray):
            dst.data[:size] = src.data[:size]
        elif isinstance(src.data, mmap.mmap) or isinstance(dst.data, mmap.mmap):
            # Read raw bytes for mmap transfers
            if isinstance(src.data, mmap.mmap):
                raw = src.data[:size]
            elif _TORCH_AVAILABLE and isinstance(src.data, torch.Tensor):
                raw = src.data[:size].numpy().tobytes()
            elif _NUMPY_AVAILABLE and isinstance(src.data, np.ndarray):
                raw = src.data[:size].tobytes()
            else:
                raw = bytes(src.data[:size])

            if isinstance(dst.data, mmap.mmap):
                dst.data[:size] = raw
            elif _TORCH_AVAILABLE and isinstance(dst.data, torch.Tensor):
                dst.data[:size].copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
            elif _NUMPY_AVAILABLE and isinstance(dst.data, np.ndarray):
                dst.data[:size] = np.frombuffer(raw, dtype=np.uint8)
        else:
            # bytearray fallback
            dst.data[:size] = src.data[:size]

    # ── matmul ────────────────────────────────────────────────────────

    def matmul(self, a: MemBlock, b: MemBlock, out: MemBlock | None = None) -> MemBlock:
        """Matrix multiply via OpenBLAS (torch) or numpy.

        Time complexity:  O(M × N × K).
        Space complexity: O(M × N).
        """
        if _TORCH_AVAILABLE:
            ta = a.data if isinstance(a.data, torch.Tensor) else torch.tensor(a.data, dtype=torch.float32)
            tb = b.data if isinstance(b.data, torch.Tensor) else torch.tensor(b.data, dtype=torch.float32)
            if out is not None and isinstance(out.data, torch.Tensor):
                torch.matmul(ta, tb, out=out.data)
                return out
            result = torch.matmul(ta, tb)
            blk = MemBlock(data=result, tier=MemTier.T0,
                            size_bytes=result.nelement() * result.element_size(),
                            address=result.data_ptr(), dtype=str(result.dtype),
                            shape=tuple(result.shape))
            self._blocks[blk.block_id] = blk
            return blk
        elif _NUMPY_AVAILABLE:
            na = a.data if isinstance(a.data, np.ndarray) else np.array(a.data, dtype=np.float32)
            nb = b.data if isinstance(b.data, np.ndarray) else np.array(b.data, dtype=np.float32)
            result = np.matmul(na, nb)
            blk = MemBlock(data=result, tier=MemTier.T0,
                            size_bytes=result.nbytes,
                            address=result.ctypes.data, dtype=str(result.dtype),
                            shape=result.shape)
            self._blocks[blk.block_id] = blk
            return blk
        raise RuntimeError("Neither PyTorch nor NumPy available for CPU matmul")

    # ── reduce ────────────────────────────────────────────────────────

    def reduce(self, tensor: MemBlock, op: ReduceOp, group: Any | None = None) -> MemBlock:
        """All-reduce (no-op on single CPU — returns input).

        For multi-process CPU, uses ``torch.distributed`` with Gloo backend
        if available.

        Time complexity:  O(size × log(world_size)) or O(1) single-process.
        Space complexity: O(size).
        """
        if _TORCH_AVAILABLE and torch.distributed.is_initialized():
            dist_op_map = {
                ReduceOp.SUM: torch.distributed.ReduceOp.SUM,
                ReduceOp.MAX: torch.distributed.ReduceOp.MAX,
                ReduceOp.MIN: torch.distributed.ReduceOp.MIN,
            }
            if op == ReduceOp.AVG:
                torch.distributed.all_reduce(tensor.data, op=torch.distributed.ReduceOp.SUM, group=group)
                tensor.data.div_(torch.distributed.get_world_size(group))
            elif op in dist_op_map:
                torch.distributed.all_reduce(tensor.data, op=dist_op_map[op], group=group)
        else:
            logger.debug("CpuHAL.reduce: single process — no-op")
        return tensor

    # ── introspection ─────────────────────────────────────────────────

    @property
    def has_avx512(self) -> bool:
        """Whether AVX-512 is available on this CPU."""
        return self._avx512

    def __repr__(self) -> str:
        return (
            f"CpuHAL(AVX-512={'yes' if self._avx512 else 'no'}, "
            f"T0+T1={_human_bytes(self._t0_used + self._t1_used)}"
            f"/{_human_bytes(self._t0_capacity)}, "
            f"T2={_human_bytes(self._t2_used)})"
        )
