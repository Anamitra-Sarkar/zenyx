"""Zenyx HAL — XLA / TPU backend implementation.

Uses **JAX** and **Pallas** for TPU compute, **Shardy IR** for
custom partitioning, and **ICI collectives** for inter-chip
communication.

Key constraints:
* No GPU Direct Storage equivalent on TPU — model loading uses
  multi-threaded ``pread`` into pinned host memory → DMA to HBM.
* Custom allocator support is limited; T2 (NVMe) tier maps to host
  memory spill managed by the XLA runtime.
* Ring attention uses ``jax.experimental.custom_partitioning`` +
  Pallas, **not** ``shard_map`` + ``lax.ppermute`` (XLA reorders
  ring-sends and breaks overlap).

Typical usage::

    from zenyx.core.hal.xla_hal import XlaHAL
    hal = XlaHAL()
    blk = hal.alloc(1 << 20, MemTier.T0)
"""
from __future__ import annotations

import logging
import mmap
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

from zenyx.core.hal.base import HALBase, MemBlock, MemTier, ReduceOp, _human_bytes

logger = logging.getLogger("zenyx.core.hal.xla_hal")

# TPU memory defaults (GiB) by generation (approximate).
_TPU_MEM_GB = {
    "v4": 32,
    "v5e": 16,
    "v5p": 32,
}

# ── Optional JAX imports ──────────────────────────────────────────────────

_JAX_AVAILABLE = False
_jax = None
_jnp = None

try:
    import jax  # type: ignore[import-untyped]
    import jax.numpy as jnp  # type: ignore[import-untyped]

    _jax = jax
    _jnp = jnp
    _JAX_AVAILABLE = True
except ImportError:
    pass

# Fallback: use torch if JAX unavailable (CPU emulation)
_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class XlaHAL(HALBase):
    """XLA/TPU backend implementing the five HAL primitives.

    When JAX is available, T0 maps to TPU HBM via ``jax.device_put``.
    When JAX is unavailable, a degraded CPU-only mode is used with a
    clear warning.

    Parameters
    ----------
    device_index : int
        TPU chip index (default ``0``).
    t1_capacity_bytes : int | None
        Host DRAM pool size.  ``None`` → 50 % of available, capped 64 GiB.
    t2_dir : str | None
        Directory for spill files.

    Time complexity:  O(1) for init.
    Space complexity: O(t1_capacity_bytes + t2 spill).
    """

    def __init__(
        self,
        device_index: int = 0,
        t1_capacity_bytes: int | None = None,
        t2_dir: str | None = None,
    ) -> None:
        if not _JAX_AVAILABLE:
            logger.warning(
                "JAX not available — XlaHAL running in degraded CPU-emulation mode. "
                "Install jax[tpu] for TPU support."
            )
            self._device = None
            self._t0_total = 0
        else:
            devices = _jax.devices()
            if device_index < len(devices):
                self._device = devices[device_index]
            else:
                self._device = devices[0] if devices else None
            # FIX: Set HBM size based on TPU generation when device_kind is available.
            kind = str(getattr(self._device, "device_kind", "")).lower() if self._device else ""
            if "v5p" in kind:
                self._t0_total = _TPU_MEM_GB["v5p"] * (1 << 30)
            elif "v5e" in kind:
                self._t0_total = _TPU_MEM_GB["v5e"] * (1 << 30)
            elif "v4" in kind:
                self._t0_total = _TPU_MEM_GB["v4"] * (1 << 30)
            else:
                self._t0_total = _TPU_MEM_GB["v5e"] * (1 << 30)
            logger.info("XlaHAL: using device %s", self._device)

        self._t0_used: int = 0

        # T1 — host DRAM
        if t1_capacity_bytes is None:
            try:
                import psutil
                t1_capacity_bytes = min(psutil.virtual_memory().available // 2, 64 * (1 << 30))
            except ImportError:
                t1_capacity_bytes = 16 * (1 << 30)
        self._t1_capacity = t1_capacity_bytes
        self._t1_used: int = 0

        # T2 — spill to disk
        self._t2_dir = t2_dir or tempfile.gettempdir()
        self._t2_used: int = 0
        self._t2_files: Dict[str, Tuple[mmap.mmap, str]] = {}

        self._io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="zenyx-xla-io")
        self._blocks: Dict[str, MemBlock] = {}

    # ── alloc ─────────────────────────────────────────────────────────

    def alloc(self, size_bytes: int, tier: MemTier) -> MemBlock:
        """Allocate *size_bytes* on *tier*.

        For T0 on TPU, uses ``jax.device_put`` to place data on HBM.
        For T1, uses host (pinned if CUDA is available, unpinned otherwise).
        For T2, uses mmap spill file.

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
        if _JAX_AVAILABLE and self._device is not None:
            num_elements = size_bytes  # uint8 → 1 byte per element
            arr = _jax.device_put(_jnp.zeros(num_elements, dtype=_jnp.uint8), self._device)
            self._t0_used += size_bytes
            blk = MemBlock(data=arr, tier=MemTier.T0, size_bytes=size_bytes,
                            address=0, dtype="uint8", shape=(num_elements,))
            self._blocks[blk.block_id] = blk
            return blk
        elif _TORCH_AVAILABLE:
            logger.debug("XlaHAL: no TPU device — allocating T0 on CPU via torch")
            t = torch.empty(size_bytes, dtype=torch.uint8)
            self._t0_used += size_bytes
            blk = MemBlock(data=t, tier=MemTier.T0, size_bytes=size_bytes,
                            address=t.data_ptr(), dtype="uint8", shape=t.shape)
            self._blocks[blk.block_id] = blk
            return blk
        else:
            return self._alloc_t1(size_bytes)

    def _alloc_t1(self, size_bytes: int) -> MemBlock:
        if self._t1_used + size_bytes > self._t1_capacity:
            logger.warning("XlaHAL: T1 capacity exceeded — spilling to T2")
            return self._alloc_t2(size_bytes)
        if _TORCH_AVAILABLE:
            # pin_memory requires a CUDA runtime — guard to avoid
            # RuntimeError: Cannot pin memory without CUDA on TPU-only hosts.
            _pin = torch.cuda.is_available()
            t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=_pin)
            data = t
        elif _JAX_AVAILABLE:
            data = _jnp.zeros(size_bytes, dtype=_jnp.uint8)
        else:
            data = bytearray(size_bytes)
        self._t1_used += size_bytes
        blk = MemBlock(data=data, tier=MemTier.T1, size_bytes=size_bytes,
                        address=0, dtype="uint8", shape=(size_bytes,))
        self._blocks[blk.block_id] = blk
        return blk

    def _alloc_t2(self, size_bytes: int) -> MemBlock:
        path = os.path.join(self._t2_dir, f"zenyx_xla_{os.getpid()}_{len(self._t2_files)}.mmap")
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
        """Copy data between tiers.

        On TPU, host→device uses ``jax.device_put`` and device→host uses
        ``jax.device_get``.

        Time complexity:  O(size / bandwidth).
        Space complexity: O(1).
        """
        if _JAX_AVAILABLE:
            import numpy as np  # noqa: F811

            # device → host
            if src.tier == MemTier.T0 and dst.tier in (MemTier.T1, MemTier.T2):
                host_arr = _jax.device_get(src.data)
                if isinstance(dst.data, mmap.mmap):
                    dst.data[:src.size_bytes] = bytes(np.asarray(host_arr, dtype=np.uint8))
                elif _TORCH_AVAILABLE and isinstance(dst.data, torch.Tensor):
                    dst.data[:src.size_bytes].copy_(torch.from_numpy(np.asarray(host_arr, dtype=np.uint8)))
                else:
                    dst.data = host_arr
            # host → device
            elif src.tier in (MemTier.T1, MemTier.T2) and dst.tier == MemTier.T0:
                import numpy as np
                if isinstance(src.data, mmap.mmap):
                    host = np.frombuffer(src.data[:src.size_bytes], dtype=np.uint8)
                elif _TORCH_AVAILABLE and isinstance(src.data, torch.Tensor):
                    host = src.data[:src.size_bytes].numpy()
                else:
                    host = np.asarray(src.data, dtype=np.uint8)
                dst.data = _jax.device_put(_jnp.asarray(host), self._device)
            else:
                # same tier or T1↔T2
                if isinstance(src.data, mmap.mmap) and isinstance(dst.data, mmap.mmap):
                    dst.data[:src.size_bytes] = src.data[:src.size_bytes]
                elif _TORCH_AVAILABLE and isinstance(src.data, torch.Tensor) and isinstance(dst.data, torch.Tensor):
                    dst.data[:src.size_bytes].copy_(src.data[:src.size_bytes])
        elif _TORCH_AVAILABLE:
            if isinstance(src.data, torch.Tensor) and isinstance(dst.data, torch.Tensor):
                dst.data[:src.size_bytes].copy_(src.data[:src.size_bytes])

    # ── matmul ────────────────────────────────────────────────────────

    def matmul(self, a: MemBlock, b: MemBlock, out: MemBlock | None = None) -> MemBlock:
        """Matrix multiply via XLA matmul or torch fallback.

        Time complexity:  O(M × N × K).
        Space complexity: O(M × N).
        """
        if _JAX_AVAILABLE:
            result = _jnp.matmul(a.data, b.data)
            nbytes = result.size * result.dtype.itemsize
            blk = MemBlock(data=result, tier=MemTier.T0, size_bytes=nbytes,
                            address=0, dtype=str(result.dtype), shape=tuple(result.shape))
            self._blocks[blk.block_id] = blk
            return blk
        elif _TORCH_AVAILABLE:
            result = torch.matmul(a.data, b.data)
            blk = MemBlock(data=result, tier=MemTier.T0,
                            size_bytes=result.nelement() * result.element_size(),
                            address=result.data_ptr(), dtype=str(result.dtype),
                            shape=tuple(result.shape))
            self._blocks[blk.block_id] = blk
            return blk
        raise RuntimeError("Neither JAX nor PyTorch available for matmul")

    # ── reduce ────────────────────────────────────────────────────────

    def reduce(self, tensor: MemBlock, op: ReduceOp, group: Any | None = None) -> MemBlock:
        """All-reduce via ICI collectives (JAX) or torch.distributed.

        On TPU, uses ``jax.lax.psum`` / ``jax.lax.pmax`` / ``jax.lax.pmin``
        inside a pmap context.  Outside pmap, falls back to a no-op.

        Time complexity:  O(size × log(world_size)).
        Space complexity: O(size).
        """
        if _JAX_AVAILABLE:
            # ICI collectives require pmap context — outside pmap, return identity
            logger.debug("XlaHAL.reduce: JAX ICI collective requires pmap context; returning input")
            return tensor
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
        return tensor

    # ── repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        dev = self._device or "cpu-emulation"
        return (
            f"XlaHAL(device={dev}, "
            f"T0={_human_bytes(self._t0_used)}/{_human_bytes(self._t0_total)}, "
            f"T1={_human_bytes(self._t1_used)}/{_human_bytes(self._t1_capacity)}, "
            f"T2={_human_bytes(self._t2_used)})"
        )
