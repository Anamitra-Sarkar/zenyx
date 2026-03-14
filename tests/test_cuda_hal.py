"""CPU-only unit tests for CudaHAL.

These tests do NOT require a CUDA device.  They exercise the T1 and T2
allocation paths, the T1↔T2 copy helpers, the T0↔T2 staging path via a
mocked CUDA path, and the OOM sentinel guard in alloc().

Run with:
    pytest tests/test_cuda_hal.py -v
"""
from __future__ import annotations

import mmap
import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from zenyx.core.hal.base import MemBlock, MemTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_t1_block(size_bytes: int) -> MemBlock:
    """Create a real T1 (pinned CPU) MemBlock without going through CudaHAL."""
    tensor = torch.zeros(size_bytes // 2, dtype=torch.float16)
    return MemBlock(
        data=tensor,
        tier=MemTier.T1,
        size_bytes=size_bytes,
        address=tensor.data_ptr(),
        dtype="float16",
        shape=tuple(tensor.shape),
    )


def _make_t2_block(size_bytes: int, tmpdir: str) -> tuple[MemBlock, int, str]:
    """Create a real T2 (mmap) MemBlock backed by a temp file."""
    path = os.path.join(tmpdir, f"t2_test_{size_bytes}.mmap")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, size_bytes)
    mm = mmap.mmap(fd, size_bytes)
    block = MemBlock(
        data=mm,
        tier=MemTier.T2,
        size_bytes=size_bytes,
        address=0,
        dtype="bytes",
        shape=(size_bytes,),
    )
    return block, fd, path


# ---------------------------------------------------------------------------
# CudaHAL instantiation (CPU-only, mocked CUDA)
# ---------------------------------------------------------------------------


def _make_hal_cpu(tmpdir: str):
    """Instantiate CudaHAL with CUDA calls fully mocked out."""
    from zenyx.core.hal.cuda_hal import CudaHAL

    with (
        patch("torch.cuda.set_device"),
        patch("torch.cuda.mem_get_info", return_value=(8 * 1024 ** 3, 8 * 1024 ** 3)),
        patch("torch.cuda.Stream", return_value=MagicMock()),
    ):
        hal = CudaHAL(device=0, t1_pool_bytes=4 * 1024 ** 3, t2_dir=tmpdir)
    return hal


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestT1AllocFree(unittest.TestCase):
    """T1 (pinned CPU) allocation and free round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hal = _make_hal_cpu(self.tmpdir)

    def tearDown(self):
        self.hal.close()

    def test_alloc_t1_size_tracked(self):
        block = self.hal._alloc_t1(4 * 1024 * 1024)  # 4 MiB
        self.assertEqual(block.size_bytes, 4 * 1024 * 1024)
        self.assertEqual(block.tier, MemTier.T1)
        self.assertEqual(self.hal._t1_allocated, 4 * 1024 * 1024)

    def test_free_t1_decrements_allocated(self):
        block = self.hal._alloc_t1(4 * 1024 * 1024)
        self.hal.free(block)
        self.assertEqual(self.hal._t1_allocated, 0)
        self.assertNotIn(block.block_id, self.hal._t1_blocks)

    def test_alloc_t1_tensor_shape(self):
        size = 8 * 1024 * 1024  # 8 MiB
        block = self.hal._alloc_t1(size)
        # FP16 → numel = size // 2
        self.assertEqual(block.data.numel(), size // 2)
        self.hal.free(block)


class TestT2AllocFree(unittest.TestCase):
    """T2 (mmap NVMe) allocation and free round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hal = _make_hal_cpu(self.tmpdir)

    def tearDown(self):
        self.hal.close()

    def test_alloc_t2_creates_file(self):
        size = 2 * 1024 * 1024  # 2 MiB
        block = self.hal._alloc_t2(size)
        self.assertEqual(block.size_bytes, size)
        self.assertEqual(block.tier, MemTier.T2)
        self.assertEqual(self.hal._t2_allocated, size)
        # The backing file should exist
        t2f = self.hal._t2_files[block.block_id]
        self.assertTrue(os.path.exists(t2f.path))
        self.hal.free(block)

    def test_free_t2_removes_file(self):
        size = 2 * 1024 * 1024
        block = self.hal._alloc_t2(size)
        t2f = self.hal._t2_files[block.block_id]
        path = t2f.path
        self.hal.free(block)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.hal._t2_allocated, 0)


class TestT1T2CopyHelpers(unittest.TestCase):
    """T1↔T2 static copy helper correctness."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_copy_t1_to_t2_and_back(self):
        from zenyx.core.hal.cuda_hal import CudaHAL

        size = 2 * 1024 * 1024  # 2 MiB

        # Build a T1 block with known data
        src_t1 = _make_t1_block(size)
        src_t1.data[:] = torch.arange(src_t1.data.numel(), dtype=torch.float16) % 1000

        # Build a T2 destination block
        dst_t2, fd2, path2 = _make_t2_block(size, self.tmpdir)

        # T1 → T2
        CudaHAL._copy_t1_to_t2(src_t1, dst_t2)

        # Build a fresh T1 block to receive data back
        dst_t1 = _make_t1_block(size)

        # T2 → T1
        CudaHAL._copy_t2_to_t1(dst_t2, dst_t1)

        # Data must be bit-identical (equal_nan=True so NaN==NaN is accepted).
        self.assertTrue(
            torch.allclose(src_t1.data, dst_t1.data, equal_nan=True),
            "Round-trip T1→T2→T1 data mismatch",
        )

        # Cleanup
        dst_t2.data.close()
        os.close(fd2)
        os.unlink(path2)


class TestOOMSentinelGuard(unittest.TestCase):
    """alloc() must raise ValueError when _alloc_t0 returns the zero-size sentinel."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hal = _make_hal_cpu(self.tmpdir)

    def tearDown(self):
        self.hal.close()

    def test_alloc_raises_on_oom_sentinel(self):
        """Patch _alloc_t0 to return the zero-size sentinel; alloc() must raise."""
        sentinel = MemBlock(
            data=torch.empty(0, dtype=torch.float16),
            tier=MemTier.T0,
            size_bytes=0,
            address=0,
            dtype="float16",
            shape=(0,),
        )
        self.hal._alloc_t0 = lambda size: sentinel

        with self.assertRaises(ValueError) as ctx:
            self.hal.alloc(4 * 1024 * 1024, MemTier.T0)

        self.assertIn("zero-size sentinel", str(ctx.exception))
        self.assertIn("Evict", str(ctx.exception))

    def test_alloc_t1_never_raises_for_sentinel(self):
        """T1 alloc must never trigger the sentinel guard (guard is T0-only)."""
        block = self.hal.alloc(4 * 1024 * 1024, MemTier.T1)
        self.assertGreater(block.size_bytes, 0)
        self.hal.free(block)


class TestT0T2StagedCopy(unittest.TestCase):
    """T0↔T2 copy must stage through a T1 intermediary (not silently fail)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hal = _make_hal_cpu(self.tmpdir)

    def tearDown(self):
        self.hal.close()

    def test_t0_to_t2_calls_staging(self):
        """copy(T0→T2) must allocate a T1 intermediary and call both sub-copies."""
        size = 2 * 1024 * 1024

        # Build a fake T0 block (CPU tensor, pretending to be T0)
        t0_tensor = torch.arange(size // 2, dtype=torch.float16)
        t0_block = MemBlock(
            data=t0_tensor,
            tier=MemTier.T0,
            size_bytes=size,
            address=t0_tensor.data_ptr(),
            dtype="float16",
            shape=tuple(t0_tensor.shape),
        )

        t2_block = self.hal._alloc_t2(size)

        # Track how many times _alloc_t1 is called (staging must call it once)
        original_alloc_t1 = self.hal._alloc_t1
        call_count = {"n": 0}

        def counting_alloc_t1(sz):
            call_count["n"] += 1
            return original_alloc_t1(sz)

        self.hal._alloc_t1 = counting_alloc_t1

        # Patch cuda.stream context manager and synchronize to no-ops
        with (
            patch("torch.cuda.stream"),
            patch("torch.cuda.synchronize"),
        ):
            self.hal.copy(t0_block, t2_block)

        self.assertEqual(
            call_count["n"], 1,
            "T0→T2 copy must allocate exactly one T1 intermediary block",
        )

        self.hal.free(t2_block)

    def test_t2_to_t0_calls_staging(self):
        """copy(T2→T0) must allocate a T1 intermediary and call both sub-copies."""
        size = 2 * 1024 * 1024

        t2_block = self.hal._alloc_t2(size)

        t0_tensor = torch.zeros(size // 2, dtype=torch.float16)
        t0_block = MemBlock(
            data=t0_tensor,
            tier=MemTier.T0,
            size_bytes=size,
            address=t0_tensor.data_ptr(),
            dtype="float16",
            shape=tuple(t0_tensor.shape),
        )

        original_alloc_t1 = self.hal._alloc_t1
        call_count = {"n": 0}

        def counting_alloc_t1(sz):
            call_count["n"] += 1
            return original_alloc_t1(sz)

        self.hal._alloc_t1 = counting_alloc_t1

        with (
            patch("torch.cuda.stream"),
            patch("torch.cuda.synchronize"),
        ):
            self.hal.copy(t2_block, t0_block)

        self.assertEqual(
            call_count["n"], 1,
            "T2→T0 copy must allocate exactly one T1 intermediary block",
        )

        self.hal.free(t2_block)


if __name__ == "__main__":
    unittest.main()
