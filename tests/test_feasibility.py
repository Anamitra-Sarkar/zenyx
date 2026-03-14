"""Tests for the OOM feasibility checker.

Verifies the corrected condition F_compute <= B_01 AND F_compute <= B_12
against known hardware configs. All tests are pure-math, no hardware needed.

Run with:
    pytest tests/test_feasibility.py -v
"""
from __future__ import annotations

import unittest

from zenyx.core.allocator.feasibility import (
    FeasibilityResult,
    check_feasibility,
    check_feasibility_for_hardware,
    compute_throughput_from_hardware,
)

# ---------------------------------------------------------------------------
# Known hardware constants
# ---------------------------------------------------------------------------

# H100 SXM5
_H100_HBM_BW     = 3.35e12   # 3.35 TB/s HBM3e
_H100_NVME_BW    = 14e9      # 14 GB/s Gen5 NVMe
_H100_TFLOPS_BF16 = 3958.0   # peak BF16 TFLOPS (sparse)

# A100 SXM4
_A100_HBM_BW     = 2.0e12    # 2 TB/s HBM2e
_A100_NVME_BW    = 7.5e9     # 7.5 GB/s Gen4 NVMe
_A100_TFLOPS_BF16 = 312.0

# Toy bottleneck config (slow bandwidth, fast compute)
_TOY_BW          = 10e9      # 10 GB/s
_TOY_TFLOPS      = 989.0     # ~V100-class

_FLOP_PER_BYTE   = 16.0      # BF16 transformer arithmetic intensity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeThroughputConversion(unittest.TestCase):
    """compute_throughput_from_hardware unit test."""

    def test_h100_conversion(self):
        ct = compute_throughput_from_hardware(_H100_TFLOPS_BF16, _FLOP_PER_BYTE)
        expected = _H100_TFLOPS_BF16 * 1e12 / _FLOP_PER_BYTE
        self.assertAlmostEqual(ct, expected, places=0)

    def test_raises_on_zero_tflops(self):
        with self.assertRaises(ValueError):
            compute_throughput_from_hardware(0.0)

    def test_raises_on_negative_flop_per_byte(self):
        with self.assertRaises(ValueError):
            compute_throughput_from_hardware(100.0, flop_per_byte=-1.0)


class TestH100Feasible(unittest.TestCase):
    """H100 + Gen5 NVMe must be feasible (was broken with inverted condition)."""

    def test_h100_is_feasible(self):
        result = check_feasibility_for_hardware(
            bandwidth_t0_t1=_H100_HBM_BW,
            bandwidth_t1_t2=_H100_NVME_BW,
            compute_tflops=_H100_TFLOPS_BF16,
            flop_per_byte=_FLOP_PER_BYTE,
        )
        self.assertIsInstance(result, FeasibilityResult)
        self.assertTrue(
            result.is_feasible,
            f"H100 should be feasible but got: {result}",
        )
        # Margin must be negative (headroom)
        self.assertLessEqual(result.margin, 0.0)


class TestA100Feasible(unittest.TestCase):
    """A100 + Gen4 NVMe must be feasible."""

    def test_a100_is_feasible(self):
        result = check_feasibility_for_hardware(
            bandwidth_t0_t1=_A100_HBM_BW,
            bandwidth_t1_t2=_A100_NVME_BW,
            compute_tflops=_A100_TFLOPS_BF16,
            flop_per_byte=_FLOP_PER_BYTE,
        )
        self.assertTrue(
            result.is_feasible,
            f"A100 should be feasible but got: {result}",
        )
        self.assertLessEqual(result.margin, 0.0)


class TestBottleneckInfeasible(unittest.TestCase):
    """Slow bandwidth + fast compute must be correctly flagged infeasible."""

    def test_toy_bottleneck_is_infeasible(self):
        result = check_feasibility_for_hardware(
            bandwidth_t0_t1=_TOY_BW,
            bandwidth_t1_t2=_TOY_BW,
            compute_tflops=_TOY_TFLOPS,
            flop_per_byte=_FLOP_PER_BYTE,
        )
        self.assertFalse(
            result.is_feasible,
            f"Toy bottleneck config should be infeasible but got: {result}",
        )
        # Margin must be positive (deficit)
        self.assertGreater(result.margin, 0.0)


class TestMarginSign(unittest.TestCase):
    """margin = max(F-B01, F-B12): positive = deficit, negative = headroom."""

    def test_feasible_has_negative_margin(self):
        result = check_feasibility(
            bandwidth_t0_t1=_H100_HBM_BW,
            bandwidth_t1_t2=_H100_NVME_BW,
            compute_throughput=compute_throughput_from_hardware(_H100_TFLOPS_BF16),
        )
        self.assertLess(result.margin, 0.0)

    def test_infeasible_has_positive_margin(self):
        result = check_feasibility(
            bandwidth_t0_t1=_TOY_BW,
            bandwidth_t1_t2=_TOY_BW,
            compute_throughput=compute_throughput_from_hardware(_TOY_TFLOPS),
        )
        self.assertGreater(result.margin, 0.0)


class TestValueErrorOnNonPositiveInputs(unittest.TestCase):
    """check_feasibility must raise ValueError for non-positive inputs."""

    def test_zero_b01_raises(self):
        with self.assertRaises(ValueError):
            check_feasibility(0.0, 1e9, 1e10)

    def test_zero_b12_raises(self):
        with self.assertRaises(ValueError):
            check_feasibility(1e12, 0.0, 1e10)

    def test_zero_compute_raises(self):
        with self.assertRaises(ValueError):
            check_feasibility(1e12, 1e9, 0.0)

    def test_negative_b01_raises(self):
        with self.assertRaises(ValueError):
            check_feasibility(-1.0, 1e9, 1e10)


if __name__ == "__main__":
    unittest.main()
