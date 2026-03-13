"""Tests for Phase 10 — Sparse ring attention with dispute resolution."""

from __future__ import annotations

import logging
import math

import pytest

from zenyx.ops.attention.sparse_ring_attn import (
    SKIP_FRACTION_PRODUCTION,
    SKIP_FRACTION_THEORETICAL,
    SparseRingAttentionKernel,
    build_hybrid_attention_mask,
    compute_skip_schedule,
    compute_skip_schedule_production,
    compute_skip_schedule_theoretical,
)


# ---------------------------------------------------------------------------
# DISPUTE 10-A: Skip fraction schedules
# ---------------------------------------------------------------------------


class TestDispute10A_SkipFraction:
    """Verify both skip schedules have correct False counts."""

    def test_production_has_3_false(self) -> None:
        """Production schedule: exactly 3 execute (False), 5 skip (True) for devices 2-7.
        Devices 0 and 1 have only 2 active steps because the Device 0 global sink
        overlaps with self/prev block. The 5/8 skip fraction is the typical case."""
        # Use device_id=3 where all 3 rules produce distinct active steps
        schedule = compute_skip_schedule_production(
            ring_degree=8, window_size=131_072, seq_len=1_000_000, world_size=8,
            device_id=3,
        )
        assert len(schedule) == 8
        false_count = sum(1 for s in schedule if not s)
        assert false_count == 3, (
            f"Expected 3 execute steps, got {false_count}. Schedule: {schedule}"
        )

    def test_theoretical_has_1_false(self) -> None:
        """Theoretical schedule: exactly 1 execute (False), 7 skip (True)."""
        schedule = compute_skip_schedule_theoretical(
            ring_degree=8, window_size=131_072, seq_len=1_000_000, world_size=8
        )
        assert len(schedule) == 8
        false_count = sum(1 for s in schedule if not s)
        assert false_count == 1

    def test_production_skip_fraction_constant(self) -> None:
        assert SKIP_FRACTION_PRODUCTION == 5 / 8

    def test_theoretical_skip_fraction_constant(self) -> None:
        assert SKIP_FRACTION_THEORETICAL == 7 / 8


# ---------------------------------------------------------------------------
# Device 0 block presence in production schedule
# ---------------------------------------------------------------------------


class TestDevice0GlobalSink:
    """Device 0's block must always be in the execute set for production mode."""

    @pytest.mark.parametrize("device_id", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_device0_block_always_active(self, device_id: int) -> None:
        """For every device, the step that carries Device 0's block must be active."""
        schedule = compute_skip_schedule_production(
            ring_degree=8, window_size=131_072, seq_len=1_000_000,
            world_size=8, device_id=device_id,
        )

        # Find which step has Device 0's block: (device_id - step) % 8 == 0
        device0_step = device_id % 8  # step where source_device = 0
        assert not schedule[device0_step], (
            f"Device {device_id}: Device 0 block at step {device0_step} "
            f"should be active (False), got skip (True). Schedule: {schedule}"
        )


# ---------------------------------------------------------------------------
# No HBM load for skipped steps
# ---------------------------------------------------------------------------


class TestNoHBMLoadForSkipped:
    """Verify no HBM load is issued for skipped steps."""

    def test_load_count_matches_active_steps(self) -> None:
        kernel = SparseRingAttentionKernel(
            ring_degree=8, window_size=131_072, seq_len=1_000_000, world_size=8,
            skip_mode="production", device_id=3,
        )

        # Execute all steps
        for step in range(8):
            kernel.execute_step(step)

        # Load count should equal active steps (3 for production, device 3)
        assert kernel.get_load_count() == 3
        assert kernel.get_skip_count() == 5

    def test_theoretical_mode_minimal_loads(self) -> None:
        kernel = SparseRingAttentionKernel(
            ring_degree=8, window_size=131_072, seq_len=1_000_000, world_size=8,
            skip_mode="theoretical",
        )

        for step in range(8):
            kernel.execute_step(step)

        assert kernel.get_load_count() == 1
        assert kernel.get_skip_count() == 7

    def test_should_load_kv_consistency(self) -> None:
        kernel = SparseRingAttentionKernel(
            ring_degree=8, window_size=131_072, seq_len=1_000_000, world_size=8,
            skip_mode="production", device_id=3,
        )

        active_steps = [i for i in range(8) if kernel.should_load_kv(i)]
        assert len(active_steps) == 3

    def test_execute_returns_none_for_skipped(self) -> None:
        kernel = SparseRingAttentionKernel(
            ring_degree=8, window_size=131_072, seq_len=1_000_000, world_size=8,
            skip_mode="production",
        )

        for step in range(8):
            result = kernel.execute_step(step)
            if kernel.should_load_kv(step):
                assert result is not None
                assert result["executed"] is True
            else:
                assert result is None


# ---------------------------------------------------------------------------
# Depth assertion
# ---------------------------------------------------------------------------


class TestDepthAssertion:
    """Model must have enough layers for transitivity-based UAP."""

    def test_sufficient_depth_passes(self) -> None:
        # 126 >= ceil(1M / 128K) = 8
        kernel = SparseRingAttentionKernel(
            num_layers=126, seq_len=1_000_000, window_size=131_072
        )
        assert kernel.num_layers >= math.ceil(kernel.seq_len / kernel.window_size)

    def test_insufficient_depth_fails(self) -> None:
        with pytest.raises(AssertionError, match="too shallow"):
            SparseRingAttentionKernel(
                num_layers=2, seq_len=1_000_000, window_size=131_072
            )

    def test_exact_minimum_passes(self) -> None:
        min_depth = math.ceil(1_000_000 / 131_072)  # = 8
        kernel = SparseRingAttentionKernel(
            num_layers=min_depth, seq_len=1_000_000, window_size=131_072
        )
        assert kernel.num_layers == 8


# ---------------------------------------------------------------------------
# Hybrid attention mask
# ---------------------------------------------------------------------------


class TestHybridMask:
    """Test hybrid local+strided attention mask."""

    def test_mask_structure(self) -> None:
        mask = build_hybrid_attention_mask(
            seq_len=1_000_000, window_size=131_072, stride_step=8_192
        )
        assert mask["window_size"] == 131_072
        assert mask["stride_step"] == 8_192
        assert mask["seq_len"] == 1_000_000
        assert mask["num_stride_tokens"] == len(mask["stride_positions"])

    def test_stride_positions_correct(self) -> None:
        mask = build_hybrid_attention_mask(
            seq_len=1_000_000, stride_step=8_192
        )
        positions = mask["stride_positions"]
        # Positions should be 0, 8192, 16384, ...
        assert positions[0] == 0
        assert positions[1] == 8_192
        assert all(positions[i + 1] - positions[i] == 8_192 for i in range(len(positions) - 1))

    def test_stride_count(self) -> None:
        mask = build_hybrid_attention_mask(seq_len=1_000_000, stride_step=8_192)
        expected = math.ceil(1_000_000 / 8_192)
        assert mask["num_stride_tokens"] == expected


# ---------------------------------------------------------------------------
# Unified skip schedule API
# ---------------------------------------------------------------------------


class TestComputeSkipSchedule:
    """Test the unified compute_skip_schedule function."""

    def test_production_mode(self) -> None:
        schedule = compute_skip_schedule(
            ring_degree=8, window_size=131_072, seq_len=1_000_000,
            world_size=8, skip_mode="production", device_id=3,
        )
        assert sum(1 for s in schedule if not s) == 3

    def test_theoretical_mode(self) -> None:
        schedule = compute_skip_schedule(
            ring_degree=8, window_size=131_072, seq_len=1_000_000,
            world_size=8, skip_mode="theoretical",
        )
        assert sum(1 for s in schedule if not s) == 1

    def test_auto_mode_defaults_to_production(self) -> None:
        schedule = compute_skip_schedule(
            ring_degree=8, window_size=131_072, seq_len=1_000_000,
            world_size=8, skip_mode="auto", device_id=3,
        )
        assert sum(1 for s in schedule if not s) == 3

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown skip_mode"):
            compute_skip_schedule(
                ring_degree=8, window_size=131_072, seq_len=1_000_000,
                world_size=8, skip_mode="invalid",
            )

    def test_skip_fraction_logged(self, caplog) -> None:
        with caplog.at_level(logging.INFO):
            compute_skip_schedule(
                ring_degree=8, window_size=131_072, seq_len=1_000_000,
                world_size=8, skip_mode="production",
            )
        assert "SKIP_FRACTION" in caplog.text
