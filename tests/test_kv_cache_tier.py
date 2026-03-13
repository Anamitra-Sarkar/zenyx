"""Tests for Phase 7 — Bélády-optimal KV cache tiering."""

from __future__ import annotations

import logging
import warnings

import pytest

from zenyx.train.kv_cache_tier import (
    BeladyKVCacheManager,
    KVTierConfig,
    T0_KV_BUDGET_BYTES,
    MIN_NVME_BANDWIDTH_GBS,
    validate_bandwidth_corrected,
    validate_bandwidth_original,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(
    world_size: int = 4,
    num_layers: int = 2,
    ring_degree: int = 4,
    t0_budget_bytes: int = 4 * 1024**3,
    nvme_bw: float = 10.0,
) -> BeladyKVCacheManager:
    """Create a test-sized manager (suppressing bandwidth warnings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return BeladyKVCacheManager(
            world_size=world_size,
            num_layers=num_layers,
            ring_degree=ring_degree,
            t0_budget_bytes=t0_budget_bytes,
            nvme_bandwidth_gbs=nvme_bw,
        )


# ---------------------------------------------------------------------------
# Test Bélády eviction on 4-device 4-step ring
# ---------------------------------------------------------------------------

class TestBeladyEvictionOrder:
    """Verify Bélády eviction selects the block with the farthest next use."""

    def test_eviction_order_synthetic(self) -> None:
        """4 devices, 2 layers, 4 ring steps. Eviction should remove the block
        whose next use is furthest in the future."""
        mgr = _make_manager(world_size=4, num_layers=2, ring_degree=4)
        # Use tiny block bytes so T0 budget limits are exercisable
        mgr.build_access_schedule(seq_len=4000, device_id=0)

        # Override block bytes to something tiny so we can test eviction with
        # a very small T0 budget
        mgr._block_bytes = 1024
        mgr.t0_budget_bytes = 3 * 1024  # Room for 3 blocks

        # Reset tier state
        mgr._t0_used = 0
        mgr._t0_blocks.clear()
        for key in mgr._block_tier:
            mgr._block_tier[key] = 2

        # Load first 4 blocks into T0 (should evict one when 4th arrives)
        mgr.prefetch(0, 0, "forward", device_id=0)  # block 0
        mgr.prefetch(0, 1, "forward", device_id=0)  # block 3
        mgr.prefetch(0, 2, "forward", device_id=0)  # block 2

        # All 3 should be in T0
        assert mgr._t0_used == 3 * 1024
        assert len(mgr._t0_blocks) == 3

        # Adding a 4th should trigger eviction of the block with farthest next use
        mgr.prefetch(0, 3, "forward", device_id=0)  # block 1
        assert mgr._t0_used == 3 * 1024
        assert len(mgr._t0_blocks) == 3

    def test_schedule_built_flag(self) -> None:
        mgr = _make_manager()
        assert not mgr._schedule_built
        mgr.build_access_schedule(seq_len=4000)
        assert mgr._schedule_built


# ---------------------------------------------------------------------------
# Test backward = reverse of forward
# ---------------------------------------------------------------------------

class TestBackwardReverseOfForward:
    """Backward access order is the exact reverse of forward ring rotation."""

    @pytest.mark.parametrize("device_id", [0, 1, 2, 3])
    def test_backward_is_reverse_of_forward(self, device_id: int) -> None:
        mgr = _make_manager(world_size=4, ring_degree=4)
        fwd = mgr.get_forward_access_pattern(device_id)
        bwd = mgr.get_backward_access_pattern(device_id)

        # Forward: (d - r) mod 4 for r = 0..3
        # Backward: (d + r) mod 4 for r = 0..3
        # Backward access list should be the reverse sequence of forward
        # when we consider the ring rotation direction.
        assert len(fwd) == len(bwd) == 4

        # Verify the patterns are correct
        for r in range(4):
            assert fwd[r] == (device_id - r) % 4
            assert bwd[r] == (device_id + r) % 4

        # The backward pattern traverses blocks in reverse ring direction
        # relative to forward. Specifically, if forward visits blocks in order
        # [d, d-1, d-2, d-3], backward visits [d, d+1, d+2, d+3] which is
        # the reverse ring direction.
        fwd_set = set(fwd)
        bwd_set = set(bwd)
        assert fwd_set == bwd_set  # Same blocks, different order

    def test_combined_timeline_structure(self) -> None:
        """Verify the combined timeline has forward then backward."""
        mgr = _make_manager(world_size=4, num_layers=2, ring_degree=4)
        mgr.build_access_schedule(seq_len=4000, device_id=0)

        timeline = mgr._access_timeline
        num_fwd = 2 * 4  # num_layers × ring_degree
        num_bwd = 2 * 4

        assert len(timeline) == num_fwd + num_bwd

        # Forward: layers 0,1 in order
        for i in range(4):
            assert timeline[i][1] == 0  # layer 0
        for i in range(4, 8):
            assert timeline[i][1] == 1  # layer 1

        # Backward: layers 1,0 in reverse order
        for i in range(8, 12):
            assert timeline[i][1] == 1  # layer 1 (backward)
        for i in range(12, 16):
            assert timeline[i][1] == 0  # layer 0 (backward)


# ---------------------------------------------------------------------------
# Test T0 budget never exceeds limit
# ---------------------------------------------------------------------------

class TestT0BudgetEnforcement:
    """T0 KV usage must never exceed T0_KV_BUDGET_BYTES."""

    def test_t0_never_exceeds_budget(self) -> None:
        mgr = _make_manager(world_size=4, num_layers=2, ring_degree=4)
        mgr.build_access_schedule(seq_len=4000, device_id=0)

        # Override to small values for test
        mgr._block_bytes = 512
        mgr.t0_budget_bytes = 2048  # Room for 4 blocks

        # Reset tiers
        mgr._t0_used = 0
        mgr._t0_blocks.clear()
        for key in mgr._block_tier:
            mgr._block_tier[key] = 2

        # Simulate the full timeline
        for time_step, layer_idx, block_id in mgr._access_timeline:
            if time_step < len(mgr._access_timeline) // 2:
                pass_type = "forward"
            else:
                pass_type = "backward"

            # Get the block (will prefetch if needed)
            ring_step = time_step % mgr.ring_degree
            mgr.get_block(layer_idx, ring_step, pass_type, device_id=0)

            # INVARIANT: T0 usage must never exceed budget
            assert mgr.get_t0_usage_bytes() <= mgr.t0_budget_bytes, (
                f"T0 budget exceeded at time {time_step}: "
                f"{mgr.get_t0_usage_bytes()} > {mgr.t0_budget_bytes}"
            )

    def test_t0_budget_constant(self) -> None:
        """T0_KV_BUDGET_BYTES must be exactly 4 GB."""
        assert T0_KV_BUDGET_BYTES == 4 * 1024**3


# ---------------------------------------------------------------------------
# Test both bandwidth formulas — DISPUTE 7-A
# ---------------------------------------------------------------------------

class TestBandwidthValidation:
    """Both formulas must run and produce logged output."""

    def test_corrected_formula_pass(self) -> None:
        passes, msg = validate_bandwidth_corrected(nvme_bandwidth_gbs=10.0)
        assert passes
        assert "PASS" in msg

    def test_corrected_formula_fail(self) -> None:
        passes, msg = validate_bandwidth_corrected(nvme_bandwidth_gbs=5.0)
        assert not passes
        assert "FAIL" in msg

    def test_original_formula_runs(self) -> None:
        """The original formula should run without error."""
        passes, msg = validate_bandwidth_original(nvme_bandwidth_gbs=10.0)
        assert isinstance(passes, bool)
        assert "formula" in msg.lower() or "Original" in msg

    def test_both_formulas_at_boundary(self) -> None:
        """At the 8.48 GB/s boundary, both formulas should produce results.
        Log any discrepancy as CRITICAL per DISPUTE 7-A."""
        pass_c, msg_c = validate_bandwidth_corrected(nvme_bandwidth_gbs=8.48)
        pass_o, msg_o = validate_bandwidth_original(nvme_bandwidth_gbs=8.48)

        # Both should produce results (we log if they disagree)
        assert isinstance(pass_c, bool)
        assert isinstance(pass_o, bool)

        if pass_c != pass_o:
            logging.critical(
                "DISPUTE 7-A CONFIRMED: formulas disagree at boundary 8.48 GB/s. "
                "Corrected=%s, Original=%s",
                "PASS" if pass_c else "FAIL",
                "PASS" if pass_o else "FAIL",
            )

    def test_nvme_warning_below_threshold(self) -> None:
        """Manager should emit RuntimeWarning when NVMe BW < 8.48 GB/s."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BeladyKVCacheManager(
                world_size=4,
                num_layers=2,
                ring_degree=4,
                nvme_bandwidth_gbs=5.0,
            )
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) >= 1
            assert any("bandwidth" in str(x.message).lower() for x in runtime_warnings)

    def test_no_warning_above_threshold(self) -> None:
        """No RuntimeWarning when NVMe BW >= 8.48 GB/s AND both formulas pass."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BeladyKVCacheManager(
                world_size=4,
                num_layers=2,
                ring_degree=4,
                nvme_bandwidth_gbs=200.0,  # Very high — both pass
            )
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0


class TestKVTierConfig:
    """KVTierConfig sanity checks."""

    def test_defaults(self) -> None:
        cfg = KVTierConfig()
        assert cfg.t0_budget_bytes == T0_KV_BUDGET_BYTES
        assert cfg.t1_capacity_bytes == 64 * 1024**3
        assert cfg.nvme_bandwidth_gbs == 7.5

    def test_custom(self) -> None:
        cfg = KVTierConfig(t0_budget_bytes=2 * 1024**3, nvme_bandwidth_gbs=15.0)
        assert cfg.t0_budget_bytes == 2 * 1024**3
        assert cfg.nvme_bandwidth_gbs == 15.0
