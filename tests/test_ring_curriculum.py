"""Tests for Phase 9 — Dynamic ring degree with dispute resolution."""

from __future__ import annotations

import logging

import pytest
import torch

from zenyx.train.ring_curriculum import (
    CurriculumConfig,
    RingCurriculumManager,
    compute_reshard_cost_optimistic,
    compute_reshard_cost_pessimistic,
)


# ---------------------------------------------------------------------------
# Curriculum schedule tests
# ---------------------------------------------------------------------------


class TestCurriculumSchedule:
    """Test exponential step-wise curriculum progression."""

    def test_default_schedule(self) -> None:
        mgr = RingCurriculumManager(max_seq_len=1_000_000, world_size=8)
        assert mgr.current_seq_len == 8192
        assert mgr.current_ring_degree == 1
        assert mgr.current_stage == 0

    def test_custom_schedule(self) -> None:
        schedule = [(1024, 1), (4096, 2), (8192, 4)]
        mgr = RingCurriculumManager(
            max_seq_len=8192, world_size=4, curriculum_schedule=schedule
        )
        assert mgr.current_seq_len == 1024
        assert mgr.current_ring_degree == 1


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------


class TestConvergence:
    """Test loss convergence detection for stage advancement."""

    def test_not_converged_insufficient_history(self) -> None:
        mgr = RingCurriculumManager(convergence_window=200)
        assert mgr.should_advance([1.0] * 100) is False

    def test_not_converged_large_delta(self) -> None:
        mgr = RingCurriculumManager(convergence_window=200)
        history = list(range(200))  # Large delta
        assert mgr.should_advance(history) is False

    def test_converged_small_delta(self) -> None:
        mgr = RingCurriculumManager(convergence_window=200)
        # Loss barely changes over 200 steps (delta = 199 * 1e-6 = 1.99e-4 < 1e-3)
        history = [0.5 + i * 1e-6 for i in range(200)]
        assert mgr.should_advance(history) is True

    def test_no_advance_at_final_stage(self) -> None:
        schedule = [(1024, 1)]
        mgr = RingCurriculumManager(curriculum_schedule=schedule)
        history = [0.5] * 200
        assert mgr.should_advance(history) is False


# ---------------------------------------------------------------------------
# Stage advancement and PRNG realignment
# ---------------------------------------------------------------------------


class TestStageAdvancement:
    """Test advance_stage with PRNG realignment."""

    def test_advance_increments_stage(self) -> None:
        schedule = [(1024, 1), (4096, 2), (8192, 4)]
        mgr = RingCurriculumManager(
            max_seq_len=8192, world_size=4, curriculum_schedule=schedule
        )

        info = mgr.advance_stage(device_id=0)
        assert info["old_stage"] == 0
        assert info["new_stage"] == 1
        assert info["new_seq_len"] == 4096
        assert info["new_ring_degree"] == 2

    def test_advance_to_final(self) -> None:
        schedule = [(1024, 1), (4096, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=4096, world_size=4, curriculum_schedule=schedule
        )
        mgr.advance_stage()
        with pytest.raises(RuntimeError, match="final curriculum stage"):
            mgr.advance_stage()

    def test_prng_keys_realigned(self) -> None:
        """After advance_stage, PRNG keys should be realigned for new shard boundaries."""
        schedule = [(1024, 1), (4096, 2), (8192, 4)]
        mgr = RingCurriculumManager(
            max_seq_len=8192, world_size=4, curriculum_schedule=schedule
        )

        # Get keys before advance
        keys_before = mgr.get_prng_keys(device_id=0)
        assert len(keys_before) == 1024  # 1024 / 1 ring_degree

        # Advance
        info = mgr.advance_stage(device_id=0)
        assert info["prng_keys_count"] > 0

        # Get keys after advance
        keys_after = mgr.get_prng_keys(device_id=0)
        assert len(keys_after) == 2048  # 4096 / 2 ring_degree

    def test_prng_keys_deterministic(self) -> None:
        """Same seed should produce identical PRNG keys."""
        mgr1 = RingCurriculumManager(max_seq_len=8192, seed=42)
        mgr2 = RingCurriculumManager(max_seq_len=8192, seed=42)
        assert torch.equal(mgr1.get_prng_keys(0), mgr2.get_prng_keys(0))

    def test_prng_keys_different_devices(self) -> None:
        """Different devices should get different key slices."""
        schedule = [(4096, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=4096, world_size=4, curriculum_schedule=schedule
        )
        keys_0 = mgr.get_prng_keys(device_id=0)
        keys_1 = mgr.get_prng_keys(device_id=1)
        # Different devices get different token ranges
        assert not torch.equal(keys_0, keys_1)

    def test_prng_keys_requires_even_division(self) -> None:
        """active_seq_len must divide evenly by ring_degree."""
        schedule = [(1001, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=1001, world_size=2, curriculum_schedule=schedule
        )
        with pytest.raises(ValueError, match="divisible"):
            mgr.get_prng_keys(device_id=0)

    def test_optimizer_state_not_resharded(self) -> None:
        """Verify optimizer state is NOT part of the reshard operation.

        Adam moments map to the weight dimension (tensor parallelism axis),
        NOT the sequence/context parallelism axis. The advance_stage return
        value should not mention optimizer state.
        """
        schedule = [(1024, 1), (4096, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=4096, world_size=4, curriculum_schedule=schedule
        )
        info = mgr.advance_stage()
        # The reshard should only involve embeddings and PRNG keys
        assert "optimizer" not in str(info).lower()


# ---------------------------------------------------------------------------
# Active mask
# ---------------------------------------------------------------------------


class TestActiveMask:
    """Test get_active_mask for zero-padding support."""

    def test_active_mask_shape(self) -> None:
        mgr = RingCurriculumManager(max_seq_len=10000, world_size=4)
        mask = mgr.get_active_mask()
        assert mask.shape == (10000,)

    def test_active_mask_values(self) -> None:
        schedule = [(1000, 1), (5000, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=10000, world_size=4, curriculum_schedule=schedule
        )
        mask = mgr.get_active_mask()
        assert mask[:1000].sum().item() == 1000
        assert mask[1000:].sum().item() == 0

    def test_active_mask_after_advance(self) -> None:
        schedule = [(1000, 1), (5000, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=10000, world_size=4, curriculum_schedule=schedule
        )
        mgr.advance_stage()
        mask = mgr.get_active_mask()
        assert mask[:5000].sum().item() == 5000
        assert mask[5000:].sum().item() == 0


# ---------------------------------------------------------------------------
# DISPUTE 9-A: reshard path
# ---------------------------------------------------------------------------


class TestDispute9A_ReshardPath:
    """Test both reshard paths and auto-detection."""

    def test_no_recompile_path(self) -> None:
        mgr = RingCurriculumManager(no_recompile=True)
        assert mgr.reshard_path_used == "no_recompile"

    def test_recompile_path(self) -> None:
        mgr = RingCurriculumManager(no_recompile=False)
        assert mgr.reshard_path_used == "recompile"

    def test_auto_detect_defaults_to_no_recompile(self) -> None:
        mgr = RingCurriculumManager(no_recompile=None)
        assert mgr.reshard_path_used == "no_recompile"

    def test_build_static_mesh_logs_recompile_status(self, caplog) -> None:
        with caplog.at_level(logging.INFO):
            mgr = RingCurriculumManager(no_recompile=True)
            mgr.build_static_mesh()
        assert "RESHARD_RECOMPILE_REQUIRED=False" in caplog.text

    def test_build_static_mesh_recompile_path(self, caplog) -> None:
        with caplog.at_level(logging.INFO):
            mgr = RingCurriculumManager(no_recompile=False)
            mgr.build_static_mesh()
        assert "RESHARD_RECOMPILE_REQUIRED=True" in caplog.text


# ---------------------------------------------------------------------------
# DISPUTE 9-B: reshard cost estimates
# ---------------------------------------------------------------------------


class TestDispute9B_ReshardCost:
    """Test both cost estimates are computed and logged."""

    def test_optimistic_estimate(self) -> None:
        cost_ms = compute_reshard_cost_optimistic()
        # 1M tokens × 4096 × 2 = 8_192_000_000 bytes / 1024^3 GB / 400 GB/s ≈ 19.07 ms
        # (uses GiB conversion internally)
        assert 15.0 < cost_ms < 25.0  # Within expected range

    def test_pessimistic_estimate(self) -> None:
        cost_ms = compute_reshard_cost_pessimistic()
        # 516 GB / 400 GB/s = 1.29 s = 1290 ms
        assert 1200 < cost_ms < 1400

    def test_advance_logs_both_estimates(self, caplog) -> None:
        schedule = [(1024, 1), (4096, 2)]
        mgr = RingCurriculumManager(
            max_seq_len=4096, world_size=4, curriculum_schedule=schedule
        )
        with caplog.at_level(logging.INFO):
            mgr.advance_stage()
        assert "optimistic" in caplog.text.lower()
        assert "pessimistic" in caplog.text.lower()
        assert "RESHARD_ACTUAL_MS" in caplog.text


class TestCurriculumConfig:
    """Test CurriculumConfig defaults and custom."""

    def test_defaults(self) -> None:
        cfg = CurriculumConfig()
        assert cfg.max_seq_len == 1_000_000
        assert cfg.convergence_window == 200
        assert cfg.no_recompile is None
        assert len(cfg.curriculum_schedule) == 5

    def test_custom(self) -> None:
        cfg = CurriculumConfig(max_seq_len=8192, no_recompile=True)
        assert cfg.max_seq_len == 8192
        assert cfg.no_recompile is True


# ---------------------------------------------------------------------------
# Issue 9 — reshard cost invariant: opt < pess for all valid positive inputs
# ---------------------------------------------------------------------------
#
# Proof (at default parameters):
#   opt_payload  = seq_len × hidden_dim × bytes_per_element
#                = seq_len × 4096 × 2
#   pess_payload = seq_len × bytes_per_token_per_layer × num_layers
#                = seq_len × 4096 × 126
#   Both share the same ici_bandwidth_gbs denominator, so:
#   pess / opt = (4096 × 126) / (4096 × 2) = 126 / 2 = 63  > 1
#   Therefore opt < pess ∀ seq_len > 0, ici_bandwidth_gbs > 0.
#
# The parametrized test below exercises 10 distinct (seq_len, world_size,
# bw_gbps) combinations and asserts opt < pess at the shared seq_len and
# bw_gbps, holding all other parameters at defaults.


@pytest.mark.parametrize(
    "seq_len, world_size, bw_gbps",
    [
        (1_000, 1, 100.0),
        (1_000_000, 8, 3200.0),
        (8_192, 2, 50.0),
        (1, 1, 0.001),
        (999_999, 7, 12.5),
        (65_536, 4, 800.0),
        (131_072, 16, 3200.0),
        (500_000, 8, 400.0),
        (1_000_000, 1, 1.0),
        (2, 2, 100.0),
    ],
)
def test_reshard_cost_invariant_opt_lt_pess(
    seq_len: int, world_size: int, bw_gbps: float
) -> None:
    """opt < pess for all valid positive inputs (same seq_len and bw_gbps).

    world_size is accepted for parametrization completeness but not passed to
    the cost functions — both functions share only seq_len and ici_bandwidth_gbs
    as common parameters.  The invariant is purely a function of the payload
    ratio (63x), independent of world_size.
    """
    opt = compute_reshard_cost_optimistic(
        seq_len=seq_len, ici_bandwidth_gbs=bw_gbps
    )
    pess = compute_reshard_cost_pessimistic(
        seq_len=seq_len, ici_bandwidth_gbs=bw_gbps
    )
    assert opt < pess, (
        f"Invariant violated: opt={opt} >= pess={pess} "
        f"(seq_len={seq_len}, bw_gbps={bw_gbps})"
    )
    # Also verify the exact ratio holds (pess/opt = 63.0 at defaults)
    ratio = pess / opt
    assert abs(ratio - 63.0) < 1e-9, (
        f"Expected pess/opt=63.0 at default params, got {ratio} "
        f"(seq_len={seq_len}, bw_gbps={bw_gbps})"
    )
