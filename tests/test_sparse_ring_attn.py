import math

import pytest

from zenyx.ops.attention.sparse_ring_attn import (
    HybridAttentionMaskDescriptor,
    SKIP_FRACTION_PRODUCTION,
    SKIP_FRACTION_THEORETICAL,
    SparseRingAttentionKernel,
    build_hybrid_attention_mask,
    compute_skip_schedule,
    compute_skip_schedule_production,
    compute_skip_schedule_theoretical,
)


def test_skip_schedule_production_ring8_has_3_active_steps():
    schedule = compute_skip_schedule_production(
        8,
        window_size=1024,
        seq_len=8192,
        world_size=8,
        device_id=3,
    )
    assert schedule.count(False) == 3


def test_skip_schedule_theoretical_ring8_has_1_active_step():
    schedule = compute_skip_schedule_theoretical(8, window_size=1024, seq_len=8192, world_size=8)
    assert schedule.count(False) == 1


def test_sparse_kernel_should_load_kv_matches_schedule():
    kernel = SparseRingAttentionKernel(ring_degree=8, seq_len=8192, window_size=1024, world_size=8)
    for step, skip in enumerate(kernel.skip_schedule):
        assert kernel.should_load_kv(step) is (not skip)


def test_build_hybrid_attention_mask_typed_descriptor():
    desc = build_hybrid_attention_mask(seq_len=4096, window_size=1024, stride_step=512)
    assert isinstance(desc, HybridAttentionMaskDescriptor)
    assert desc.num_local_blocks == math.ceil(4096 / 1024)


def test_sparse_kernel_depth_assertion_fires():
    seq_len = 4096
    window_size = 1024
    min_depth = math.ceil(seq_len / window_size)
    with pytest.raises(ValueError):
        SparseRingAttentionKernel(
            ring_degree=8,
            window_size=window_size,
            seq_len=seq_len,
            world_size=8,
            num_layers=min_depth - 1,
        )


def test_production_skip_fraction_constant():
    assert SKIP_FRACTION_PRODUCTION == 5 / 8


def test_theoretical_skip_fraction_constant():
    assert SKIP_FRACTION_THEORETICAL == 7 / 8


@pytest.mark.parametrize("device_id", range(8))
def test_device0_block_always_active(device_id: int):
    schedule = compute_skip_schedule_production(
        ring_degree=8,
        window_size=131_072,
        seq_len=1_000_000,
        world_size=8,
        device_id=device_id,
    )
    device0_step = device_id % 8
    assert schedule[device0_step] is False


def test_load_count_matches_active_steps():
    """execute_step() raises NotImplementedError for active (non-skipped) steps.

    With Fix 2, execute_step() raises NotImplementedError instead of returning
    a sentinel dict, because the Pallas kernel is not yet implemented.
    Skipped steps still return None.
    """
    kernel = SparseRingAttentionKernel(
        ring_degree=8,
        window_size=131_072,
        seq_len=1_000_000,
        world_size=8,
        skip_mode="production",
        device_id=3,
    )
    skip_count = 0
    error_count = 0
    for step in range(8):
        if not kernel.should_load_kv(step):
            result = kernel.execute_step(step)
            assert result is None
            skip_count += 1
        else:
            with pytest.raises(NotImplementedError, match="execute_step"):
                kernel.execute_step(step)
            error_count += 1

    assert skip_count == 5
    assert error_count == 3


def test_execute_returns_none_for_skipped():
    """execute_step() returns None for skipped steps, raises NotImplementedError for active steps."""
    kernel = SparseRingAttentionKernel(
        ring_degree=8,
        window_size=131_072,
        seq_len=1_000_000,
        world_size=8,
        skip_mode="production",
        device_id=3,
    )
    for step in range(8):
        if kernel.should_load_kv(step):
            with pytest.raises(NotImplementedError):
                kernel.execute_step(step)
        else:
            out = kernel.execute_step(step)
            assert out is None


def test_sufficient_depth_passes():
    SparseRingAttentionKernel(num_layers=126, seq_len=1_000_000, window_size=131_072)


def test_exact_minimum_depth_passes():
    min_depth = math.ceil(1_000_000 / 131_072)
    SparseRingAttentionKernel(num_layers=min_depth, seq_len=1_000_000, window_size=131_072)


def test_insufficient_depth_fails():
    with pytest.raises(ValueError, match="too shallow"):
        SparseRingAttentionKernel(num_layers=2, seq_len=1_000_000, window_size=131_072)


def test_ring_degree_must_match_world_size():
    """ring_degree=4 with world_size=8 is now valid (8 % 4 == 0).

    With Fix 10, the hard equality constraint (ring_degree == world_size) is
    relaxed to a divisibility check, allowing combined TP+PP+Ring configurations
    like ring_degree=4, world_size=8.
    """
    # This should now succeed since 8 % 4 == 0
    kernel = SparseRingAttentionKernel(ring_degree=4, world_size=8)
    assert kernel is not None

    # Indivisible case must still raise
    with pytest.raises(ValueError, match="divisible"):
        SparseRingAttentionKernel(ring_degree=3, world_size=8)


def test_stride_positions_are_correct():
    desc = build_hybrid_attention_mask(seq_len=1_000_000, stride_step=8_192)
    assert desc.stride_positions[0] == 0
    assert desc.stride_positions[1] == 8_192
    diffs = [b - a for a, b in zip(desc.stride_positions, desc.stride_positions[1:])]
    assert all(d == 8_192 for d in diffs)


def test_stride_count_correct():
    desc = build_hybrid_attention_mask(seq_len=1_000_000, stride_step=8_192)
    assert desc.num_stride_tokens == math.ceil(1_000_000 / 8_192)


def test_compute_skip_schedule_production_mode():
    schedule = compute_skip_schedule(
        ring_degree=8,
        window_size=131_072,
        seq_len=1_000_000,
        world_size=8,
        skip_mode="production",
        device_id=3,
    )
    assert sum(1 for s in schedule if not s) == 3


def test_compute_skip_schedule_theoretical_mode():
    schedule = compute_skip_schedule(
        ring_degree=8,
        window_size=131_072,
        seq_len=1_000_000,
        world_size=8,
        skip_mode="theoretical",
        device_id=3,
    )
    assert sum(1 for s in schedule if not s) == 1


def test_compute_skip_schedule_auto_defaults_to_production():
    schedule = compute_skip_schedule(
        ring_degree=8,
        window_size=131_072,
        seq_len=1_000_000,
        world_size=8,
        skip_mode="auto",
        device_id=3,
    )
    assert sum(1 for s in schedule if not s) == 3


def test_compute_skip_schedule_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown skip_mode"):
        compute_skip_schedule(
            ring_degree=8,
            window_size=131_072,
            seq_len=1_000_000,
            world_size=8,
            skip_mode="invalid",
            device_id=3,
        )
