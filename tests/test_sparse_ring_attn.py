import importlib.util
import sys
import math
from pathlib import Path

import pytest

_module_path = Path(__file__).resolve().parents[1] / 'zenyx/ops/attention/sparse_ring_attn.py'
_spec = importlib.util.spec_from_file_location('sparse_ring_module', _module_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

HybridAttentionMaskDescriptor = _mod.HybridAttentionMaskDescriptor
SparseRingAttentionKernel = _mod.SparseRingAttentionKernel
build_hybrid_attention_mask = _mod.build_hybrid_attention_mask
compute_skip_schedule_production = _mod.compute_skip_schedule_production
compute_skip_schedule_theoretical = _mod.compute_skip_schedule_theoretical


def test_skip_schedule_production_ring8_has_3_active_steps():
    schedule = compute_skip_schedule_production(8, window_size=1024, seq_len=8192, world_size=8, device_id=3)
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
    with pytest.raises(AssertionError):
        SparseRingAttentionKernel(
            ring_degree=8,
            window_size=window_size,
            seq_len=seq_len,
            world_size=8,
            num_layers=min_depth - 1,
        )
