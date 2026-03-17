import math

import pytest

from zenyx.core.allocator.feasibility import (
    check_feasibility,
    compute_throughput_from_hardware,
    estimate_memory_budget,
)

_H100_HBM_BW = 3.35e12
_H100_NVME_BW = 14e9
_H100_TFLOPS_BF16 = 3958.0
_A100_HBM_BW = 2.0e12
_A100_NVME_BW = 7.5e9
_A100_TFLOPS_BF16 = 312.0
_TOY_BW = 10e9
_TOY_TFLOPS = 989.0
_FLOP_PER_BYTE = 16.0


def test_compute_throughput_from_hardware_known_value():
    out = compute_throughput_from_hardware(160.0, flop_per_byte=16.0)
    assert out == 10e12


def test_check_feasibility_true_when_bandwidth_sufficient():
    result = check_feasibility(1e11, 5e10, 5e12)
    assert result.is_feasible is True


def test_check_feasibility_false_when_compute_exceeds_bandwidth():
    result = check_feasibility(1e9, 1e9, 2e11)
    assert result.is_feasible is False


def test_estimate_memory_budget_activation_formula_accuracy():
    micro_bs = 2
    seq_len = 128
    d_ff = 256
    n_layers = 4
    dtype_bytes = 2

    budget = estimate_memory_budget(
        params=10_000_000,
        vocab_size=32_000,
        context_len=seq_len,
        d_model=256,
        n_layers=n_layers,
        n_kv_heads=4,
        dtype_bytes=dtype_bytes,
        device_count=1,
        micro_bs=micro_bs,
        seq_len=seq_len,
        d_ff=d_ff,
    )
    expected = micro_bs * seq_len * d_ff * n_layers * dtype_bytes / (1024**3)
    assert math.isclose(budget.activations_gb, expected, rel_tol=0, abs_tol=1e-12)


def test_estimate_memory_budget_legacy_fallback_warns(caplog):
    estimate_memory_budget(
        params=10_000_000,
        vocab_size=32_000,
        context_len=128,
        d_model=256,
        n_layers=4,
        n_kv_heads=4,
    )
    assert "without micro_bs/seq_len/d_ff" in caplog.text


def test_h100_is_feasible():
    result = check_feasibility(
        bandwidth_t0_t1=_H100_HBM_BW,
        bandwidth_t1_t2=_H100_NVME_BW,
        compute_throughput=compute_throughput_from_hardware(
            _H100_TFLOPS_BF16,
            _FLOP_PER_BYTE,
        ),
    )
    assert result.is_feasible is True
    assert result.margin <= 0.0


def test_a100_is_feasible():
    result = check_feasibility(
        bandwidth_t0_t1=_A100_HBM_BW,
        bandwidth_t1_t2=_A100_NVME_BW,
        compute_throughput=compute_throughput_from_hardware(
            _A100_TFLOPS_BF16,
            _FLOP_PER_BYTE,
        ),
    )
    assert result.is_feasible is True
    assert result.margin <= 0.0


def test_toy_bottleneck_is_infeasible():
    result = check_feasibility(
        bandwidth_t0_t1=_TOY_BW,
        bandwidth_t1_t2=_TOY_BW,
        compute_throughput=compute_throughput_from_hardware(
            _TOY_TFLOPS,
            _FLOP_PER_BYTE,
        ),
    )
    assert result.is_feasible is False
    assert result.margin > 0.0


def test_feasible_has_negative_margin():
    result = check_feasibility(
        bandwidth_t0_t1=_H100_HBM_BW,
        bandwidth_t1_t2=_H100_NVME_BW,
        compute_throughput=compute_throughput_from_hardware(
            _H100_TFLOPS_BF16,
            _FLOP_PER_BYTE,
        ),
    )
    assert result.margin < 0.0


def test_infeasible_has_positive_margin():
    result = check_feasibility(
        bandwidth_t0_t1=_TOY_BW,
        bandwidth_t1_t2=_TOY_BW,
        compute_throughput=compute_throughput_from_hardware(
            _TOY_TFLOPS,
            _FLOP_PER_BYTE,
        ),
    )
    assert result.margin > 0.0


def test_zero_b01_raises():
    with pytest.raises(ValueError):
        check_feasibility(0.0, 1e9, 1e10)


def test_zero_b12_raises():
    with pytest.raises(ValueError):
        check_feasibility(1e12, 0.0, 1e10)


def test_zero_compute_raises():
    with pytest.raises(ValueError):
        check_feasibility(1e12, 1e9, 0.0)


def test_negative_b01_raises():
    with pytest.raises(ValueError):
        check_feasibility(-1.0, 1e9, 1e10)


def test_compute_throughput_raises_on_zero_tflops():
    with pytest.raises(ValueError):
        compute_throughput_from_hardware(0.0)


def test_compute_throughput_raises_on_negative_flop_per_byte():
    with pytest.raises(ValueError):
        compute_throughput_from_hardware(100.0, flop_per_byte=-1.0)
