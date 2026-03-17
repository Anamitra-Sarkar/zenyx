import importlib.util
import sys
import math
from pathlib import Path

_module_path = Path(__file__).resolve().parents[1] / 'zenyx/core/allocator/feasibility.py'
_spec = importlib.util.spec_from_file_location('feasibility_module', _module_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

check_feasibility = _mod.check_feasibility
compute_throughput_from_hardware = _mod.compute_throughput_from_hardware
estimate_memory_budget = _mod.estimate_memory_budget


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
