import importlib.util
import sys
from pathlib import Path

import torch

_module_path = Path(__file__).resolve().parents[1] / 'zenyx/ops/attention/flash_cpu.py'
_spec = importlib.util.spec_from_file_location('flash_cpu_module', _module_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
flash_attention_cpu = _mod.flash_attention_cpu


def test_flash_attention_cpu_shape_and_finite():
    torch.manual_seed(0)
    q = torch.randn(1, 64, 4, 32, device="cpu")
    k = torch.randn(1, 64, 4, 32, device="cpu")
    v = torch.randn(1, 64, 4, 32, device="cpu")
    out = flash_attention_cpu(q, k, v, causal=True)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_flash_attention_cpu_causal_no_future_q_effect_on_past_outputs():
    torch.manual_seed(1)
    q = torch.randn(1, 64, 4, 32, device="cpu")
    k = torch.randn(1, 64, 4, 32, device="cpu")
    v = torch.randn(1, 64, 4, 32, device="cpu")

    i = 20
    out_ref = flash_attention_cpu(q, k, v, causal=True)

    q_mut = q.clone()
    q_mut[:, i + 1 :, :, :] = 0
    out_mut = flash_attention_cpu(q_mut, k, v, causal=True)

    assert torch.allclose(out_ref[:, :i, :, :], out_mut[:, :i, :, :], atol=1e-5, rtol=1e-5)


def test_flash_attention_cpu_chunked_path():
    torch.manual_seed(2)
    q = torch.randn(1, 64, 4, 32, device="cpu")
    k = torch.randn(1, 64, 4, 32, device="cpu")
    v = torch.randn(1, 64, 4, 32, device="cpu")

    out = flash_attention_cpu(q, k, v, causal=True, chunk_size=8)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
