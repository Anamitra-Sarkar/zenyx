import importlib.util
import sys
from pathlib import Path

import torch

_module_path = Path(__file__).resolve().parents[1] / 'zenyx/train/grad_scaler.py'
_spec = importlib.util.spec_from_file_location('grad_scaler_module', _module_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
ZenyxGradScaler = _mod.ZenyxGradScaler


class _TrackingSGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-2):
        super().__init__(params, lr=lr)
        self.step_called = 0

    def step(self, closure=None):
        self.step_called += 1
        return super().step(closure=closure)


def test_grad_scaler_disabled_is_noop_step():
    p = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    p.grad = torch.tensor([0.5], device="cpu")
    opt = _TrackingSGD([p], lr=0.1)
    scaler = ZenyxGradScaler(enabled=False)

    before = p.detach().clone()
    scaler.step(opt)
    assert opt.step_called == 1
    assert not torch.equal(before, p.detach())


def test_manual_path_overflow_skips_step_and_unscales_all_grads_once():
    p1 = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    p2 = torch.nn.Parameter(torch.tensor([2.0], device="cpu"))
    opt = _TrackingSGD([p1, p2], lr=0.1)

    scaler = ZenyxGradScaler(init_scale=8.0, enabled=True)
    scaler._scaler = None

    p1.grad = torch.tensor([16.0], device="cpu")
    p2.grad = torch.tensor([float("inf")], device="cpu")

    scaler.step(opt)

    assert opt.step_called == 0
    assert torch.allclose(p1.grad, torch.tensor([2.0], device="cpu"))
    assert torch.isinf(p2.grad).all()


def test_grad_scaler_state_dict_roundtrip_manual_path():
    scaler = ZenyxGradScaler(init_scale=32.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=3)
    scaler._scaler = None
    scaler._growth_tracker = 2
    scaler._found_inf = True
    state = scaler.state_dict()

    restored = ZenyxGradScaler(init_scale=1.0)
    restored._scaler = None
    restored.load_state_dict(state)

    assert restored.state_dict()["scale"] == 32.0
    assert restored.state_dict()["growth_tracker"] == 2
    assert restored.state_dict()["found_inf"] is True
