import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

_module_path = Path(__file__).resolve().parents[1] / 'zenyx/train/activation_checkpoint.py'
_spec = importlib.util.spec_from_file_location('activation_checkpoint_module', _module_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
CheckpointedBlock = _mod.CheckpointedBlock
selective_checkpoint_wrapper = _mod.selective_checkpoint_wrapper


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _FallbackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 8)

    def forward(self, x):
        return self.c(self.b(self.a(x)))


def test_selective_checkpoint_wraps_every_nth_in_layers_container():
    model = _ToyModel()
    wrapped = selective_checkpoint_wrapper(model, checkpoint_every_nth=2)
    assert isinstance(wrapped.layers[0], CheckpointedBlock)
    assert isinstance(wrapped.layers[2], CheckpointedBlock)
    assert isinstance(wrapped.layers[4], CheckpointedBlock)


def test_selective_checkpoint_fallback_wraps_top_level_children():
    model = _FallbackModel()
    wrapped = selective_checkpoint_wrapper(model, checkpoint_every_nth=2)
    assert isinstance(wrapped.a, CheckpointedBlock)
    assert not isinstance(wrapped.b, CheckpointedBlock)
    assert isinstance(wrapped.c, CheckpointedBlock)


def test_forward_pass_runs_after_wrapping():
    model = _ToyModel()
    wrapped = selective_checkpoint_wrapper(model, checkpoint_every_nth=2)
    x = torch.randn(2, 8, requires_grad=True)
    y = wrapped(x)
    loss = y.sum()
    loss.backward()
    assert y.shape == (2, 8)
