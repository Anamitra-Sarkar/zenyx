import copy

import torch
import torch.nn as nn

from zenyx.train.activation_checkpoint import (
    CheckpointedBlock,
    selective_checkpoint_wrapper,
)


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


def test_checkpointed_block_instantiation():
    wrapped = CheckpointedBlock(nn.Linear(8, 8))
    assert wrapped is not None


def test_every_nth_1_wraps_all_children():
    model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(5)])
    wrapped = selective_checkpoint_wrapper(model, checkpoint_every_nth=1)
    for child in wrapped.children():
        assert isinstance(child, CheckpointedBlock)


def test_every_nth_999_wraps_only_first():
    model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(4)])
    wrapped = selective_checkpoint_wrapper(model, checkpoint_every_nth=999)
    children = list(wrapped.children())
    assert isinstance(children[0], CheckpointedBlock)
    assert not isinstance(children[1], CheckpointedBlock)
    assert not isinstance(children[2], CheckpointedBlock)
    assert not isinstance(children[3], CheckpointedBlock)


def test_wrapped_model_same_output():
    torch.manual_seed(0)
    model_a = nn.Sequential(*[nn.Linear(8, 8) for _ in range(4)])
    model_b = copy.deepcopy(model_a)
    model_b = selective_checkpoint_wrapper(model_b, checkpoint_every_nth=1)

    x = torch.randn(2, 8, device=torch.device("cpu"))
    out_a = model_a(x)
    out_b = model_b(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_checkpointed_block_forward_same_output():
    torch.manual_seed(0)
    block = nn.Linear(8, 8)
    wrapped = CheckpointedBlock(block)
    x = torch.randn(2, 8, requires_grad=True, device=torch.device("cpu"))
    assert torch.allclose(block(x), wrapped(x), atol=1e-6)
