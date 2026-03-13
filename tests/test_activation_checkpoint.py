"""Tests for activation_checkpoint.py — CheckpointedBlock and selective_checkpoint_wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from zenyx.train.activation_checkpoint import (
    CheckpointedBlock,
    selective_checkpoint_wrapper,
)


# ---------------------------------------------------------------------------
# Helper model
# ---------------------------------------------------------------------------


class _TinyBlock(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestActivationCheckpointSmoke:

    def test_checkpointed_block_instantiation(self) -> None:
        block = CheckpointedBlock(_TinyBlock())
        assert block is not None

    def test_selective_wrapper_instantiation(self) -> None:
        model = nn.Sequential(*[_TinyBlock() for _ in range(4)])
        wrapped = selective_checkpoint_wrapper(model, checkpoint_every_nth=2)
        assert wrapped is not None


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


class TestCheckpointedBlockBehavior:

    def test_forward_produces_same_output(self) -> None:
        """CheckpointedBlock forward produces the same output as the raw block."""
        torch.manual_seed(42)
        block = _TinyBlock()
        wrapped = CheckpointedBlock(block)
        x = torch.randn(2, 8, requires_grad=True)
        out_raw = block(x)
        out_wrapped = wrapped(x)
        assert torch.allclose(out_raw, out_wrapped, atol=1e-6)


class TestSelectiveCheckpointWrapper:

    def test_every_nth_1_wraps_all(self) -> None:
        """every_nth=1 should wrap ALL children."""
        model = nn.Sequential(*[_TinyBlock() for _ in range(5)])
        selective_checkpoint_wrapper(model, checkpoint_every_nth=1)
        for child in model.children():
            assert isinstance(child, CheckpointedBlock)

    def test_every_nth_999_wraps_only_first(self) -> None:
        """every_nth=999 (> num layers) — only index 0 gets wrapped."""
        model = nn.Sequential(*[_TinyBlock() for _ in range(4)])
        selective_checkpoint_wrapper(model, checkpoint_every_nth=999)
        children = list(model.children())
        assert isinstance(children[0], CheckpointedBlock)
        for child in children[1:]:
            assert not isinstance(child, CheckpointedBlock)

    def test_wrapped_model_same_output(self) -> None:
        """Wrapped model produces the same output as the unwrapped model."""
        torch.manual_seed(0)
        model_a = nn.Sequential(*[_TinyBlock() for _ in range(4)])
        # Deep copy weights to model_b
        import copy
        model_b = copy.deepcopy(model_a)
        selective_checkpoint_wrapper(model_b, checkpoint_every_nth=1)

        x = torch.randn(2, 8)
        out_a = model_a(x)
        out_b = model_b(x)
        assert torch.allclose(out_a, out_b, atol=1e-6)
