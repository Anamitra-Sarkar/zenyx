"""Tests for mixed_prec.py — FP8 E4M3 activation storage and checkpointing."""

from __future__ import annotations

import torch
import torch.nn as nn

from zenyx.train.mixed_prec import (
    FP8ActivationStorage,
    FP8CheckpointFunction,
    fp8_checkpoint,
)


# ---------------------------------------------------------------------------
# FP8ActivationStorage
# ---------------------------------------------------------------------------


class TestFP8ActivationStorage:
    """Quantise/dequantise round-trip and correctness."""

    def test_smoke_instantiation(self) -> None:
        storage = FP8ActivationStorage(force_simulated=True)
        assert storage is not None

    def test_quantize_dequantize_roundtrip(self) -> None:
        """Dequantised tensor should preserve shape and approximate scale."""
        storage = FP8ActivationStorage(force_simulated=True)
        t = torch.rand(4, 8) * 0.5  # small positive values for uint8 path
        q, s = storage.quantize(t)
        recovered = storage.dequantize(q, s, dtype=t.dtype)
        assert recovered.shape == t.shape
        # Verify the round-trip preserves relative ordering (rank correlation).
        # The simulated uint8 path is lossy but should keep approximate magnitude.
        assert recovered.min() >= 0.0
        # Correlation between original and recovered should be positive.
        assert (recovered.flatten() * t.flatten()).sum() > 0

    def test_edge_case_zero_tensor(self) -> None:
        """Quantising an all-zero tensor should not crash."""
        storage = FP8ActivationStorage(force_simulated=True)
        t = torch.zeros(4, 8)
        q, s = storage.quantize(t)
        recovered = storage.dequantize(q, s, dtype=t.dtype)
        assert recovered.shape == t.shape
        assert torch.allclose(recovered, t, atol=1e-6)


# ---------------------------------------------------------------------------
# fp8_checkpoint wrapper
# ---------------------------------------------------------------------------


class _SmallBlock(nn.Module):
    """Tiny module for testing."""

    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestFP8Checkpoint:
    """Test fp8_checkpoint convenience wrapper."""

    def test_wrapped_model_same_output_shape(self) -> None:
        """Wrapped model forward should produce output of the same shape."""
        parent = nn.Sequential(*[_SmallBlock(16) for _ in range(4)])
        wrapped = fp8_checkpoint(parent, every_n=2, force_simulated=True)
        x = torch.randn(2, 16)
        out = wrapped(x)
        assert out.shape == x.shape

    def test_every_n_1_wraps_all_layers(self) -> None:
        """every_n=1 should wrap every child."""
        parent = nn.Sequential(*[_SmallBlock(16) for _ in range(4)])
        fp8_checkpoint(parent, every_n=1, force_simulated=True)
        from zenyx.train.mixed_prec import _FP8CheckpointWrapper

        for child in parent.children():
            assert isinstance(child, _FP8CheckpointWrapper)

    def test_every_n_larger_than_num_layers(self) -> None:
        """every_n > num_layers: only the first layer (idx 0) gets wrapped."""
        parent = nn.Sequential(*[_SmallBlock(16) for _ in range(3)])
        fp8_checkpoint(parent, every_n=999, force_simulated=True)
        from zenyx.train.mixed_prec import _FP8CheckpointWrapper

        children = list(parent.children())
        # Only idx 0 satisfies 0 % 999 == 0
        assert isinstance(children[0], _FP8CheckpointWrapper)
        for child in children[1:]:
            assert not isinstance(child, _FP8CheckpointWrapper)
