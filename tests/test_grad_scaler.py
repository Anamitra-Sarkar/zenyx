"""Tests for grad_scaler.py — ZenyxGradScaler."""

from __future__ import annotations

import torch
import torch.nn as nn

from zenyx.train.grad_scaler import ZenyxGradScaler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestZenyxGradScalerSmoke:
    """Module imports and main class instantiates without error."""

    def test_smoke_enabled(self) -> None:
        scaler = ZenyxGradScaler(enabled=True)
        assert scaler is not None

    def test_smoke_disabled(self) -> None:
        scaler = ZenyxGradScaler(enabled=False)
        assert scaler is not None


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


class TestGradScalerBehavior:
    """Test real backward pass behaviour with manual scaler."""

    def test_enabled_backward_pass_has_gradients(self) -> None:
        """With enabled=True (manual path), gradients exist after step."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = ZenyxGradScaler(enabled=True)

        x = torch.randn(3, 4)
        y = model(x).sum()
        scaled = scaler.scale(y)
        scaled.backward()

        # Gradients should exist
        for p in model.parameters():
            assert p.grad is not None

        scaler.step(optimizer)
        scaler.update()

    def test_disabled_scale_is_identity(self) -> None:
        """With enabled=False, scale() returns the loss unchanged."""
        scaler = ZenyxGradScaler(enabled=False)
        loss = torch.tensor(3.14)
        scaled = scaler.scale(loss)
        assert torch.equal(scaled, loss)
        assert scaler.get_scale() == scaler._scale


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestGradScalerEdgeCases:
    """Test inf/NaN gradient handling."""

    def test_update_does_not_raise_on_nan_gradient(self) -> None:
        """update() should not raise even when gradients contain NaN."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = ZenyxGradScaler(enabled=True)

        x = torch.randn(3, 4)
        y = model(x).sum()
        scaled = scaler.scale(y)
        scaled.backward()

        # Inject NaN into gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("nan"))

        # step() should skip the optimizer step, not raise
        scaler.step(optimizer)
        scaler.update()  # Should not raise

        # Scale should have decreased (backoff)
        assert scaler._scale < 2.0**16

    def test_update_does_not_raise_on_inf_gradient(self) -> None:
        """update() should not raise on inf gradients."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = ZenyxGradScaler(enabled=True)

        x = torch.randn(3, 4)
        y = model(x).sum()
        scaled = scaler.scale(y)
        scaled.backward()

        # Inject Inf
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("inf"))

        scaler.step(optimizer)
        scaler.update()
        assert scaler._scale < 2.0**16
