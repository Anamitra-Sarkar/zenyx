"""Tests for lr_schedule.py — CosineWithWarmup and state_dict."""

from __future__ import annotations

import math

import torch

from zenyx.train.lr_schedule import CosineWithWarmup


def _make_scheduler(
    peak_lr: float = 1e-4,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> CosineWithWarmup:
    """Helper to create a scheduler with a dummy optimizer."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    return CosineWithWarmup(
        optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps,
        total_steps=total_steps, min_lr_ratio=min_lr_ratio,
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestCosineWarmupSmoke:

    def test_instantiation(self) -> None:
        sched = _make_scheduler()
        assert sched is not None

    def test_repr(self) -> None:
        sched = _make_scheduler()
        assert "CosineWithWarmup" in repr(sched)


# ---------------------------------------------------------------------------
# Behavioural tests — LR trajectory
# ---------------------------------------------------------------------------


class TestCosineWarmupTrajectory:

    def test_lr_at_step_0_is_zero(self) -> None:
        """Before any step, LR should be 0."""
        sched = _make_scheduler()
        assert sched.get_lr() == 0.0

    def test_lr_at_warmup_end_is_peak(self) -> None:
        """At step == warmup_steps, LR should equal peak_lr."""
        sched = _make_scheduler(warmup_steps=100, peak_lr=1e-4)
        for _ in range(100):
            sched.step()
        assert abs(sched.get_lr() - 1e-4) < 1e-10

    def test_lr_at_total_steps_is_near_min(self) -> None:
        """At step == total_steps, LR should be near min_lr."""
        sched = _make_scheduler(
            peak_lr=1e-4, warmup_steps=100, total_steps=1000, min_lr_ratio=0.1,
        )
        for _ in range(1000):
            sched.step()
        min_lr = 1e-4 * 0.1
        assert abs(sched.get_lr() - min_lr) < 1e-8

    def test_lr_never_negative(self) -> None:
        """LR must never go negative at any step."""
        sched = _make_scheduler(total_steps=500)
        for _ in range(600):  # overshoot total
            lr = sched.step()
            assert lr >= 0.0


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


class TestCosineWarmupStateDictRoundtrip:

    def test_state_dict_roundtrip(self) -> None:
        """Saving and loading state should reproduce the exact LR."""
        sched = _make_scheduler(warmup_steps=50, total_steps=500)
        for _ in range(123):
            sched.step()

        sd = sched.state_dict()
        assert sd["step"] == 123

        sched2 = _make_scheduler(warmup_steps=50, total_steps=500)
        sched2.load_state_dict(sd)
        assert sched2.current_step == 123
        assert abs(sched2.get_lr() - sched.get_lr()) < 1e-12

    def test_state_dict_keys(self) -> None:
        sched = _make_scheduler()
        sd = sched.state_dict()
        for key in ("peak_lr", "warmup_steps", "total_steps", "min_lr", "step"):
            assert key in sd
