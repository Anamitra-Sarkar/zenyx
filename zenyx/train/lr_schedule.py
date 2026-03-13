"""Cosine learning rate schedule with linear warmup.

Standard LLM training schedule: linear warmup from 0 → ``peak_lr`` over
``warmup_steps``, then cosine decay from ``peak_lr`` → ``min_lr`` over
the remaining steps.

``min_lr`` defaults to ``peak_lr × 0.1`` — the ratio widely used in
Chinchilla, LLaMA, and GPT-4 recipes.
"""

from __future__ import annotations

import math
from typing import List

import torch

__all__ = ["CosineWithWarmup"]


class CosineWithWarmup:
    """Cosine LR schedule with linear warmup.

    During warmup (steps 0 → warmup_steps): lr increases linearly from 0 to peak_lr.
    After warmup: lr follows cosine decay from peak_lr to min_lr.

    min_lr defaults to peak_lr / 10 (standard for LLM training).

    Args:
        optimizer: torch.optim.Optimizer
        peak_lr: Maximum learning rate (reached at end of warmup)
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: min_lr = peak_lr * min_lr_ratio (default 0.1)

    Usage:
        scheduler = CosineWithWarmup(optimizer, peak_lr=1e-4,
                                     warmup_steps=2000, total_steps=100_000)
        for step in range(total_steps):
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self._optimizer = optimizer
        self._peak_lr = peak_lr
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps
        self._min_lr = peak_lr * min_lr_ratio
        self._step: int = 0

    def step(self) -> float:
        """Advance one step and update optimizer LR. Returns current LR."""
        self._step += 1
        lr = self._compute_lr(self._step)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        """Return current LR without advancing."""
        return self._compute_lr(self._step)

    @property
    def current_step(self) -> int:
        return self._step

    def _compute_lr(self, step: int) -> float:
        """Compute the learning rate for the given step.

        Warmup: lr = peak_lr * (step / warmup_steps)
        Cosine: lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
        where progress = (step - warmup_steps) / (total_steps - warmup_steps)
        """
        if step <= 0:
            return 0.0

        if self._warmup_steps > 0 and step <= self._warmup_steps:
            return self._peak_lr * (step / self._warmup_steps)

        if self._total_steps <= self._warmup_steps:
            return self._peak_lr

        decay_steps = self._total_steps - self._warmup_steps
        progress = min((step - self._warmup_steps) / decay_steps, 1.0)
        return self._min_lr + 0.5 * (self._peak_lr - self._min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def state_dict(self) -> dict:
        """Return serialisable scheduler state.

        Saves every piece of internal state needed to resume from exactly the
        same point.
        """
        return {
            "peak_lr": self._peak_lr,
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "min_lr": self._min_lr,
            "step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from a dict produced by :meth:`state_dict`.

        After loading, the optimizer's LR is set to the value that corresponds
        to the restored step.
        """
        self._peak_lr = state["peak_lr"]
        self._warmup_steps = state["warmup_steps"]
        self._total_steps = state["total_steps"]
        self._min_lr = state["min_lr"]
        self._step = state["step"]
        # Apply the correct LR to the optimizer.
        lr = self._compute_lr(self._step)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def __repr__(self) -> str:
        return (
            f"CosineWithWarmup(peak_lr={self._peak_lr}, warmup={self._warmup_steps}, "
            f"total={self._total_steps}, step={self._step})"
        )
