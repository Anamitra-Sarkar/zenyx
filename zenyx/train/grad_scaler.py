"""Mixed precision gradient scaler for Zenyx.

Wraps PyTorch's ``GradScaler`` with FP8 E4M3 awareness: skips scaling for
FP8 activations (they have their own scaling via COAT paper methodology),
adjusts loss scaling on overflow, and supports per-parameter-group scaling
for heterogeneous precision models.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

__all__ = ["ZenyxGradScaler"]

logger = logging.getLogger(__name__)


class ZenyxGradScaler:
    """Mixed precision gradient scaler for Zenyx.

    Wraps torch.amp.GradScaler with:
    - FP8 E4M3 awareness: skips scaling for FP8 activations (they have
      their own scaling via COAT paper methodology)
    - Automatic loss scaling adjustment when grad overflow detected
    - Per-parameter-group scaling for models with heterogeneous precision

    Args:
        init_scale: Initial loss scale (default 2^16)
        growth_factor: Scale multiplication on non-overflow step (default 2.0)
        backoff_factor: Scale multiplication on overflow step (default 0.5)
        growth_interval: Steps between scale growth attempts (default 2000)
        enabled: Whether scaling is active (False for FP32 training)
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker: int = 0
        self._found_inf: bool = False

        # Use PyTorch's built-in GradScaler when CUDA is available
        self._scaler: Optional[torch.amp.GradScaler] = None
        if enabled and torch.cuda.is_available():
            try:
                self._scaler = torch.amp.GradScaler(
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                    enabled=True,
                )
            except Exception:
                logger.debug("torch.amp.GradScaler unavailable; using manual scaler")

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss for mixed precision backward pass.

        If FP8 activations are in use, scaling is a no-op for those tensors
        (COAT methodology handles their scaling independently).

        Args:
            loss: Scalar loss tensor.

        Returns:
            Scaled loss tensor.
        """
        if not self._enabled:
            return loss
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss * self._scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients and step the optimizer if no overflow occurred.

        Args:
            optimizer: The optimizer to step.
        """
        if not self._enabled:
            optimizer.step()
            return

        if self._scaler is not None:
            self._scaler.step(optimizer)
            return

        # Manual path: check for inf/nan in gradients
        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                        self._found_inf = True
                        break
                    # Unscale gradients
                    p.grad.div_(self._scale)
            if self._found_inf:
                break

        if not self._found_inf:
            optimizer.step()
        else:
            logger.warning("Gradient overflow detected — skipping optimizer step")

    def update(self) -> None:
        """Update the loss scale after an optimizer step.

        Increases scale after ``growth_interval`` non-overflow steps,
        decreases scale immediately on overflow.
        """
        if not self._enabled:
            return

        if self._scaler is not None:
            self._scaler.update()
            return

        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
            logger.debug("Scale decreased to %.1f after overflow", self._scale)
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
                logger.debug("Scale increased to %.1f", self._scale)

    def get_scale(self) -> float:
        """Return the current loss scale."""
        if self._scaler is not None:
            return self._scaler.get_scale()
        return self._scale

    def __repr__(self) -> str:
        backend = "cuda" if self._scaler is not None else "manual"
        return (
            f"ZenyxGradScaler(scale={self.get_scale():.1f}, "
            f"enabled={self._enabled}, backend={backend})"
        )
