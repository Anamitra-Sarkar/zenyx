"""Basic execution scheduler for forward and backward passes.

Currently supports synchronous execution. Future: async overlap.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class Scheduler:
    """Schedules forward and backward passes.

    Currently synchronous. Future phases will add:
    - Overlap of compute and communication
    - Pipeline parallelism scheduling
    - Dynamic micro-batch scheduling

    Usage:
        >>> scheduler = Scheduler()
        >>> output = scheduler.forward(model, batch)
        >>> loss = output.mean()
        >>> scheduler.backward(loss, optimizer)
    """

    def __init__(self, accumulation_steps: int = 1):
        """Initialize the scheduler.

        Parameters
        ----------
        accumulation_steps : int
            Number of steps before optimizer update. Default: 1 (no accumulation).
        """
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.accumulated_loss = 0.0

        logger.info("Scheduler(accumulation_steps=%d)", accumulation_steps)

    def forward(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Execute forward pass.

        Parameters
        ----------
        model : nn.Module
            The model.
        batch : torch.Tensor
            Input batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return model(batch)

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Execute backward pass and optionally update weights.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor.
        optimizer : Optional[Optimizer]
            Optimizer to use for updates. If None, only backward.
        """
        # Backward pass
        loss.backward()

        # Check if we should update
        self.step_counter += 1
        if self.step_counter % self.accumulation_steps == 0:
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
            self.accumulated_loss = 0.0
        else:
            self.accumulated_loss += loss.item()

    def synchronize(self) -> None:
        """Synchronize all pending operations.

        In Phase 1, this is a no-op. Future phases will use this for
        collective operation synchronization.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns
        -------
        dict[str, Any]
            Statistics.
        """
        return {
            "step": self.step_counter,
            "accumulated_loss": self.accumulated_loss,
            "accumulation_steps": self.accumulation_steps,
        }


class ExecutionPlan:
    """Plans execution of a forward-backward training loop.

    Usage:
        >>> plan = ExecutionPlan.from_model(model, batch_size=32)
        >>> print(plan.estimate_memory_gb())
    """

    def __init__(
        self,
        num_forward_ops: int,
        num_backward_ops: int,
        total_compute_ms: float,
        total_memory_mb: float,
    ):
        """Initialize execution plan.

        Parameters
        ----------
        num_forward_ops : int
            Number of forward operations.
        num_backward_ops : int
            Number of backward operations.
        total_compute_ms : float
            Estimated compute time.
        total_memory_mb : float
            Estimated memory in MB.
        """
        self.num_forward_ops = num_forward_ops
        self.num_backward_ops = num_backward_ops
        self.total_compute_ms = total_compute_ms
        self.total_memory_mb = total_memory_mb

    def estimate_memory_gb(self) -> float:
        """Estimate total memory in GB.

        Returns
        -------
        float
            Memory in GB.
        """
        return self.total_memory_mb / 1024.0

    def estimate_time_seconds(self) -> float:
        """Estimate total execution time in seconds.

        Returns
        -------
        float
            Time in seconds (forward + backward).
        """
        return (self.total_compute_ms * 2) / 1000.0  # Forward + backward

    @staticmethod
    def from_model(
        model: nn.Module,
        batch_size: int,
        seq_len: int,
    ) -> ExecutionPlan:
        """Create an execution plan from a model.

        Parameters
        ----------
        model : nn.Module
            The model.
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.

        Returns
        -------
        ExecutionPlan
            Execution plan.
        """
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Rough estimates
        num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Module))
        compute_ms = (num_params * batch_size * seq_len) / 1e9 * 1000  # Rough estimate
        memory_mb = (num_params * 2) / 1e6  # float16

        return ExecutionPlan(
            num_forward_ops=num_layers,
            num_backward_ops=num_layers,
            total_compute_ms=compute_ms,
            total_memory_mb=memory_mb,
        )
