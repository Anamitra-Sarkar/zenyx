"""Offload policy for memory tier management.

Determines what data stays in GPU memory vs what can be offloaded to CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OffloadPolicy:
    """Policy for deciding what to offload to CPU.

    Attributes:
        offload_large_activations: Offload activations > this size (bytes)
        offload_optimizer_states: Offload optimizer states to CPU
        recompute_instead_of_offload: Prefer recomputation over offloading
    """

    offload_large_activations: int = 100 * 1024 * 1024  # 100 MB
    offload_optimizer_states: bool = True
    recompute_instead_of_offload: bool = True  # Prefer gradient checkpointing


def make_offload_policy(
    gpu_memory_gb: float = 80.0,
    batch_size: int = 32,
) -> OffloadPolicy:
    """Create an offload policy based on GPU memory.

    Parameters
    ----------
    gpu_memory_gb : float
        Available GPU memory in GB.
    batch_size : int
        Training batch size.

    Returns
    -------
    OffloadPolicy
        Offload policy.
    """
    # Simple heuristic: offload activations larger than 10% of GPU memory
    threshold = int(gpu_memory_gb * 0.1 * 1024 * 1024 * 1024)

    return OffloadPolicy(
        offload_large_activations=threshold,
        offload_optimizer_states=True,
        recompute_instead_of_offload=True,
    )


class OffloadManager:
    """Manages offloading of tensors between GPU and CPU.

    Usage:
        >>> policy = OffloadPolicy()
        >>> manager = OffloadManager(policy)
        >>> offloaded = manager.maybe_offload(tensor)
    """

    def __init__(self, policy: OffloadPolicy):
        """Initialize with an offload policy.

        Parameters
        ----------
        policy : OffloadPolicy
            The offload policy.
        """
        self.policy = policy
        logger.info("OffloadManager(policy=%s)", policy)

    def maybe_offload(self, tensor: torch.Tensor) -> torch.Tensor:
        """Conditionally offload a tensor to CPU.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to consider offloading.

        Returns
        -------
        torch.Tensor
            The tensor (possibly on CPU).
        """
        if not tensor.is_cuda:
            return tensor

        # Check size
        size_bytes = tensor.element_size() * tensor.numel()
        if size_bytes > self.policy.offload_large_activations:
            # Offload to CPU
            return tensor.cpu()

        return tensor

    def maybe_load(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Conditionally load a tensor back to GPU.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to load.
        device : torch.device
            Target device.

        Returns
        -------
        torch.Tensor
            The tensor on the target device.
        """
        if tensor.device == device:
            return tensor
        return tensor.to(device)

    @staticmethod
    def estimate_offload_savings(
        total_activations_gb: float,
        offload_threshold_gb: float = 10.0,
    ) -> dict[str, float]:
        """Estimate memory savings from offloading.

        Parameters
        ----------
        total_activations_gb : float
            Total activation memory in GB.
        offload_threshold_gb : float
            Offload threshold in GB.

        Returns
        -------
        dict[str, float]
            Savings estimate.
        """
        offloadable = max(0, total_activations_gb - offload_threshold_gb)
        return {
            "offloadable_gb": offloadable,
            "gpu_savings_gb": offloadable,
            "cpu_overhead_gb": offloadable * 1.05,  # 5% CPU overhead
        }
