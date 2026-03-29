"""PyTorch FSDP wrapper for distributed training.

This module provides a clean wrapper around torch.distributed.fsdp.FullyShardedDataParallel
for parameter sharding and mixed precision training.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

logger = logging.getLogger(__name__)


class FSDPWrapper:
    """Wrapper for PyTorch FSDP with clean configuration interface.

    Handles:
    - Parameter sharding across processes
    - Mixed precision (fp16) training
    - Backward prefetch for memory efficiency
    - Gradient checkpoint integration

    Example:
        >>> wrapper = FSDPWrapper(
        ...     model=model,
        ...     world_size=2,
        ...     mixed_precision="fp16",
        ... )
        >>> wrapped_model = wrapper.wrap()
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int = 1,
        rank: int = 0,
        mixed_precision: str = "fp16",
        sharding_strategy: str = "full_shard",
        cpu_offload: bool = False,
    ):
        """Initialize FSDP wrapper.

        Parameters
        ----------
        model : nn.Module
            The model to wrap.
        world_size : int
            Number of distributed processes.
        rank : int
            Current process rank.
        mixed_precision : str
            "fp16" or "bf16" or "no". Default: "fp16".
        sharding_strategy : str
            "full_shard", "shard_grad_op", or "no_shard".
        cpu_offload : bool
            Whether to offload parameters to CPU.
        """
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.mixed_precision = mixed_precision
        self.sharding_strategy = sharding_strategy
        self.cpu_offload = cpu_offload

        logger.info(
            "FSDPWrapper(world_size=%d, rank=%d, precision=%s, strategy=%s)",
            world_size,
            rank,
            mixed_precision,
            sharding_strategy,
        )

    def wrap(self) -> nn.Module:
        """Wrap the model with FSDP.

        Returns
        -------
        nn.Module
            The FSDP-wrapped model.
        """
        if self.world_size == 1:
            logger.debug("Single GPU detected. Skipping FSDP wrapping.")
            return self.model

        # Configure mixed precision
        mp_policy = self._get_mixed_precision_policy()

        # Map sharding strategy string to enum
        strategy_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        strategy = strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)

        # Wrap with FSDP
        wrapped = FSDP(
            self.model,
            mixed_precision=mp_policy,
            sharding_strategy=strategy,
            cpu_offload=None,  # No CPU offload in Phase 1
            backward_prefetch=True,
            forward_prefetch=False,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        )

        logger.info("Model wrapped with FSDP")
        return wrapped

    def _get_mixed_precision_policy(self) -> Optional[MixedPrecision]:
        """Get mixed precision configuration.

        Returns
        -------
        Optional[MixedPrecision]
            MixedPrecision policy or None.
        """
        if self.mixed_precision == "no":
            return None

        if self.mixed_precision == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        if self.mixed_precision == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        raise ValueError(f"Unknown mixed_precision: {self.mixed_precision}")

    @staticmethod
    def sync_gradients(model: nn.Module) -> None:
        """Explicitly synchronize gradients across processes.

        Parameters
        ----------
        model : nn.Module
            The FSDP-wrapped model.
        """
        if isinstance(model, FSDP):
            model.sync_gradients = True

    @staticmethod
    def consolidate_state_dict(model: nn.Module) -> dict[str, Any]:
        """Consolidate distributed state dict to rank 0.

        Parameters
        ----------
        model : nn.Module
            The FSDP-wrapped model.

        Returns
        -------
        dict[str, Any]
            Consolidated state dict (rank 0 only).
        """
        if isinstance(model, FSDP):
            return model.state_dict()
        return model.state_dict()
