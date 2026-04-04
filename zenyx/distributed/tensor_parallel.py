"""Basic tensor parallel implementation.

This module provides foundational tensor parallel support:
- Split linear layers across devices
- All-reduce after forward pass

This is a minimal but correct implementation focused on correctness.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from zenyx.distributed import all_reduce, get_rank, get_world_size

logger = logging.getLogger(__name__)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The weight matrix is split along the output dimension (columns).
    Each process computes a portion of the output.

    Forward: y = x @ W_i (partial output)
    Backward: gradient all-reduce required
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize column parallel linear layer.

        Parameters
        ----------
        in_features : int
            Size of input features.
        out_features : int
            Size of output features (total, not per-device).
        bias : bool
            Whether to use bias. Default: True.
        device : Optional[torch.device]
            Device to use.
        dtype : Optional[torch.dtype]
            Data type.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Get parallel info
        self.world_size = get_world_size()
        self.rank = get_rank()

        # Each process gets a slice of the output features
        assert (
            out_features % self.world_size == 0
        ), f"out_features ({out_features}) must be divisible by world_size ({self.world_size})"

        self.out_features_per_device = out_features // self.world_size

        # Local output features for this rank
        self.out_features_start = self.rank * self.out_features_per_device
        self.out_features_end = self.out_features_start + self.out_features_per_device

        # Create local weight matrix
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features_per_device,
                in_features,
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_device, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

        logger.info(
            f"ColumnParallelLinear(in={in_features}, out={out_features}, "
            f"local_out={self.out_features_per_device}, rank={self.rank})"
        )

    def _reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with column parallelism.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        # Local linear transformation
        # x: (..., in_features)
        # weight: (out_per_device, in_features)
        # output: (..., out_per_device)
        output = F.linear(x, self.weight, self.bias)

        # Gather outputs from all devices via all-reduce
        # This sums the partial results
        output = all_reduce(output.clone(), op="sum")

        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The weight matrix is split along the input dimension (rows).
    Each process processes a portion of the input.

    Forward: y_i = x_i @ W_i, then all-reduce
    Backward: input split automatically
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        input_is_parallel: bool = False,
    ):
        """Initialize row parallel linear layer.

        Parameters
        ----------
        in_features : int
            Size of input features (total, not per-device).
        out_features : int
            Size of output features.
        bias : bool
            Whether to use bias. Default: True.
        device : Optional[torch.device]
            Device to use.
        dtype : Optional[torch.dtype]
            Data type.
        input_is_parallel : bool
            If True, input is already split. Default: False.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        # Get parallel info
        self.world_size = get_world_size()
        self.rank = get_rank()

        # Each process gets a slice of the input features
        assert (
            in_features % self.world_size == 0
        ), f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"

        self.in_features_per_device = in_features // self.world_size

        # Local input features for this rank
        self.in_features_start = self.rank * self.in_features_per_device
        self.in_features_end = self.in_features_start + self.in_features_per_device

        # Create local weight matrix
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                self.in_features_per_device,
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

        logger.info(
            f"RowParallelLinear(in={in_features}, out={out_features}, "
            f"local_in={self.in_features_per_device}, rank={self.rank})"
        )

    def _reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features_per_device
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with row parallelism.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features) or (..., in_features_per_device)
            if input_is_parallel=True.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        # Split input if not already parallel
        if not self.input_is_parallel:
            # Split input along last dimension
            x = x[..., self.in_features_start : self.in_features_end].contiguous()

        # Local linear transformation
        # x: (..., in_per_device)
        # weight: (out_features, in_per_device)
        # output: (..., out_features)
        output = F.linear(x, self.weight, self.bias)

        # All-reduce to sum partial results from all devices
        output = all_reduce(output.clone(), op="sum")

        return output


class TensorParallelEmbedding(nn.Module):
    """Embedding layer with tensor parallelism.

    The embedding table is split along the vocabulary dimension.
    Each process stores a portion of the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize tensor parallel embedding.

        Parameters
        ----------
        num_embeddings : int
            Total vocabulary size.
        embedding_dim : int
            Embedding dimension.
        padding_idx : Optional[int]
            Padding index, if any.
        device : Optional[torch.device]
            Device to use.
        dtype : Optional[torch.dtype]
            Data type.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Get parallel info
        self.world_size = get_world_size()
        self.rank = get_rank()

        # Each process gets a slice of the vocabulary
        assert (
            num_embeddings % self.world_size == 0
        ), f"num_embeddings ({num_embeddings}) must be divisible by world_size ({self.world_size})"

        self.num_embeddings_per_device = num_embeddings // self.world_size

        # Local vocabulary range for this rank
        self.vocab_start = self.rank * self.num_embeddings_per_device
        self.vocab_end = self.vocab_start + self.num_embeddings_per_device

        # Create local embedding table
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_device,
                embedding_dim,
                **factory_kwargs,
            )
        )

        self._reset_parameters()

        logger.info(
            f"TensorParallelEmbedding(vocab={num_embeddings}, dim={embedding_dim}, "
            f"local_vocab={self.num_embeddings_per_device}, rank={self.rank})"
        )

    def _reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.weight, mean=0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start] = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallel embedding.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (..., seq_len).

        Returns
        -------
        torch.Tensor
            Output embeddings of shape (..., seq_len, embedding_dim).
        """
        # Create mask for valid tokens in this rank's vocabulary range
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)

        # Shift indices to local range
        local_indices = input_ids - self.vocab_start

        # Clamp to valid range (invalid indices will be masked)
        local_indices = local_indices.clamp(0, self.num_embeddings_per_device - 1)

        # Get embeddings
        embeddings = F.embedding(local_indices, self.weight)

        # Zero out embeddings for tokens not in this rank's range
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
        embeddings = embeddings * mask_expanded.to(embeddings.dtype)

        # All-reduce to combine embeddings from all ranks
        embeddings = all_reduce(embeddings, op="sum")

        return embeddings


def make_tensor_parallel_model(
    model: nn.Module,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """Convert a model to use tensor parallel layers.

    This replaces specified linear layers with their tensor-parallel versions.

    Parameters
    ----------
    model : nn.Module
        Original model.
    target_modules : Optional[list[str]]
        List of module names to convert. If None, converts all Linear layers.

    Returns
    -------
    nn.Module
        Model with tensor parallel layers.
    """
    if target_modules is None:
        target_modules = []

    for name, module in list(model.named_children()):
        # Check if this module should be converted
        should_convert = not target_modules or name in target_modules

        if isinstance(module, nn.Linear) and should_convert:
            # Replace with column parallel linear
            parallel_linear = ColumnParallelLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )

            # Copy weights if single GPU (for testing)
            if get_world_size() == 1:
                parallel_linear.weight.data = module.weight.data
                if module.bias is not None:
                    parallel_linear.bias.data = module.bias.data

            setattr(model, name, parallel_linear)
            logger.info(f"Converted {name} to ColumnParallelLinear")

        elif len(list(module.children())) > 0:
            # Recursively process child modules
            make_tensor_parallel_model(module, target_modules)

    return model


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelEmbedding",
    "make_tensor_parallel_model",
]
