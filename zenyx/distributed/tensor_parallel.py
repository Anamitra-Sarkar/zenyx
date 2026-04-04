"""Tensor parallel layers with correct forward semantics."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from zenyx.distributed import all_reduce, get_rank, get_world_size

logger = logging.getLogger(__name__)


def _all_gather_last_dim(local: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1 or not dist.is_initialized():
        return local
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=-1)


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = get_world_size()
        self.rank = get_rank()
        if out_features % self.world_size != 0:
            raise ValueError(f"out_features={out_features} must divide world_size={self.world_size}")
        self.out_features_per_device = out_features // self.world_size

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.out_features_per_device, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_device, **factory_kwargs)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.weight, self.bias)
        return _all_gather_last_dim(local, self.world_size)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        input_is_parallel: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.world_size = get_world_size()
        self.rank = get_rank()
        if in_features % self.world_size != 0:
            raise ValueError(f"in_features={in_features} must divide world_size={self.world_size}")

        self.in_features_per_device = in_features // self.world_size
        self.in_features_start = self.rank * self.in_features_per_device
        self.in_features_end = self.in_features_start + self.in_features_per_device

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_device, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias and self.rank == 0 else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features_per_device**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = x[..., self.in_features_start : self.in_features_end].contiguous()
        local = F.linear(x, self.weight, None)
        summed = all_reduce(local, op="sum", async_op=False)
        if self.bias is not None:
            summed = summed + self.bias
        return summed


class TensorParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.world_size = get_world_size()
        self.rank = get_rank()
        if num_embeddings % self.world_size != 0:
            raise ValueError("num_embeddings must divide world_size")

        self.num_embeddings_per_device = num_embeddings // self.world_size
        self.vocab_start = self.rank * self.num_embeddings_per_device
        self.vocab_end = self.vocab_start + self.num_embeddings_per_device

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_device, embedding_dim, device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_indices = (input_ids - self.vocab_start).clamp(0, self.num_embeddings_per_device - 1)
        embeddings = F.embedding(local_indices, self.weight)
        embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        return all_reduce(embeddings, op="sum", async_op=False)


def make_tensor_parallel_model(model: nn.Module, target_modules: Optional[list[str]] = None) -> nn.Module:
    target_modules = target_modules or []
    for name, module in list(model.named_children()):
        should_convert = not target_modules or name in target_modules
        if isinstance(module, nn.Linear) and should_convert:
            tp = ColumnParallelLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            if get_world_size() == 1:
                tp.weight.data.copy_(module.weight.data)
                if module.bias is not None and tp.bias is not None:
                    tp.bias.data.copy_(module.bias.data)
            setattr(model, name, tp)
        elif len(list(module.children())) > 0:
            make_tensor_parallel_model(module, target_modules)
    return model


__all__ = ["ColumnParallelLinear", "RowParallelLinear", "TensorParallelEmbedding", "make_tensor_parallel_model"]
