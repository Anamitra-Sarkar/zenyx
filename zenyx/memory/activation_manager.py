"""Selective activation checkpointing policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPolicy:
    checkpoint_attention: bool = False
    checkpoint_ffn: bool = True
    min_trainable_params: int = 50_000


class ActivationManager:
    def __init__(self, use_checkpoint: bool = True, policy: CheckpointPolicy | None = None):
        self.use_checkpoint = use_checkpoint
        self.policy = policy or CheckpointPolicy()

    def _should_checkpoint_module(self, name: str, module: nn.Module) -> bool:
        if not self.use_checkpoint:
            return False
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if trainable_params < self.policy.min_trainable_params:
            return False

        lname = name.lower()
        if self.policy.checkpoint_attention and ("attention" in lname or "attn" in lname):
            return True
        if self.policy.checkpoint_ffn and ("mlp" in lname or "ffn" in lname or "feed_forward" in lname):
            return True
        return False

    def checkpoint_module(self, module: nn.Module) -> Callable[..., torch.Tensor]:
        if not self.use_checkpoint:
            return module.forward

        def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
            if not any(torch.is_tensor(a) and a.requires_grad for a in args):
                return module.forward(*args, **kwargs)
            return checkpoint(module.forward, *args, use_reentrant=False, **kwargs)

        return wrapped

    @staticmethod
    def hook_into_model(model: nn.Module, use_checkpoint: bool = True, policy: CheckpointPolicy | None = None) -> None:
        manager = ActivationManager(use_checkpoint=use_checkpoint, policy=policy)
        if not manager.use_checkpoint:
            return

        for name, module in model.named_modules():
            if not manager._should_checkpoint_module(name, module):
                continue
            original = module.forward

            def make_forward(orig_forward: Callable[..., Any]) -> Callable[..., Any]:
                def checkpoint_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                    if not any(torch.is_tensor(a) and a.requires_grad for a in args):
                        return orig_forward(*args, **kwargs)
                    return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

                return checkpoint_forward

            module.forward = make_forward(original)

    @staticmethod
    def estimate_memory_saving(num_layers: int, hidden_dim: int, seq_len: int, batch_size: int) -> dict[str, float]:
        bytes_per_param = 2
        attn_size = batch_size * seq_len * hidden_dim * bytes_per_param
        ffn_size = batch_size * seq_len * (hidden_dim * 4) * bytes_per_param
        attn_save = attn_size * num_layers / 1e6
        ffn_save = ffn_size * num_layers / 1e6
        return {
            "attention_memory_mb": attn_save,
            "ffn_memory_mb": ffn_save,
            "total_memory_gb": (attn_save + ffn_save) / 1e3,
        }


__all__ = ["ActivationManager", "CheckpointPolicy"]
