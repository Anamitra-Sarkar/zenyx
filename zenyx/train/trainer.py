"""Minimal executable trainer for single-GPU and DDP-style multi-GPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from zenyx.distributed import all_reduce, get_world_size, init_distributed_from_env
from zenyx.memory import MemoryTracker
from zenyx.runtime import Scheduler


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    accumulation_steps: int = 1
    overlap: bool = True
    safety_barrier: bool = False


class Trainer:
    def __init__(self, model: nn.Module, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.ctx = init_distributed_from_env()
        self.model = model.to(self.ctx.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = Scheduler(
            accumulation_steps=self.config.accumulation_steps,
            enable_overlap=self.config.overlap,
        )
        self.memory = MemoryTracker()

    def train_step(self, batch: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> float:
        self.model.train()
        batch = batch.to(self.ctx.device)
        target = target.to(self.ctx.device)

        output = self.scheduler.forward(self.model, batch)
        loss = criterion(output, target)
        self.scheduler.backward(loss, model=self.model, optimizer=self.optimizer)

        detached_loss = loss.detach().clone()
        reduced = all_reduce(detached_loss, op="sum", async_op=False, safety_barrier=self.config.safety_barrier)
        mean_loss = float(reduced.item() / get_world_size())
        self.memory.snapshot()
        return mean_loss

    def fit(self, data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]], criterion: nn.Module, steps: int) -> list[float]:
        losses: list[float] = []
        for i, (batch, target) in enumerate(data_iter):
            if i >= steps:
                break
            losses.append(self.train_step(batch, target, criterion))
        return losses


__all__ = ["Trainer", "TrainerConfig"]
