"""Pipeline parallel building blocks and schedules (GPipe + partial 1F1B + distributed stage runtime)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

import torch.distributed as dist

from zenyx.distributed.communication import get_rank, get_world_size

logger = logging.getLogger(__name__)


class PipelineStage(nn.Module):
    def __init__(self, stage_id: int, num_stages: int, layers: list[nn.Module], input_shape: tuple[int, ...], device: torch.device):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers = nn.ModuleList(layers)
        self.input_shape = input_shape
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DistributedPipelineEngine:
    """One stage per rank pipeline runtime using dist.send/recv."""

    def __init__(
        self,
        stage: PipelineStage,
        micro_batch_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ):
        self.stage = stage
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.prev_rank = self.rank - 1 if self.rank > 0 else None
        self.next_rank = self.rank + 1 if self.rank < self.world_size - 1 else None
        self.micro_batch_shape = micro_batch_shape
        self.dtype = dtype
        self.device = stage.device

    def run_step(
        self,
        input_batch: Optional[torch.Tensor],
        target_batch: Optional[torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float | None:
        optimizer.zero_grad(set_to_none=True)

        if self.prev_rank is None:
            if input_batch is None:
                raise ValueError("First stage requires input_batch")
            activation_in = input_batch.to(self.device)
        else:
            activation_in = torch.zeros(self.micro_batch_shape, device=self.device, dtype=self.dtype)
            dist.recv(activation_in, src=self.prev_rank)
            activation_in.requires_grad_(True)

        activation_out = self.stage(activation_in)

        if self.next_rank is not None:
            dist.send(activation_out.detach(), dst=self.next_rank)

        if self.next_rank is None:
            if target_batch is None:
                raise ValueError("Last stage requires target_batch")
            loss = loss_fn(activation_out, target_batch.to(self.device))
            loss.backward()
            grad_to_prev = activation_in.grad
            if self.prev_rank is not None and grad_to_prev is not None:
                dist.send(grad_to_prev.detach(), dst=self.prev_rank)
            optimizer.step()
            return float(loss.item())

        grad_from_next = torch.zeros_like(activation_out)
        dist.recv(grad_from_next, src=self.next_rank)
        activation_out.backward(grad_from_next)

        grad_to_prev = activation_in.grad
        if self.prev_rank is not None and grad_to_prev is not None:
            dist.send(grad_to_prev.detach(), dst=self.prev_rank)

        optimizer.step()
        return None


class PipelineParallelModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_stages: Optional[int] = None,
        sample_input: Optional[torch.Tensor] = None,
        device_ids: Optional[list[int]] = None,
    ):
        super().__init__()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.num_stages = num_stages or self.world_size
        if device_ids is None:
            device_ids = list(range(max(1, min(self.num_stages, torch.cuda.device_count() or 1))))
        self.device_ids = device_ids
        self.device = torch.device(f"cuda:{device_ids[self.rank % len(device_ids)]}") if torch.cuda.is_available() else torch.device("cpu")

        self.stages = self._split_model_into_stages(model, sample_input)
        self.current_stage_id = self.rank % self.num_stages
        self.current_stage = self.stages[self.current_stage_id]

    def _split_model_into_stages(self, model: nn.Module, sample_input: Optional[torch.Tensor]) -> list[PipelineStage]:
        layers = list(model.children())
        if not layers:
            raise ValueError("Model has no child modules to split")
        layers_per_stage = (len(layers) + self.num_stages - 1) // self.num_stages
        stages: list[PipelineStage] = []
        for stage_id in range(self.num_stages):
            start_idx = stage_id * layers_per_stage
            end_idx = min(start_idx + layers_per_stage, len(layers))
            stage_layers = layers[start_idx:end_idx] if start_idx < len(layers) else [nn.Identity()]
            in_shape = tuple(sample_input.shape) if sample_input is not None and stage_id == 0 else (1, 1)
            stages.append(PipelineStage(stage_id, self.num_stages, stage_layers, in_shape, self.device))
        return stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.current_stage(x.to(self.device))


def create_pipeline_from_sequential(model: nn.Sequential, num_stages: int, device_ids: Optional[list[int]] = None) -> list[PipelineStage]:
    layers = list(model.children())
    layers_per_stage = (len(layers) + num_stages - 1) // num_stages
    stages = []
    for stage_id in range(num_stages):
        start_idx = stage_id * layers_per_stage
        end_idx = min(start_idx + layers_per_stage, len(layers))
        stage_layers = layers[start_idx:end_idx] if start_idx < len(layers) else [nn.Identity()]
        device = torch.device(f"cuda:{device_ids[stage_id]}") if device_ids and torch.cuda.is_available() else torch.device("cpu")
        stages.append(PipelineStage(stage_id, num_stages, stage_layers, (1, 1), device))
    return stages


class SimpleGPipeSchedule:
    def __init__(self, stages: list[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_microbatches = num_microbatches

    def step(self, inputs: list[torch.Tensor], loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], targets: list[torch.Tensor]) -> list[float]:
        outputs = []
        for inp in inputs[: self.num_microbatches]:
            x = inp
            for stage in self.stages:
                x = stage(x)
            outputs.append(x)

        losses = [loss_fn(o, t) for o, t in zip(outputs, targets[: self.num_microbatches])]
        for i, loss in enumerate(losses):
            loss.backward(retain_graph=(i < len(losses) - 1))
        return [float(l.item()) for l in losses]


class Partial1F1BSchedule:
    """Single-process simulator for deadlock-safe partial 1F1B ordering."""

    def __init__(self, stages: list[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches

    def step(self, inputs: list[torch.Tensor], loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], targets: list[torch.Tensor]) -> list[float]:
        if len(inputs) < self.num_microbatches or len(targets) < self.num_microbatches:
            raise ValueError("inputs/targets shorter than num_microbatches")

        losses: dict[int, torch.Tensor] = {}
        loss_values: list[float] = []

        warmup = self.num_stages - 1

        for micro in range(min(warmup, self.num_microbatches)):
            x = inputs[micro]
            for stage in self.stages:
                x = stage(x)
            loss = loss_fn(x, targets[micro])
            losses[micro] = loss
            loss_values.append(float(loss.item()))

        for micro in range(warmup, self.num_microbatches):
            x = inputs[micro]
            for stage in self.stages:
                x = stage(x)
            loss = loss_fn(x, targets[micro])
            losses[micro] = loss
            loss_values.append(float(loss.item()))

            back_micro = micro - warmup
            losses[back_micro].backward(retain_graph=True)

        for back_micro in range(max(0, self.num_microbatches - warmup), self.num_microbatches):
            losses[back_micro].backward(retain_graph=(back_micro < self.num_microbatches - 1))

        return loss_values


__all__ = [
    "PipelineStage",
    "DistributedPipelineEngine",
    "PipelineParallelModel",
    "create_pipeline_from_sequential",
    "SimpleGPipeSchedule",
    "Partial1F1BSchedule",
]
