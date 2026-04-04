"""Basic pipeline parallel implementation (foundation).

This module provides foundational pipeline parallel support:
- Split model into stages
- Assign stages to devices
- Micro-batch scheduling
- Forward/backward pass across stages

This is a minimal but correct implementation focused on correctness.
Performance optimizations (1F1B, interleaving) are for future phases.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from zenyx.distributed import broadcast, get_rank, get_world_size

logger = logging.getLogger(__name__)


class PipelineStage(nn.Module):
    """Represents a single stage in a pipeline parallel model.

    A stage contains a subset of the model's layers and runs on
    a specific device.
    """

    def __init__(
        self,
        stage_id: int,
        num_stages: int,
        layers: list[nn.Module],
        input_shape: tuple[int, ...],
        device: torch.device,
    ):
        """Initialize pipeline stage.

        Parameters
        ----------
        stage_id : int
            This stage's ID (0 to num_stages-1).
        num_stages : int
            Total number of stages.
        layers : list[nn.Module]
            List of layers in this stage.
        input_shape : tuple[int, ...]
            Expected input shape (for first stage).
        device : torch.device
            Device to run this stage on.
        """
        super().__init__()

        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers = nn.ModuleList(layers)
        self.input_shape = input_shape
        self.device = device

        # Move all layers to device
        self.to(device)

        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1

        logger.info(
            f"PipelineStage(id={stage_id}, layers={len(layers)}, "
            f"device={device}, first={self.is_first_stage}, last={self.is_last_stage})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this stage.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class PipelineParallelModel(nn.Module):
    """Pipeline parallel model wrapper.

    Splits a model into stages and distributes them across devices.
    Supports micro-batching for pipeline parallelism.

    Usage:
        >>> model = create_model()
        >>> pp_model = PipelineParallelModel(model, num_stages=4)
        >>> output = pp_model.forward_microbatches(inputs, num_microbatches=8)
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: Optional[int] = None,
        sample_input: Optional[torch.Tensor] = None,
        device_ids: Optional[list[int]] = None,
    ):
        """Initialize pipeline parallel model.

        Parameters
        ----------
        model : nn.Module
            Model to parallelize.
        num_stages : Optional[int]
            Number of pipeline stages. If None, uses world_size.
        sample_input : Optional[torch.Tensor]
            Sample input for shape inference.
        device_ids : Optional[list[int]]
            Device IDs to use. If None, uses all available.
        """
        super().__init__()

        self.world_size = get_world_size()
        self.rank = get_rank()

        # Determine number of stages
        if num_stages is None:
            num_stages = self.world_size
        self.num_stages = num_stages

        # Determine device IDs
        if device_ids is None:
            device_ids = list(range(min(num_stages, self.world_size)))
        self.device_ids = device_ids

        # Get current device
        if torch.cuda.is_available() and len(device_ids) > 0:
            self.device = torch.device(f"cuda:{device_ids[self.rank % len(device_ids)]}")
        else:
            self.device = torch.device("cpu")

        # Split model into stages
        self.stages = self._split_model_into_stages(model, sample_input)

        # Current stage for this rank
        self.current_stage_id = self.rank % num_stages
        self.current_stage = self.stages[self.current_stage_id]

        logger.info(
            f"PipelineParallelModel(stages={num_stages}, "
            f"rank={self.rank}, current_stage={self.current_stage_id})"
        )

    def _split_model_into_stages(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor],
    ) -> list[PipelineStage]:
        """Split model into pipeline stages.

        Parameters
        ----------
        model : nn.Module
            Model to split.
        sample_input : Optional[torch.Tensor]
            Sample input for shape inference.

        Returns
        -------
        list[PipelineStage]
            List of pipeline stages.
        """
        # Get all child modules (layers)
        layers = list(model.children())

        if not layers:
            raise ValueError("Model has no child modules to split")

        # Distribute layers across stages
        num_layers = len(layers)
        layers_per_stage = (num_layers + self.num_stages - 1) // self.num_stages

        stages = []
        for stage_id in range(self.num_stages):
            start_idx = stage_id * layers_per_stage
            end_idx = min(start_idx + layers_per_stage, num_layers)

            if start_idx >= num_layers:
                # No layers for this stage (more stages than layers)
                stage_layers = [nn.Identity()]
            else:
                stage_layers = layers[start_idx:end_idx]

            # Infer input shape from sample input or use default
            input_shape = (1, 1)  # Default
            if sample_input is not None and stage_id == 0:
                input_shape = tuple(sample_input.shape)

            stage = PipelineStage(
                stage_id=stage_id,
                num_stages=self.num_stages,
                layers=stage_layers,
                input_shape=input_shape,
                device=self.device,
            )
            stages.append(stage)

        return stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (single batch, local stage only).

        For pipeline parallel forward, use forward_microbatches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (only used for first stage).

        Returns
        -------
        torch.Tensor
            Output tensor (only valid for last stage).
        """
        if self.current_stage_id == 0:
            # First stage receives input
            output = self.current_stage(x.to(self.device))
        else:
            # Other stages receive from previous stage
            # In real implementation, this would be via IPC/RPC
            # For now, we assume input is provided
            output = self.current_stage(x)

        return output

    def forward_microbatches(
        self,
        inputs: list[torch.Tensor],
        num_microbatches: int,
    ) -> list[torch.Tensor]:
        """Forward pass with micro-batching.

        This implements a simple GPipe-style schedule:
        - All micro-batches go through forward pass
        - Stages process sequentially

        Parameters
        ----------
        inputs : list[torch.Tensor]
            List of micro-batch inputs.
        num_microbatches : int
            Number of micro-batches.

        Returns
        -------
        list[torch.Tensor]
            List of micro-batch outputs.
        """
        if self.current_stage_id != 0:
            # Non-first stages need to receive inputs
            # Simplified: assume inputs are pre-distributed
            raise NotImplementedError(
                "Multi-stage pipeline requires RPC/IPC for inter-stage communication"
            )

        outputs = []
        for i, micro_input in enumerate(inputs[:num_microbatches]):
            logger.debug(f"Processing micro-batch {i}/{num_microbatches}")
            output = self.current_stage(micro_input.to(self.device))
            outputs.append(output)

        return outputs

    def backward_microbatches(
        self,
        losses: list[torch.Tensor],
    ) -> None:
        """Backward pass with micro-batching.

        Parameters
        ----------
        losses : list[torch.Tensor]
            List of loss tensors for each micro-batch.
        """
        for i, loss in enumerate(losses):
            logger.debug(f"Backward micro-batch {i}/{len(losses)}")
            loss.backward(retain_graph=(i < len(losses) - 1))


def create_pipeline_from_sequential(
    model: nn.Sequential,
    num_stages: int,
    device_ids: Optional[list[int]] = None,
) -> list[PipelineStage]:
    """Create pipeline stages from a Sequential model.

    Parameters
    ----------
    model : nn.Sequential
        Sequential model to split.
    num_stages : int
        Number of pipeline stages.
    device_ids : Optional[list[int]]
        Device IDs to use.

    Returns
    -------
    list[PipelineStage]
        List of pipeline stages.
    """
    layers = list(model.children())
    num_layers = len(layers)

    if num_layers < num_stages:
        logger.warning(
            f"More stages ({num_stages}) than layers ({num_layers}). "
            "Some stages will be empty."
        )

    layers_per_stage = (num_layers + num_stages - 1) // num_stages
    stages = []

    for stage_id in range(num_stages):
        start_idx = stage_id * layers_per_stage
        end_idx = min(start_idx + layers_per_stage, num_layers)

        if start_idx >= num_layers:
            stage_layers = [nn.Identity()]
        else:
            stage_layers = layers[start_idx:end_idx]

        device = torch.device(f"cuda:{device_ids[stage_id]}") if (
            device_ids and torch.cuda.is_available()
        ) else torch.device("cpu")

        stage = PipelineStage(
            stage_id=stage_id,
            num_stages=num_stages,
            layers=stage_layers,
            input_shape=(1, 1),  # Placeholder
            device=device,
        )
        stages.append(stage)

    return stages


class SimpleGPipeSchedule:
    """Simple GPipe-style pipeline scheduler.

    Implements basic forward-then-backward schedule:
    1. Forward pass all micro-batches
    2. Backward pass all micro-batches

    This is not optimal (1F1B is better) but is correct and simple.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        num_microbatches: int,
    ):
        """Initialize GPipe scheduler.

        Parameters
        ----------
        stages : list[PipelineStage]
            Pipeline stages.
        num_microbatches : int
            Number of micro-batches.
        """
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.num_stages = len(stages)

        logger.info(
            f"SimpleGPipeSchedule(stages={self.num_stages}, "
            f"microbatches={num_microbatches})"
        )

    def step(
        self,
        inputs: list[torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        targets: list[torch.Tensor],
    ) -> list[float]:
        """Execute one training step.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Micro-batch inputs.
        loss_fn : Callable
            Loss function.
        targets : list[torch.Tensor]
            Micro-batch targets.

        Returns
        -------
        list[float]
            Loss values for each micro-batch.
        """
        # Forward pass all micro-batches
        outputs = []
        for i, inp in enumerate(inputs[: self.num_microbatches]):
            x = inp
            for stage in self.stages:
                x = stage(x)
            outputs.append(x)

        # Compute losses - keep both tensor and float value
        loss_tensors = []
        loss_values = []
        for i, (output, target) in enumerate(
            zip(outputs, targets[: self.num_microbatches])
        ):
            loss_tensor = loss_fn(output, target)
            loss_tensors.append(loss_tensor)
            loss_values.append(loss_tensor.item())

        # Backward pass all micro-batches
        for i, loss_tensor in enumerate(loss_tensors):
            loss_tensor.backward(retain_graph=(i < len(loss_tensors) - 1))

        return loss_values


__all__ = [
    "PipelineStage",
    "PipelineParallelModel",
    "create_pipeline_from_sequential",
    "SimpleGPipeSchedule",
]
