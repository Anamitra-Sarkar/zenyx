"""Zenyx training subsystem — mixed precision, pipeline, checkpointing, loop."""

from __future__ import annotations

from zenyx.train.checkpoint import AsyncCheckpointer
from zenyx.train.loop import wrap
from zenyx.train.mixed_prec import (
    FP8ActivationStorage,
    FP8CheckpointFunction,
    fp8_checkpoint,
)
from zenyx.train.pipeline import BraidedPipeline, ScheduleStep, StepAction
from zenyx.train.trainer import Trainer, train
from zenyx.train.lr_schedule import CosineWithWarmup
from zenyx.train.grad_scaler import ZenyxGradScaler
from zenyx.train.distributed_setup import (
    auto_init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    cleanup,
)
from zenyx.train.activation_checkpoint import (
    CheckpointedBlock,
    selective_checkpoint_wrapper,
)

__all__ = [
    # mixed_prec
    "FP8ActivationStorage",
    "FP8CheckpointFunction",
    "fp8_checkpoint",
    # pipeline
    "BraidedPipeline",
    "ScheduleStep",
    "StepAction",
    # checkpoint
    "AsyncCheckpointer",
    # loop
    "wrap",
    # trainer (Phase 4)
    "Trainer",
    "train",
    # lr_schedule
    "CosineWithWarmup",
    # grad_scaler
    "ZenyxGradScaler",
    # distributed_setup
    "auto_init_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "cleanup",
    # activation_checkpoint
    "CheckpointedBlock",
    "selective_checkpoint_wrapper",
]
