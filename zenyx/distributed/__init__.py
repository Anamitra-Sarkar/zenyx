"""Distributed training utilities — FSDP and collective operations."""

from __future__ import annotations

from zenyx.distributed.communication import (
    CollectiveGroup,
    CommunicationError,
    all_reduce,
    barrier,
    broadcast,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
)
from zenyx.distributed.fsdp_wrapper import FSDPWrapper
from zenyx.distributed.pipeline_parallel import (
    PipelineParallelModel,
    PipelineStage,
    SimpleGPipeSchedule,
    create_pipeline_from_sequential,
)
from zenyx.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelEmbedding,
    make_tensor_parallel_model,
)

__all__ = [
    "FSDPWrapper",
    "all_reduce",
    "broadcast",
    "barrier",
    "get_world_size",
    "get_rank",
    "init_process_group",
    "destroy_process_group",
    "CollectiveGroup",
    "CommunicationError",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelEmbedding",
    "make_tensor_parallel_model",
    "PipelineStage",
    "PipelineParallelModel",
    "create_pipeline_from_sequential",
    "SimpleGPipeSchedule",
]
