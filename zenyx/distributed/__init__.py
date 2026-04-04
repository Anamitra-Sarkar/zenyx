"""Distributed training utilities — FSDP and collective operations."""

from __future__ import annotations

from zenyx.distributed.communication import (
    AsyncCollectiveHandle,
    CollectiveGroup,
    CommunicationError,
    DistributedContext,
    all_reduce,
    barrier,
    broadcast,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_distributed_from_env,
    init_process_group,
)
from zenyx.distributed.fsdp_wrapper import FSDPWrapper
from zenyx.distributed.pipeline_parallel import (
    DistributedPipelineEngine,
    Partial1F1BSchedule,
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
    "init_distributed_from_env",
    "destroy_process_group",
    "DistributedContext",
    "AsyncCollectiveHandle",
    "CollectiveGroup",
    "CommunicationError",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelEmbedding",
    "make_tensor_parallel_model",
    "PipelineStage",
    "DistributedPipelineEngine",
    "PipelineParallelModel",
    "create_pipeline_from_sequential",
    "SimpleGPipeSchedule",
    "Partial1F1BSchedule",
]
