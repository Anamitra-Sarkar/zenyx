"""Distributed communication primitives with explicit async handle management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CommunicationError(Exception):
    pass


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


class AsyncCollectiveHandle:
    """Track an async collective + its output tensor."""

    def __init__(self, work: dist.Work, tensor: torch.Tensor, name: str):
        self.work = work
        self.tensor = tensor
        self.name = name
        self._done = False

    def wait(self) -> torch.Tensor:
        if not self._done:
            self.work.wait()
            self._done = True
        return self.tensor

    def is_completed(self) -> bool:
        if self._done:
            return True
        if hasattr(self.work, "is_completed") and self.work.is_completed():
            self._done = True
            return True
        return False


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    return dist.get_world_size(group) if dist.is_initialized() else 1


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    return dist.get_rank(group) if dist.is_initialized() else 0


def _ensure_tensor_ready(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def all_reduce(
    tensor: torch.Tensor,
    op: str = "sum",
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    safety_barrier: bool = False,
) -> torch.Tensor | AsyncCollectiveHandle:
    if not dist.is_initialized() or get_world_size(group) == 1:
        return tensor

    op_map = {
        "sum": dist.ReduceOp.SUM,
        "prod": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "avg": dist.ReduceOp.AVG,
    }
    if op not in op_map:
        raise ValueError(f"Unknown reduce op: {op}")

    tensor = _ensure_tensor_ready(tensor)
    if safety_barrier:
        dist.barrier(group=group)

    try:
        work = dist.all_reduce(tensor, op=op_map[op], group=group, async_op=async_op)
        if async_op:
            return AsyncCollectiveHandle(work=work, tensor=tensor, name=f"all_reduce:{op}")
        return tensor
    except Exception as exc:
        if async_op:
            logger.warning("Async all_reduce failed (%s). Falling back to sync mode.", exc)
            dist.all_reduce(tensor, op=op_map[op], group=group, async_op=False)
            return tensor
        raise CommunicationError(f"All-reduce failed: {exc}") from exc


def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    safety_barrier: bool = False,
) -> torch.Tensor | AsyncCollectiveHandle:
    if not dist.is_initialized() or get_world_size(group) == 1:
        return tensor

    tensor = _ensure_tensor_ready(tensor)
    if safety_barrier:
        dist.barrier(group=group)

    try:
        work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
        if async_op:
            return AsyncCollectiveHandle(work=work, tensor=tensor, name=f"broadcast:{src}")
        return tensor
    except Exception as exc:
        if async_op:
            logger.warning("Async broadcast failed (%s). Falling back to sync mode.", exc)
            dist.broadcast(tensor, src=src, group=group, async_op=False)
            return tensor
        raise CommunicationError(f"Broadcast failed: {exc}") from exc


def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    if dist.is_initialized() and get_world_size(group) > 1:
        dist.barrier(group=group)


def init_process_group(backend: str = "nccl", timeout_seconds: int = 1800) -> None:
    if dist.is_initialized():
        logger.warning("Process group already initialized")
        return
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    logger.info("Initialized process group: rank=%s world_size=%s", get_rank(), get_world_size())


def init_distributed_from_env(backend: Optional[str] = None, timeout_seconds: int = 1800) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if world_size > 1 and not dist.is_initialized():
        init_process_group(backend=backend, timeout_seconds=timeout_seconds)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return DistributedContext(
        rank=get_rank(),
        world_size=get_world_size(),
        local_rank=local_rank,
        device=device,
    )


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class CollectiveGroup:
    def __init__(self, world_size: int, rank: int, group: Optional[dist.ProcessGroup] = None):
        self.world_size = world_size
        self.rank = rank
        self.group = group

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
        async_op: bool = False,
        safety_barrier: bool = False,
    ) -> torch.Tensor | AsyncCollectiveHandle:
        return all_reduce(
            tensor,
            op=op,
            group=self.group,
            async_op=async_op,
            safety_barrier=safety_barrier,
        )

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        async_op: bool = False,
        safety_barrier: bool = False,
    ) -> torch.Tensor | AsyncCollectiveHandle:
        return broadcast(
            tensor,
            src=src,
            group=self.group,
            async_op=async_op,
            safety_barrier=safety_barrier,
        )

    def barrier(self) -> None:
        barrier(self.group)


__all__ = [
    "all_reduce",
    "broadcast",
    "barrier",
    "get_world_size",
    "get_rank",
    "init_process_group",
    "init_distributed_from_env",
    "destroy_process_group",
    "CollectiveGroup",
    "CommunicationError",
    "AsyncCollectiveHandle",
    "DistributedContext",
]
