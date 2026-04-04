"""Distributed communication primitives.

This module provides wrappers around torch.distributed operations:
- all_reduce: Sum tensors across all processes
- broadcast: Broadcast tensor from source to all processes

These wrappers handle:
- Device placement
- Tensor consistency checks
- Process group management
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CommunicationError(Exception):
    """Raised when a communication operation fails."""

    pass


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get world size safely.

    Parameters
    ----------
    group : Optional[dist.ProcessGroup]
        Process group. If None, use default process group.

    Returns
    -------
    int
        World size (number of processes).
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get rank safely.

    Parameters
    ----------
    group : Optional[dist.ProcessGroup]
        Process group. If None, use default process group.

    Returns
    -------
    int
        Current process rank.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def all_reduce(
    tensor: torch.Tensor,
    op: str = "sum",
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor | dist.Work:
    """Perform all-reduce operation.

    All processes contribute a tensor, and the result is the element-wise
    reduction (sum by default) across all processes.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor. Will contain the result after operation.
    op : str
        Reduction operation: "sum", "prod", "min", "max", "avg".
        Default: "sum".
    group : Optional[dist.ProcessGroup]
        Process group. If None, use default process group.
    async_op : bool
        If True, return immediately with a Work object.
        Default: False (blocking).

    Returns
    -------
    torch.Tensor | dist.Work
        The same tensor (modified in-place) if async_op=False,
        or a Work object if async_op=True.

    Raises
    ------
    CommunicationError
        If distributed is not initialized or operation fails.
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, skipping all_reduce")
        return tensor if not async_op else dist.no_work()

    # Map string op to torch.distributed ReduceOp
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "prod": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "avg": dist.ReduceOp.AVG,
    }

    if op not in op_map:
        raise ValueError(f"Unknown reduce op: {op}. Choose from {list(op_map.keys())}")

    reduce_op = op_map[op]

    # Ensure tensor is on CUDA if available
    if tensor.device.type != "cuda" and torch.cuda.is_available():
        logger.debug("Moving tensor to CUDA for all_reduce")
        tensor = tensor.cuda()

    # Ensure tensor is contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    try:
        work = dist.all_reduce(tensor, op=reduce_op, group=group, async_op=async_op)
        if async_op:
            return work
        return tensor
    except Exception as e:
        raise CommunicationError(f"All-reduce failed: {e}") from e


def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor | dist.Work:
    """Broadcast tensor from source process to all other processes.

    Parameters
    ----------
    tensor : torch.Tensor
        On src: input tensor to broadcast.
        On other ranks: output tensor (must have same shape).
    src : int
        Source rank to broadcast from. Default: 0.
    group : Optional[dist.ProcessGroup]
        Process group. If None, use default process group.
    async_op : bool
        If True, return immediately with a Work object.
        Default: False (blocking).

    Returns
    -------
    torch.Tensor | dist.Work
        The same tensor (modified in-place on non-src ranks) if async_op=False,
        or a Work object if async_op=True.

    Raises
    ------
    CommunicationError
        If distributed is not initialized or operation fails.
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, skipping broadcast")
        return tensor if not async_op else dist.no_work()

    # Ensure tensor is on CUDA if available
    if tensor.device.type != "cuda" and torch.cuda.is_available():
        logger.debug("Moving tensor to CUDA for broadcast")
        tensor = tensor.cuda()

    # Ensure tensor is contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    try:
        work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
        if async_op:
            return work
        return tensor
    except Exception as e:
        raise CommunicationError(f"Broadcast failed: {e}") from e


def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    """Synchronize all processes.

    Parameters
    ----------
    group : Optional[dist.ProcessGroup]
        Process group. If None, use default process group.
    """
    if not dist.is_initialized():
        return
    dist.barrier(group=group)


def init_process_group(
    backend: str = "nccl",
    timeout_seconds: int = 1800,
) -> None:
    """Initialize distributed process group.

    Parameters
    ----------
    backend : str
        Backend to use: "nccl" (GPU), "gloo" (CPU), "mpi".
        Default: "nccl".
    timeout_seconds : int
        Timeout for collective operations. Default: 1800 (30 min).

    Raises
    ------
    RuntimeError
        If already initialized.
    """
    if dist.is_initialized():
        logger.warning("Process group already initialized")
        return

    timeout = dist.default_pg_timeout if dist.default_pg_timeout else timeout_seconds

    try:
        dist.init_process_group(backend=backend, timeout=__import__("datetime").timedelta(seconds=timeout))
        logger.info(
            f"Initialized process group: rank={get_rank()}, world_size={get_world_size()}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize process group: {e}") from e


def destroy_process_group() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed process group")


class CollectiveGroup:
    """Manages a group of processes for collective operations.

    This provides a cleaner interface for repeated collective operations
    on the same set of devices.

    Usage:
        >>> group = CollectiveGroup(world_size=2)
        >>> result = group.all_reduce(tensor)
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """Initialize collective group.

        Parameters
        ----------
        world_size : int
            Number of processes in group.
        rank : int
            This process's rank.
        group : Optional[dist.ProcessGroup]
            PyTorch process group. If None, use default.
        """
        self.world_size = world_size
        self.rank = rank
        self.group = group
        self._logger = logging.getLogger(f"{__name__}.CollectiveGroup")

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
        async_op: bool = False,
    ) -> torch.Tensor | dist.Work:
        """Perform all-reduce within this group.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.
        op : str
            Reduction operation. Default: "sum".
        async_op : bool
            Async operation. Default: False.

        Returns
        -------
        torch.Tensor | dist.Work
            Result tensor or Work object.
        """
        return all_reduce(tensor, op=op, group=self.group, async_op=async_op)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        async_op: bool = False,
    ) -> torch.Tensor | dist.Work:
        """Perform broadcast within this group.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to broadcast.
        src : int
            Source rank. Default: 0.
        async_op : bool
            Async operation. Default: False.

        Returns
        -------
        torch.Tensor | dist.Work
            Result tensor or Work object.
        """
        return broadcast(tensor, src=src, group=self.group, async_op=async_op)

    def barrier(self) -> None:
        """Synchronize all processes in group."""
        barrier(self.group)

    def __repr__(self) -> str:
        return f"CollectiveGroup(world_size={self.world_size}, rank={self.rank})"


__all__ = [
    "all_reduce",
    "broadcast",
    "barrier",
    "get_world_size",
    "get_rank",
    "init_process_group",
    "destroy_process_group",
    "CollectiveGroup",
    "CommunicationError",
]
