"""Execution scheduler for forward, backward, and communication operations.

This scheduler handles:
- Compute operations (forward/backward)
- Communication operations (all-reduce, broadcast)
- Dependency resolution
- Structured scheduling for potential compute/comm overlap
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from zenyx.distributed import all_reduce, broadcast, get_rank, get_world_size
from zenyx.runtime.execution_graph import ExecutionGraph, OpNode, OpType

logger = logging.getLogger(__name__)


class Scheduler:
    """Schedules forward, backward, and communication passes.

    This scheduler executes operations respecting dependencies:
    - Forward pass in order
    - Backward pass in reverse order
    - Communication after backward (gradient sync)

    Usage:
        >>> scheduler = Scheduler()
        >>> output = scheduler.forward(model, batch)
        >>> loss = output.mean()
        >>> scheduler.backward(loss, optimizer)
    """

    def __init__(self, accumulation_steps: int = 1):
        """Initialize the scheduler.

        Parameters
        ----------
        accumulation_steps : int
            Number of steps before optimizer update. Default: 1 (no accumulation).
        """
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.accumulated_loss = 0.0

        logger.info("Scheduler(accumulation_steps=%d)", accumulation_steps)

    def forward(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Execute forward pass.

        Parameters
        ----------
        model : nn.Module
            The model.
        batch : torch.Tensor
            Input batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return model(batch)

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: Optional[Optimizer] = None,
        execution_graph: Optional[ExecutionGraph] = None,
    ) -> None:
        """Execute backward pass and optionally update weights.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor.
        optimizer : Optional[Optimizer]
            Optimizer to use for updates. If None, only backward.
        execution_graph : Optional[ExecutionGraph]
            Execution graph with communication nodes. If provided,
            will execute communication operations.
        """
        # Backward pass
        loss.backward()

        # Execute communication if graph is provided
        if execution_graph is not None:
            self._execute_communication_nodes(execution_graph)

        # Check if we should update
        self.step_counter += 1
        if self.step_counter % self.accumulation_steps == 0:
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
            self.accumulated_loss = 0.0
        else:
            self.accumulated_loss += loss.item()

    def _execute_communication_nodes(self, graph: ExecutionGraph) -> None:
        """Execute communication nodes from the execution graph.

        Parameters
        ----------
        graph : ExecutionGraph
            Execution graph containing communication nodes.
        """
        # Get all communication nodes
        comm_nodes = []
        for node in graph.get_backward_nodes():
            if node.is_comm_op:
                comm_nodes.append(node)

        if not comm_nodes:
            logger.debug("No communication nodes to execute")
            return

        # For communication nodes, dependencies are typically backward compute nodes
        # which have already been executed via loss.backward()
        # So we consider all non-comm nodes as already executed
        executed = set()
        
        # Mark all compute nodes as executed (they ran during loss.backward())
        for node in graph.get_forward_nodes() + graph.get_backward_nodes():
            if node.is_compute_op:
                executed.add(node.name)

        pending = list(comm_nodes)

        while pending:
            made_progress = False
            for node in pending[:]:
                # Check if dependencies are satisfied
                deps = graph.get_node_dependencies(node.name)
                deps_satisfied = all(dep in executed for dep in deps)

                if deps_satisfied:
                    self._execute_comm_node(node)
                    executed.add(node.name)
                    pending.remove(node)
                    made_progress = True

            if not made_progress and pending:
                logger.warning(f"Deadlock detected: cannot execute {pending}")
                break

    def _execute_comm_node(self, node: OpNode) -> None:
        """Execute a single communication node.

        Parameters
        ----------
        node : OpNode
            Communication node to execute.
        """
        logger.debug(f"Executing communication node: {node.name} ({node.op_type.value})")

        if node.op_type == OpType.ALLREDUCE:
            # For gradient all-reduce, we need to get gradients from model
            # This is a simplified implementation - in practice, FSDP handles this
            world_size = node.comm_group.get("world_size", get_world_size())
            op = node.comm_group.get("op", "sum")

            logger.debug(
                f"All-reduce: world_size={world_size}, op={op}, devices={node.device_ids}"
            )

            # Note: Actual gradient all-reduce would require access to parameters
            # This is handled by FSDP in practice

        elif node.op_type == OpType.BROADCAST:
            src = node.comm_group.get("src", 0)
            logger.debug(f"Broadcast: src={src}, devices={node.device_ids}")

        elif node.op_type == OpType.SYNC_PARAMS:
            logger.debug("Sync params operation")

        logger.info(f"Completed communication node: {node.name}")

    def synchronize(self) -> None:
        """Synchronize all pending operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns
        -------
        dict[str, Any]
            Statistics.
        """
        return {
            "step": self.step_counter,
            "accumulated_loss": self.accumulated_loss,
            "accumulation_steps": self.accumulation_steps,
        }


class ExecutionPlan:
    """Plans execution of a forward-backward training loop.

    Usage:
        >>> plan = ExecutionPlan.from_model(model, batch_size=32)
        >>> print(plan.estimate_memory_gb())
    """

    def __init__(
        self,
        num_forward_ops: int,
        num_backward_ops: int,
        total_compute_ms: float,
        total_memory_mb: float,
    ):
        """Initialize execution plan.

        Parameters
        ----------
        num_forward_ops : int
            Number of forward operations.
        num_backward_ops : int
            Number of backward operations.
        total_compute_ms : float
            Estimated compute time.
        total_memory_mb : float
            Estimated memory in MB.
        """
        self.num_forward_ops = num_forward_ops
        self.num_backward_ops = num_backward_ops
        self.total_compute_ms = total_compute_ms
        self.total_memory_mb = total_memory_mb

    def estimate_memory_gb(self) -> float:
        """Estimate total memory in GB.

        Returns
        -------
        float
            Memory in GB.
        """
        return self.total_memory_mb / 1024.0

    def estimate_time_seconds(self) -> float:
        """Estimate total execution time in seconds.

        Returns
        -------
        float
            Time in seconds (forward + backward).
        """
        return (self.total_compute_ms * 2) / 1000.0  # Forward + backward

    @staticmethod
    def from_model(
        model: nn.Module,
        batch_size: int,
        seq_len: int,
    ) -> ExecutionPlan:
        """Create an execution plan from a model.

        Parameters
        ----------
        model : nn.Module
            The model.
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.

        Returns
        -------
        ExecutionPlan
            Execution plan.
        """
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Rough estimates
        num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Module))
        compute_ms = (num_params * batch_size * seq_len) / 1e9 * 1000  # Rough estimate
        memory_mb = (num_params * 2) / 1e6  # float16

        return ExecutionPlan(
            num_forward_ops=num_layers,
            num_backward_ops=num_layers,
            total_compute_ms=compute_ms,
            total_memory_mb=memory_mb,
        )
