"""Dependency-driven scheduler with compute/communication overlap primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from zenyx.distributed import AsyncCollectiveHandle, all_reduce, broadcast, get_world_size
from zenyx.runtime.execution_graph import ExecutionGraph, OpNode, OpType

logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    intra_node_ranks: set[int]
    inter_node_ranks: set[int]


class GradientBucketReducer:
    """Bucket gradients to reduce collective launch overhead."""

    def __init__(self, bucket_size_mb: float = 25.0, async_safe: bool = True):
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.async_safe = async_safe

    def reduce(self, model: nn.Module) -> list[AsyncCollectiveHandle]:
        if get_world_size() <= 1:
            return []

        handles: list[AsyncCollectiveHandle] = []
        current: list[torch.Tensor] = []
        current_bytes = 0

        def flush() -> None:
            nonlocal current, current_bytes
            if not current:
                return

            views = [t.view(-1) for t in current]
            flat = torch.cat(views)
            handle_or_tensor = all_reduce(flat, op="sum", async_op=self.async_safe)
            if isinstance(handle_or_tensor, AsyncCollectiveHandle):
                handles.append(
                    _BucketHandle(
                        handle=handle_or_tensor,
                        grads=current.copy(),
                        world_size=get_world_size(),
                    )
                )
            else:
                _scatter_bucket(flat, current, get_world_size())
            current = []
            current_bytes = 0

        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not g.is_contiguous():
                g = g.contiguous()
            g_bytes = g.numel() * g.element_size()
            if g_bytes > self.bucket_size_bytes:
                flush()
                h_or_t = all_reduce(g.view(-1), op="sum", async_op=self.async_safe)
                if isinstance(h_or_t, AsyncCollectiveHandle):
                    handles.append(_BucketHandle(handle=h_or_t, grads=[g], world_size=get_world_size()))
                else:
                    _scatter_bucket(h_or_t.view(-1), [g], get_world_size())
                continue
            if current_bytes + g_bytes > self.bucket_size_bytes:
                flush()
            current.append(g)
            current_bytes += g_bytes

        flush()
        return handles


def _scatter_bucket(flat: torch.Tensor, grads: list[torch.Tensor], world_size: int) -> None:
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset : offset + count].view_as(grad) / world_size)
        offset += count


class _BucketHandle(AsyncCollectiveHandle):
    """Handle that writes reduced bucket values back into parameter grads on wait."""

    def __init__(self, handle: AsyncCollectiveHandle, grads: list[torch.Tensor], world_size: int):
        self.work = handle.work
        self.tensor = handle.tensor
        self.name = handle.name
        self._done = False
        self._grads = grads
        self._world_size = world_size

    def wait(self) -> torch.Tensor:
        reduced = super().wait()
        _scatter_bucket(reduced.view(-1), self._grads, self._world_size)
        return reduced


class Scheduler:
    def __init__(
        self,
        accumulation_steps: int = 1,
        enable_overlap: bool = True,
        bucket_size_mb: float = 25.0,
        topology: Optional[TopologyConfig] = None,
    ):
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.accumulated_loss = 0.0
        self.enable_overlap = enable_overlap and torch.cuda.is_available()
        self.topology = topology
        self.bucket_reducer = GradientBucketReducer(bucket_size_mb=bucket_size_mb, async_safe=self.enable_overlap)

        if self.enable_overlap:
            self.compute_stream = torch.cuda.default_stream()
            self.comm_stream = torch.cuda.Stream()
        else:
            self.compute_stream = None
            self.comm_stream = None

    def forward(self, model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return model(batch)

    def backward(
        self,
        loss: torch.Tensor,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        execution_graph: Optional[ExecutionGraph] = None,
    ) -> None:
        loss.backward()

        pending_handles: list[AsyncCollectiveHandle] = []

        if execution_graph is not None:
            pending_handles.extend(self._execute_graph_nodes(execution_graph))

        # Bucketed gradient sync is mandatory when model is provided.
        if model is not None:
            pending_handles.extend(self.bucket_reducer.reduce(model))

        for handle in pending_handles:
            handle.wait()

        self.step_counter += 1
        if self.step_counter % self.accumulation_steps == 0:
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self.accumulated_loss = 0.0
        else:
            self.accumulated_loss += float(loss.detach().item())

    def _execute_graph_nodes(self, graph: ExecutionGraph) -> list[AsyncCollectiveHandle]:
        if not graph.validate():
            raise ValueError("Refusing to execute invalid graph")

        nodes = graph.get_all_nodes()
        indegree = {n.name: 0 for n in nodes}
        by_name = {n.name: n for n in nodes}
        for node in nodes:
            for dep in graph.get_node_dependencies(node.name):
                indegree[node.name] += 1

        ready: set[str] = {name for name, d in indegree.items() if d == 0}
        done: set[str] = set()
        pending_handles: list[AsyncCollectiveHandle] = []

        while ready:
            name = self._select_next_ready_node(ready, by_name)
            ready.remove(name)
            node = by_name[name]
            handle = self._execute_node(node)
            if handle is not None:
                pending_handles.append(handle)
            done.add(name)

            for dependent in graph.get_dependents(name):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.add(dependent)

        if len(done) != len(nodes):
            raise RuntimeError("Deadlock/cycle detected during graph execution")

        return pending_handles


    def _select_next_ready_node(self, ready: set[str], by_name: dict[str, OpNode]) -> str:
        if self.topology is None:
            return sorted(ready)[0]

        def score(node_name: str) -> tuple[int, str]:
            node = by_name[node_name]
            if not node.is_comm_op:
                return (0, node_name)
            ranks = set(node.device_ids)
            inter = len(ranks & self.topology.inter_node_ranks) > 0
            return (2 if inter else 1, node_name)

        return sorted(ready, key=score)[0]

    def _execute_node(self, node: OpNode) -> Optional[AsyncCollectiveHandle]:
        if node.is_compute_op:
            return None  # Compute nodes are executed by autograd/forward call sites.

        if node.op_type == OpType.ALLREDUCE:
            tensor = node.metadata.get("tensor")
            if tensor is None:
                raise ValueError(f"ALLREDUCE node '{node.name}' missing metadata['tensor']")
            return self._launch_allreduce(tensor, op=node.comm_group.get("op", "sum"))

        if node.op_type == OpType.BROADCAST:
            tensor = node.metadata.get("tensor")
            if tensor is None:
                raise ValueError(f"BROADCAST node '{node.name}' missing metadata['tensor']")
            src = int(node.comm_group.get("src", 0))
            out = broadcast(tensor, src=src, async_op=self.enable_overlap)
            if isinstance(out, AsyncCollectiveHandle):
                return out
            return None

        return None

    def _launch_allreduce(self, tensor: torch.Tensor, op: str = "sum") -> Optional[AsyncCollectiveHandle]:
        if get_world_size() <= 1:
            return None

        if self.enable_overlap and self.comm_stream is not None:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(event)
                handle_or_tensor = all_reduce(tensor, op=op, async_op=True)
            if isinstance(handle_or_tensor, AsyncCollectiveHandle):
                return handle_or_tensor
            return None

        handle_or_tensor = all_reduce(tensor, op=op, async_op=False)
        if isinstance(handle_or_tensor, AsyncCollectiveHandle):
            return handle_or_tensor
        return None

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_stats(self) -> dict[str, Any]:
        return {
            "step": self.step_counter,
            "accumulated_loss": self.accumulated_loss,
            "accumulation_steps": self.accumulation_steps,
            "overlap_enabled": self.enable_overlap,
            "bucket_size_mb": self.bucket_reducer.bucket_size_bytes / (1024 * 1024),
            "topology": None
            if self.topology is None
            else {
                "intra_node_ranks": sorted(self.topology.intra_node_ranks),
                "inter_node_ranks": sorted(self.topology.inter_node_ranks),
            },
        }


class ExecutionPlan:
    def __init__(self, num_forward_ops: int, num_backward_ops: int, total_compute_ms: float, total_memory_mb: float):
        self.num_forward_ops = num_forward_ops
        self.num_backward_ops = num_backward_ops
        self.total_compute_ms = total_compute_ms
        self.total_memory_mb = total_memory_mb

    def estimate_memory_gb(self) -> float:
        return self.total_memory_mb / 1024.0

    def estimate_time_seconds(self) -> float:
        return (self.total_compute_ms * 2) / 1000.0

    @staticmethod
    def from_model(model: nn.Module, batch_size: int, seq_len: int) -> "ExecutionPlan":
        num_params = sum(p.numel() for p in model.parameters())
        num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Module))
        compute_ms = (num_params * batch_size * seq_len) / 1e9 * 1000
        memory_mb = (num_params * 2) / 1e6
        return ExecutionPlan(num_layers, num_layers, compute_ms, memory_mb)
