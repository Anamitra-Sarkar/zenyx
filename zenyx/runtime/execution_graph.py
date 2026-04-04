"""Forward/backward/communication execution graph primitives.

The graph is a *strict DAG* and is used by the scheduler to drive execution
based on dependency readiness (not list order).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OpType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    ALLREDUCE = "allreduce"
    BROADCAST = "broadcast"
    SYNC_PARAMS = "sync_params"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"


@dataclass
class OpNode:
    name: str
    op_type: OpType
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    module_name: str = ""
    compute_time_ms: float = 0.0
    memory_bytes: int = 0
    device_ids: list[int] = field(default_factory=list)
    comm_group: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_compute_op(self) -> bool:
        return self.op_type in (OpType.FORWARD, OpType.BACKWARD)

    @property
    def is_comm_op(self) -> bool:
        return self.op_type in (OpType.ALLREDUCE, OpType.BROADCAST, OpType.SYNC_PARAMS)

    @property
    def is_memory_op(self) -> bool:
        return self.op_type in (OpType.ACTIVATE, OpType.DEACTIVATE)


class ExecutionGraph:
    """Strict DAG for executable operations."""

    def __init__(self):
        self._forward_nodes: list[OpNode] = []
        self._backward_nodes: list[OpNode] = []
        self._all_nodes_by_name: dict[str, OpNode] = {}
        self._dependencies: dict[str, list[str]] = {}
        self._reverse_deps: dict[str, list[str]] = {}

    def _register_node(self, node: OpNode) -> None:
        if node.name in self._all_nodes_by_name:
            raise ValueError(f"Duplicate node name: {node.name}")

        self._all_nodes_by_name[node.name] = node
        self._dependencies[node.name] = []
        self._reverse_deps.setdefault(node.name, [])

        for dep in node.inputs:
            self.add_dependency(node.name, dep)

    def add_forward_node(self, node: OpNode) -> None:
        self._register_node(node)
        self._forward_nodes.append(node)

    def add_backward_node(self, node: OpNode) -> None:
        self._register_node(node)
        self._backward_nodes.append(node)

    def add_dependency(self, node_name: str, depends_on: str) -> None:
        if node_name not in self._all_nodes_by_name:
            raise KeyError(f"Unknown node: {node_name}")
        if depends_on not in self._all_nodes_by_name:
            raise KeyError(f"Dependency node does not exist: {depends_on}")

        deps = self._dependencies.setdefault(node_name, [])
        if depends_on in deps:
            return

        deps.append(depends_on)
        self._reverse_deps.setdefault(depends_on, []).append(node_name)

        if self._has_cycle():
            deps.remove(depends_on)
            self._reverse_deps[depends_on].remove(node_name)
            raise ValueError(f"Adding dependency {node_name} <- {depends_on} creates a cycle")

    def _topological_sort(self, nodes: list[OpNode]) -> list[OpNode]:
        node_set = {n.name for n in nodes}
        indegree = {name: 0 for name in node_set}

        for name in node_set:
            for dep in self._dependencies.get(name, []):
                if dep in node_set:
                    indegree[name] += 1

        ready = deque(sorted([n for n, d in indegree.items() if d == 0]))
        ordered_names: list[str] = []

        while ready:
            current = ready.popleft()
            ordered_names.append(current)
            for nxt in self._reverse_deps.get(current, []):
                if nxt not in indegree:
                    continue
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(ordered_names) != len(node_set):
            raise ValueError("Cycle detected while topologically sorting graph")

        by_name = {n.name: n for n in nodes}
        return [by_name[name] for name in ordered_names]

    def _has_cycle(self) -> bool:
        all_nodes = list(self._all_nodes_by_name.values())
        if not all_nodes:
            return False
        try:
            self._topological_sort(all_nodes)
            return False
        except ValueError:
            return True

    def get_node(self, name: str) -> Optional[OpNode]:
        return self._all_nodes_by_name.get(name)

    def get_node_dependencies(self, node_name: str) -> list[str]:
        return self._dependencies.get(node_name, []).copy()

    def get_dependents(self, node_name: str) -> list[str]:
        return self._reverse_deps.get(node_name, []).copy()

    def get_forward_nodes(self) -> list[OpNode]:
        return self._forward_nodes.copy()

    def get_backward_nodes(self) -> list[OpNode]:
        return self._backward_nodes.copy()

    def get_all_nodes(self) -> list[OpNode]:
        return list(self._all_nodes_by_name.values())

    def get_forward_execution_order(self) -> list[OpNode]:
        return self._topological_sort(self._forward_nodes)

    def get_backward_execution_order(self) -> list[OpNode]:
        return self._topological_sort(self._backward_nodes)

    def validate(self) -> bool:
        try:
            self._topological_sort(self.get_all_nodes())
        except ValueError:
            return False

        for node_name, deps in self._dependencies.items():
            if node_name not in self._all_nodes_by_name:
                return False
            for dep in deps:
                if dep not in self._all_nodes_by_name:
                    return False
        return True

    def summarize(self) -> dict[str, Any]:
        all_nodes = self.get_all_nodes()
        return {
            "num_forward_ops": len(self._forward_nodes),
            "num_backward_ops": len(self._backward_nodes),
            "total_ops": len(all_nodes),
            "total_compute_ms": sum(n.compute_time_ms for n in all_nodes),
            "total_memory_mb": sum(n.memory_bytes for n in all_nodes) / (1024 * 1024),
            "num_dependencies": sum(len(v) for v in self._dependencies.values()),
            "is_dag": self.validate(),
        }


class ExecutionGraphBuilder:
    """Build a conservative execution graph from a model trace."""

    def __init__(self):
        self._node_counter = 0

    def build_from_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        world_size: int = 1,
    ) -> ExecutionGraph:
        graph = ExecutionGraph()

        leaf_modules: list[tuple[str, nn.Module]] = [
            (name, module)
            for name, module in model.named_modules()
            if not list(module.children())
        ]

        execution_order: list[tuple[str, int, float]] = []
        hook_handles = []

        def make_forward_hook(module_name: str) -> Callable[..., None]:
            def hook(module: nn.Module, _input: Any, output: Any) -> None:
                node_id = self._node_counter
                self._node_counter += 1
                execution_order.append((module_name, node_id, self._estimate_tensor_size(output)))

            return hook

        for name, module in leaf_modules:
            hook_handles.append(module.register_forward_hook(make_forward_hook(name)))

        try:
            with torch.no_grad():
                _ = model(sample_input)
        finally:
            for hook in hook_handles:
                hook.remove()

        prev_fwd: Optional[str] = None
        backward_names: list[str] = []
        for module_name, node_id, out_size in execution_order:
            fwd_name = f"forward_{module_name}_{node_id}"
            bwd_name = f"backward_{module_name}_{node_id}"

            fwd = OpNode(
                name=fwd_name,
                op_type=OpType.FORWARD,
                module_name=module_name,
                compute_time_ms=1.0,
                memory_bytes=int(out_size),
            )
            graph.add_forward_node(fwd)
            if prev_fwd is not None:
                graph.add_dependency(fwd_name, prev_fwd)
            prev_fwd = fwd_name

            bwd = OpNode(
                name=bwd_name,
                op_type=OpType.BACKWARD,
                module_name=module_name,
                compute_time_ms=2.0,
                memory_bytes=int(out_size),
            )
            graph.add_backward_node(bwd)
            backward_names.append(bwd_name)

        # Backward dependencies: reverse chain + each backward depends on its forward.
        for idx in range(len(backward_names) - 1, -1, -1):
            current = backward_names[idx]
            forward_ref = current.replace("backward_", "forward_", 1)
            if graph.get_node(forward_ref) is not None:
                graph.add_dependency(current, forward_ref)
            if idx < len(backward_names) - 1:
                graph.add_dependency(current, backward_names[idx + 1])

        if world_size > 1 and backward_names:
            allreduce = OpNode(
                name="allreduce_gradients",
                op_type=OpType.ALLREDUCE,
                compute_time_ms=10.0,
                device_ids=list(range(world_size)),
                comm_group={"world_size": world_size, "op": "sum", "bucketed": True},
            )
            graph.add_backward_node(allreduce)
            for bwd_name in backward_names:
                graph.add_dependency(allreduce.name, bwd_name)

        if not graph.validate():
            raise ValueError("Generated execution graph is invalid")
        return graph

    def _estimate_tensor_size(self, tensor_or_tuple: Any) -> float:
        if isinstance(tensor_or_tuple, torch.Tensor):
            return float(tensor_or_tuple.element_size() * tensor_or_tuple.numel())
        if isinstance(tensor_or_tuple, (tuple, list)):
            return float(sum(self._estimate_tensor_size(item) for item in tensor_or_tuple))
        return 0.0


__all__ = ["ExecutionGraph", "ExecutionGraphBuilder", "OpNode", "OpType"]
