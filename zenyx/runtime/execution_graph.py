"""Forward and backward execution graph with dependencies.

This module defines the execution graph that represents:
- Forward operations
- Backward operations
- Dependencies between them
- Communication placeholders (all-reduce, parameter sync)

The graph is deterministic, traversable, and designed to be extended
with async execution and communication overlap in future phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OpType(Enum):
    """Operation types in the execution graph."""

    FORWARD = "forward"
    BACKWARD = "backward"
    ALLREDUCE = "allreduce"
    SYNC_PARAMS = "sync_params"
    ACTIVATE = "activate"  # Restore activations from checkpoint
    DEACTIVATE = "deactivate"  # Release activations


@dataclass
class OpNode:
    """Represents a single operation in the execution graph.

    Attributes:
        name: Unique operation identifier
        op_type: Type of operation (from OpType enum)
        inputs: List of input node names (dependencies)
        outputs: List of output tensor names
        module_name: Name of the module this operation belongs to
        compute_time_ms: Estimated compute time in milliseconds
        memory_bytes: Memory footprint of this operation
        metadata: Additional metadata (tensor shapes, etc.)
    """

    name: str
    op_type: OpType
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    module_name: str = ""
    compute_time_ms: float = 0.0
    memory_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OpNode):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        return f"OpNode({self.name}, {self.op_type.value})"


class ExecutionGraph:
    """Represents the complete forward-backward execution graph.

    The graph consists of:
    - Forward nodes: in order of execution
    - Backward nodes: in reverse topological order
    - Communication nodes: all-reduce, parameter sync
    - Memory nodes: activate/deactivate for checkpointing

    The graph is deterministic and fully traversable.

    Usage:
        >>> graph = ExecutionGraph()
        >>> graph.add_forward_node(node1)
        >>> graph.add_backward_node(node2)
        >>> forward_order = graph.get_forward_execution_order()
        >>> backward_order = graph.get_backward_execution_order()
    """

    def __init__(self):
        """Initialize empty execution graph."""
        self._forward_nodes: list[OpNode] = []
        self._backward_nodes: list[OpNode] = []
        self._all_nodes_by_name: dict[str, OpNode] = {}
        self._dependencies: dict[str, list[str]] = {}  # node_name -> [dep_node_names]
        self._node_counter = 0

    def add_forward_node(self, node: OpNode) -> None:
        """Add a forward operation node.

        Parameters
        ----------
        node : OpNode
            The forward operation node to add.
        """
        self._forward_nodes.append(node)
        self._all_nodes_by_name[node.name] = node
        self._dependencies[node.name] = node.inputs.copy()
        logger.debug(f"Added forward node: {node.name}")

    def add_backward_node(self, node: OpNode) -> None:
        """Add a backward operation node.

        Parameters
        ----------
        node : OpNode
            The backward operation node to add.
        """
        self._backward_nodes.append(node)
        self._all_nodes_by_name[node.name] = node
        self._dependencies[node.name] = node.inputs.copy()
        logger.debug(f"Added backward node: {node.name}")

    def add_dependency(self, node_name: str, depends_on: str) -> None:
        """Add a dependency between two nodes.

        Parameters
        ----------
        node_name : str
            The node that depends on another.
        depends_on : str
            The node it depends on.
        """
        if node_name not in self._dependencies:
            self._dependencies[node_name] = []

        if depends_on not in self._dependencies[node_name]:
            self._dependencies[node_name].append(depends_on)
            logger.debug(f"Added dependency: {node_name} <- {depends_on}")

    def get_forward_nodes(self) -> list[OpNode]:
        """Get all forward operation nodes.

        Returns
        -------
        list[OpNode]
            Forward nodes in execution order.
        """
        return self._forward_nodes.copy()

    def get_backward_nodes(self) -> list[OpNode]:
        """Get all backward operation nodes.

        Returns
        -------
        list[OpNode]
            Backward nodes in execution order (reverse topological).
        """
        return self._backward_nodes.copy()

    def get_forward_execution_order(self) -> list[OpNode]:
        """Get forward nodes in correct execution order.

        Returns
        -------
        list[OpNode]
            Forward nodes topologically sorted by dependencies.
        """
        return self._topological_sort(self._forward_nodes)

    def get_backward_execution_order(self) -> list[OpNode]:
        """Get backward nodes in correct execution order.

        Returns
        -------
        list[OpNode]
            Backward nodes topologically sorted by dependencies.
        """
        return self._topological_sort(self._backward_nodes)

    def _topological_sort(self, nodes: list[OpNode]) -> list[OpNode]:
        """Perform topological sort on nodes respecting dependencies.

        Parameters
        ----------
        nodes : list[OpNode]
            Nodes to sort.

        Returns
        -------
        list[OpNode]
            Topologically sorted nodes.
        """
        visited = set()
        sorted_nodes = []

        def visit(node: OpNode) -> None:
            if node.name in visited:
                return

            # Visit dependencies first
            for dep_name in self._dependencies.get(node.name, []):
                if dep_name in self._all_nodes_by_name:
                    visit(self._all_nodes_by_name[dep_name])

            visited.add(node.name)
            sorted_nodes.append(node)

        for node in nodes:
            visit(node)

        return sorted_nodes

    def get_node(self, name: str) -> Optional[OpNode]:
        """Get a node by name.

        Parameters
        ----------
        name : str
            Node name.

        Returns
        -------
        Optional[OpNode]
            The node, or None if not found.
        """
        return self._all_nodes_by_name.get(name)

    def get_node_dependencies(self, node_name: str) -> list[str]:
        """Get dependencies for a node.

        Parameters
        ----------
        node_name : str
            Node name.

        Returns
        -------
        list[str]
            Names of nodes this node depends on.
        """
        return self._dependencies.get(node_name, []).copy()

    def summarize(self) -> dict[str, Any]:
        """Summarize the execution graph.

        Returns
        -------
        dict[str, Any]
            Summary statistics.
        """
        all_nodes = self._forward_nodes + self._backward_nodes
        total_compute = sum(node.compute_time_ms for node in all_nodes)
        total_memory = sum(node.memory_bytes for node in all_nodes)

        return {
            "num_forward_ops": len(self._forward_nodes),
            "num_backward_ops": len(self._backward_nodes),
            "total_ops": len(all_nodes),
            "total_compute_ms": total_compute,
            "total_memory_mb": total_memory / (1024 * 1024),
            "num_dependencies": len(self._dependencies),
        }

    def validate(self) -> bool:
        """Validate graph structure for correctness.

        Returns
        -------
        bool
            True if graph is valid.
        """
        # Check all dependencies exist
        for node_name, deps in self._dependencies.items():
            for dep in deps:
                if dep not in self._all_nodes_by_name:
                    logger.warning(
                        f"Node {node_name} depends on non-existent node {dep}"
                    )
                    return False

        # Check no cycles (simplified check)
        visited = set()
        rec_stack = set()

        def has_cycle(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)

            for dep in self._dependencies.get(node_name, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node_name)
            return False

        for node_name in self._all_nodes_by_name:
            if node_name not in visited:
                if has_cycle(node_name):
                    logger.warning(f"Cycle detected in graph starting from {node_name}")
                    return False

        return True


class ExecutionGraphBuilder:
    """Builds execution graphs by tracing models.

    This builder inspects a model and creates a complete execution graph
    including forward nodes, backward nodes (implied by reversal), and
    communication placeholders for distributed training.

    Usage:
        >>> builder = ExecutionGraphBuilder()
        >>> graph = builder.build_from_model(model, sample_input, world_size=2)
        >>> forward_order = graph.get_forward_execution_order()
        >>> backward_order = graph.get_backward_execution_order()
    """

    def __init__(self):
        """Initialize the graph builder."""
        self._graph = ExecutionGraph()
        self._module_names: list[str] = []
        self._node_counter = 0

    def build_from_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        world_size: int = 1,
    ) -> ExecutionGraph:
        """Build execution graph by tracing model.

        Parameters
        ----------
        model : nn.Module
            The model to trace.
        sample_input : torch.Tensor
            Sample input for tracing.
        world_size : int
            Number of distributed training processes. Default: 1 (single GPU).

        Returns
        -------
        ExecutionGraph
            The built execution graph.
        """
        logger.info(
            f"Building execution graph from model (world_size={world_size})"
        )

        # Collect all leaf modules
        leaf_modules = []
        for name, module in model.named_modules():
            if not list(module.children()):
                leaf_modules.append((name, module))

        # Build forward nodes by hooking into execution
        forward_nodes_map: dict[str, OpNode] = {}
        backward_nodes_map: dict[str, OpNode] = {}
        hook_handles = []

        def make_forward_hook(module_name: str) -> Callable[..., None]:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                node_id = self._node_counter
                self._node_counter += 1

                # Calculate output size
                output_size = self._estimate_tensor_size(output)

                # Forward node
                forward_node = OpNode(
                    name=f"forward_{module_name}_{node_id}",
                    op_type=OpType.FORWARD,
                    module_name=module_name,
                    memory_bytes=int(output_size),
                    compute_time_ms=1.0,  # Placeholder
                    metadata={"module": module_name, "order": node_id},
                )

                # Corresponding backward node (implicit)
                backward_node = OpNode(
                    name=f"backward_{module_name}_{node_id}",
                    op_type=OpType.BACKWARD,
                    module_name=module_name,
                    memory_bytes=int(output_size),
                    compute_time_ms=2.0,  # Backward typically ~2x
                    metadata={"module": module_name, "order": node_id},
                )

                forward_nodes_map[forward_node.name] = forward_node
                backward_nodes_map[backward_node.name] = backward_node
                self._graph.add_forward_node(forward_node)
                self._graph.add_backward_node(backward_node)

            return hook

        # Install hooks
        for name, module in leaf_modules:
            hook = module.register_forward_hook(make_forward_hook(name))
            hook_handles.append(hook)

        # Trace forward pass
        try:
            with torch.no_grad():
                _ = model(sample_input)
        finally:
            for hook in hook_handles:
                hook.remove()

        # Add communication nodes for distributed training
        if world_size > 1:
            self._add_communication_nodes(world_size)

        logger.info(
            f"Built graph with {len(self._graph.get_forward_nodes())} "
            f"forward and {len(self._graph.get_backward_nodes())} backward nodes"
        )

        return self._graph

    def _estimate_tensor_size(self, tensor_or_tuple: Any) -> float:
        """Estimate memory size of tensor(s).

        Parameters
        ----------
        tensor_or_tuple : Any
            A tensor or tuple of tensors.

        Returns
        -------
        float
            Size in bytes.
        """
        total_size = 0.0

        if isinstance(tensor_or_tuple, torch.Tensor):
            total_size = float(tensor_or_tuple.element_size() * tensor_or_tuple.numel())
        elif isinstance(tensor_or_tuple, (tuple, list)):
            for item in tensor_or_tuple:
                total_size += self._estimate_tensor_size(item)

        return total_size

    def _add_communication_nodes(self, world_size: int) -> None:
        """Add communication placeholder nodes for distributed training.

        In Phase 2, these are placeholders. Phase 3 will implement actual
        all-reduce and parameter sync operations.

        Parameters
        ----------
        world_size : int
            Number of processes.
        """
        # Add all-reduce node after backward
        allreduce_node = OpNode(
            name="allreduce_gradients",
            op_type=OpType.ALLREDUCE,
            compute_time_ms=10.0,  # Placeholder
            metadata={"world_size": world_size},
        )
        self._graph.add_backward_node(allreduce_node)

        # Make all-reduce depend on all backward nodes
        for backward_node in self._graph.get_backward_nodes()[:-1]:  # Exclude allreduce
            self._graph.add_dependency("allreduce_gradients", backward_node.name)

        logger.debug(f"Added communication nodes for world_size={world_size}")


__all__ = ["ExecutionGraph", "ExecutionGraphBuilder", "OpNode", "OpType"]
