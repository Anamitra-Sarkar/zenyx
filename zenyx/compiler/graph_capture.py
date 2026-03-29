"""Graph capture for forward pass computation.

Captures the forward graph for analysis and future torch.compile integration.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GraphNode:
    """Represents a node in the forward computation graph.

    Attributes:
        name: Operation name
        op: Operation type (e.g., "linear", "attention", "activation")
        inputs: Input tensor shapes
        outputs: Output tensor shapes
        params: Number of parameters
        flops: Estimated FLOPs
    """

    def __init__(
        self,
        name: str,
        op: str,
        inputs: list[tuple[int, ...]],
        outputs: list[tuple[int, ...]],
        params: int = 0,
        flops: int = 0,
    ):
        self.name = name
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.flops = flops

    def __repr__(self) -> str:
        return (
            f"GraphNode(name={self.name!r}, op={self.op!r}, "
            f"params={self.params}, flops={self.flops})"
        )


class ExecutionGraph:
    """Captures the forward computation graph.

    Usage:
        >>> graph_capturer = ExecutionGraph()
        >>> graph = graph_capturer.capture(model, sample_input)
        >>> print(graph.nodes)
    """

    def __init__(self):
        """Initialize the graph capturer."""
        self.nodes: list[GraphNode] = []
        self._node_counter = 0

    def add_node(
        self,
        name: str,
        op: str,
        inputs: list[tuple[int, ...]],
        outputs: list[tuple[int, ...]],
        params: int = 0,
        flops: int = 0,
    ) -> GraphNode:
        """Add a node to the graph.

        Parameters
        ----------
        name : str
            Operation name
        op : str
            Operation type
        inputs : list[tuple[int, ...]]
            Input shapes
        outputs : list[tuple[int, ...]]
            Output shapes
        params : int
            Number of parameters
        flops : int
            Estimated FLOPs

        Returns
        -------
        GraphNode
            The added node.
        """
        node = GraphNode(
            name=name,
            op=op,
            inputs=inputs,
            outputs=outputs,
            params=params,
            flops=flops,
        )
        self.nodes.append(node)
        return node

    def capture(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> ExecutionGraph:
        """Capture the forward graph by tracing through the model.

        Parameters
        ----------
        model : nn.Module
            The model to trace.
        sample_input : torch.Tensor
            Sample input tensor.

        Returns
        -------
        ExecutionGraph
            Self (for chaining).
        """
        logger.info("Capturing execution graph for model")

        # Hook into modules to track forward calls
        hooks = []
        node_map = {}

        def make_forward_hook(
            module_name: str,
        ) -> Callable[[nn.Module, Any, Any], None]:
            def forward_hook(
                module: nn.Module,
                input: tuple[Any, ...],
                output: Any,
            ) -> None:
                # Extract shapes
                input_shapes = []
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        input_shapes.append(tuple(inp.shape))

                output_shapes = []
                if isinstance(output, torch.Tensor):
                    output_shapes.append(tuple(output.shape))
                elif isinstance(output, (tuple, list)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            output_shapes.append(tuple(out.shape))

                # Count parameters
                num_params = sum(p.numel() for p in module.parameters())

                # Infer operation type
                op_type = module.__class__.__name__.lower()

                node = self.add_node(
                    name=module_name,
                    op=op_type,
                    inputs=input_shapes,
                    outputs=output_shapes,
                    params=num_params,
                )
                node_map[module_name] = node

            return forward_hook

        # Register hooks
        for name, module in model.named_modules():
            if not list(module.children()):  # Leaf modules only
                hook = module.register_forward_hook(make_forward_hook(name))
                hooks.append(hook)

        # Trace forward
        try:
            with torch.no_grad():
                _ = model(sample_input)
        finally:
            for hook in hooks:
                hook.remove()

        logger.info(f"Captured {len(self.nodes)} nodes in execution graph")
        return self

    def summarize(self) -> dict[str, Any]:
        """Summarize graph statistics.

        Returns
        -------
        dict[str, Any]
            Statistics about the graph.
        """
        total_params = sum(node.params for node in self.nodes)
        total_flops = sum(node.flops for node in self.nodes)

        return {
            "num_nodes": len(self.nodes),
            "total_params": total_params,
            "total_flops": total_flops,
        }
