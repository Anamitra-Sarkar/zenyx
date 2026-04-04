"""Execution graph validator with simulation-based checks."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from zenyx.runtime.execution_graph import ExecutionGraph, OpType

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    severity: str
    category: str
    message: str
    node_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, category: str, message: str, node_name: str | None = None, details: dict[str, Any] | None = None) -> None:
        self.errors.append(ValidationError("error", category, message, node_name, details or {}))
        self.is_valid = False

    def add_warning(self, category: str, message: str, node_name: str | None = None, details: dict[str, Any] | None = None) -> None:
        self.warnings.append(ValidationError("warning", category, message, node_name, details or {}))


class GraphValidator:
    def validate(self, graph: ExecutionGraph) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        self._validate_dependencies_exist(graph, result)
        self._simulate_execution(graph, result)
        self._validate_communication_nodes(graph, result)
        return result

    def _validate_dependencies_exist(self, graph: ExecutionGraph, result: ValidationResult) -> None:
        for node in graph.get_all_nodes():
            for dep in graph.get_node_dependencies(node.name):
                if graph.get_node(dep) is None:
                    result.add_error("dependency", f"Missing dependency '{dep}'", node.name)

    def _simulate_execution(self, graph: ExecutionGraph, result: ValidationResult) -> None:
        nodes = graph.get_all_nodes()
        indegree = {n.name: 0 for n in nodes}
        for n in nodes:
            for _dep in graph.get_node_dependencies(n.name):
                indegree[n.name] += 1

        ready = deque(sorted([name for name, deg in indegree.items() if deg == 0]))
        executed: list[str] = []

        while ready:
            current = ready.popleft()
            executed.append(current)
            for dependent in graph.get_dependents(current):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if len(executed) != len(nodes):
            blocked = [name for name, deg in indegree.items() if deg > 0]
            result.add_error(
                "deadlock",
                "Simulation found blocked nodes (cycle or unsatisfiable dependencies)",
                details={"blocked_nodes": blocked},
            )

        execution_pos = {name: idx for idx, name in enumerate(executed)}
        for node in nodes:
            for dep in graph.get_node_dependencies(node.name):
                if dep in execution_pos and node.name in execution_pos and execution_pos[dep] > execution_pos[node.name]:
                    result.add_error(
                        "dependency",
                        f"Node '{node.name}' executes before dependency '{dep}'",
                        node.name,
                    )

    def _validate_communication_nodes(self, graph: ExecutionGraph, result: ValidationResult) -> None:
        for node in graph.get_all_nodes():
            if not node.is_comm_op:
                continue
            if "tensor" not in node.metadata:
                result.add_error(
                    "communication",
                    "Communication node missing metadata['tensor'] for executable collective",
                    node.name,
                )
            if node.op_type == OpType.ALLREDUCE:
                op = node.comm_group.get("op", "sum")
                if op not in {"sum", "prod", "min", "max", "avg"}:
                    result.add_error("communication", f"Invalid all-reduce op '{op}'", node.name)
            if node.op_type == OpType.BROADCAST and "src" not in node.comm_group:
                result.add_warning("communication", "Broadcast node missing explicit source rank", node.name)
            if not node.device_ids:
                result.add_warning("communication", "Communication node missing device_ids", node.name)


def validate_graph(graph: ExecutionGraph) -> ValidationResult:
    return GraphValidator().validate(graph)


__all__ = ["GraphValidator", "validate_graph", "ValidationError", "ValidationResult"]
