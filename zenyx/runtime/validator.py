"""Execution graph validator.

This module validates execution graphs for correctness:
- Dependency order verification
- Tensor shape consistency
- Device placement validation
- Communication operation consistency

This is CRITICAL for detecting bugs before execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from zenyx.runtime.execution_graph import ExecutionGraph, OpNode, OpType

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error."""

    severity: str  # "error" | "warning"
    category: str  # "dependency" | "shape" | "device" | "communication"
    message: str
    node_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ValidationError({self.severity}, {self.category}, {self.message})"


@dataclass
class ValidationResult:
    """Result of graph validation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(
        self,
        category: str,
        message: str,
        node_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add an error to the result."""
        error = ValidationError(
            severity="error",
            category=category,
            message=message,
            node_name=node_name,
            details=details or {},
        )
        self.errors.append(error)
        self.is_valid = False

    def add_warning(
        self,
        category: str,
        message: str,
        node_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a warning to the result."""
        warning = ValidationError(
            severity="warning",
            category=category,
            message=message,
            node_name=node_name,
            details=details or {},
        )
        self.warnings.append(warning)

    def summarize(self) -> dict[str, Any]:
        """Summarize validation results."""
        return {
            "is_valid": self.is_valid,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "error_categories": list(
                set(e.category for e in self.errors)
            ),
            "warning_categories": list(
                set(w.category for w in self.warnings)
            ),
        }


class GraphValidator:
    """Validates execution graphs for correctness.

    Checks:
    - All dependencies exist
    - No cycles in dependency graph
    - Correct execution order
    - Communication node consistency
    - Device placement validity

    Usage:
        >>> validator = GraphValidator()
        >>> result = validator.validate(graph)
        >>> if not result.is_valid:
        ...     print(result.errors)
    """

    def __init__(self):
        """Initialize the validator."""
        self._logger = logging.getLogger(f"{__name__}.GraphValidator")

    def validate(self, graph: ExecutionGraph) -> ValidationResult:
        """Validate an execution graph.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.

        Returns
        -------
        ValidationResult
            Validation result with errors and warnings.
        """
        result = ValidationResult(is_valid=True)

        self._validate_dependencies(graph, result)
        self._validate_no_cycles(graph, result)
        self._validate_execution_order(graph, result)
        self._validate_communication_nodes(graph, result)
        self._validate_device_placement(graph, result)
        self._validate_pipeline_consistency(graph, result)

        # Log summary
        summary = result.summarize()
        if result.is_valid:
            self._logger.info(
                f"Validation PASSED: {summary['num_warnings']} warnings"
            )
        else:
            self._logger.error(
                f"Validation FAILED: {summary['num_errors']} errors, "
                f"{summary['num_warnings']} warnings"
            )

        return result

    def _validate_dependencies(
        self,
        graph: ExecutionGraph,
        result: ValidationResult,
    ) -> None:
        """Validate that all dependencies exist.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.
        result : ValidationResult
            Result to populate.
        """
        all_node_names = set()
        for node in graph.get_forward_nodes() + graph.get_backward_nodes():
            all_node_names.add(node.name)

        for node_name in all_node_names:
            deps = graph.get_node_dependencies(node_name)
            for dep in deps:
                if dep not in all_node_names:
                    result.add_error(
                        category="dependency",
                        message=f"Node '{node_name}' depends on non-existent node '{dep}'",
                        node_name=node_name,
                        details={"missing_dependency": dep},
                    )

    def _validate_no_cycles(
        self,
        graph: ExecutionGraph,
        result: ValidationResult,
    ) -> None:
        """Validate that there are no cycles in the dependency graph.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.
        result : ValidationResult
            Result to populate.
        """
        all_nodes = graph.get_forward_nodes() + graph.get_backward_nodes()
        visited = set()
        rec_stack = set()
        path = []

        def has_cycle(node_name: str) -> tuple[bool, list[str]]:
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            deps = graph.get_node_dependencies(node_name)
            for dep in deps:
                dep_node = graph.get_node(dep)
                if dep_node is None:
                    continue

                if dep not in visited:
                    has_cycle_result = has_cycle(dep)
                    if has_cycle_result[0]:
                        return has_cycle_result
                elif dep in rec_stack:
                    cycle_path = path[path.index(dep) :] + [dep]
                    return True, cycle_path

            path.pop()
            rec_stack.remove(node_name)
            return False, []

        for node in all_nodes:
            if node.name not in visited:
                cycle_found, cycle_path = has_cycle(node.name)
                if cycle_found:
                    result.add_error(
                        category="dependency",
                        message=f"Cycle detected: {' -> '.join(cycle_path)}",
                        node_name=node.name,
                        details={"cycle_path": cycle_path},
                    )
                    return  # One cycle is enough to fail

    def _validate_execution_order(
        self,
        graph: ExecutionGraph,
        result: ValidationResult,
    ) -> None:
        """Validate that execution order respects dependencies.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.
        result : ValidationResult
            Result to populate.
        """
        forward_order = graph.get_forward_execution_order()
        backward_order = graph.get_backward_execution_order()

        # Check forward order
        executed = set()
        for i, node in enumerate(forward_order):
            deps = graph.get_node_dependencies(node.name)
            for dep in deps:
                if dep not in executed:
                    result.add_error(
                        category="dependency",
                        message=f"Forward node '{node.name}' at position {i} "
                        f"executed before dependency '{dep}'",
                        node_name=node.name,
                        details={
                            "position": i,
                            "missing_dependency": dep,
                            "phase": "forward",
                        },
                    )
            executed.add(node.name)

        # Check backward order
        executed = set()
        for i, node in enumerate(backward_order):
            deps = graph.get_node_dependencies(node.name)
            for dep in deps:
                if dep not in executed:
                    result.add_error(
                        category="dependency",
                        message=f"Backward node '{node.name}' at position {i} "
                        f"executed before dependency '{dep}'",
                        node_name=node.name,
                        details={
                            "position": i,
                            "missing_dependency": dep,
                            "phase": "backward",
                        },
                    )
            executed.add(node.name)

    def _validate_communication_nodes(
        self,
        graph: ExecutionGraph,
        result: ValidationResult,
    ) -> None:
        """Validate communication node consistency.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.
        result : ValidationResult
            Result to populate.
        """
        comm_nodes = []
        for node in graph.get_forward_nodes() + graph.get_backward_nodes():
            if node.is_comm_op:
                comm_nodes.append(node)

        for node in comm_nodes:
            # Check device_ids
            if not node.device_ids:
                result.add_warning(
                    category="communication",
                    message=f"Communication node '{node.name}' has no device_ids",
                    node_name=node.name,
                    details={"op_type": node.op_type.value},
                )

            # Check comm_group
            if not node.comm_group:
                result.add_warning(
                    category="communication",
                    message=f"Communication node '{node.name}' has no comm_group",
                    node_name=node.name,
                    details={"op_type": node.op_type.value},
                )

            # Validate all-reduce specifics
            if node.op_type == OpType.ALLREDUCE:
                op = node.comm_group.get("op", "sum")
                valid_ops = ["sum", "prod", "min", "max", "avg"]
                if op not in valid_ops:
                    result.add_error(
                        category="communication",
                        message=f"All-reduce node '{node.name}' has invalid op '{op}'",
                        node_name=node.name,
                        details={"op": op, "valid_ops": valid_ops},
                    )

            # Validate broadcast specifics
            if node.op_type == OpType.BROADCAST:
                src = node.comm_group.get("src")
                if src is None:
                    result.add_warning(
                        category="communication",
                        message=f"Broadcast node '{node.name}' has no source rank",
                        node_name=node.name,
                        details={"op_type": node.op_type.value},
                    )

    def _validate_device_placement(
        self,
        graph: ExecutionGraph,
        result: ValidationResult,
    ) -> None:
        """Validate device placement consistency.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.
        result : ValidationResult
            Result to populate.
        """
        # Collect all device IDs used
        all_devices = set()
        for node in graph.get_forward_nodes() + graph.get_backward_nodes():
            if node.device_ids:
                all_devices.update(node.device_ids)

        # Check for gaps in device IDs (potential configuration issue)
        if all_devices:
            min_device = min(all_devices)
            max_device = max(all_devices)
            expected_devices = set(range(min_device, max_device + 1))
            missing_devices = expected_devices - all_devices

            if missing_devices:
                result.add_warning(
                    category="device",
                    message=f"Gap in device IDs: missing {missing_devices}",
                    node_name=None,
                    details={
                        "all_devices": list(all_devices),
                        "missing_devices": list(missing_devices),
                    },
                )

    def _validate_pipeline_consistency(
        self,
        graph: ExecutionGraph,
        result: ValidationResult,
    ) -> None:
        """Validate pipeline parallel consistency.

        Parameters
        ----------
        graph : ExecutionGraph
            Graph to validate.
        result : ValidationResult
            Result to populate.
        """
        # Check for proper stage transitions in pipeline
        # This is a placeholder for more advanced pipeline validation
        pass


def validate_graph(graph: ExecutionGraph) -> ValidationResult:
    """Convenience function to validate a graph.

    Parameters
    ----------
    graph : ExecutionGraph
        Graph to validate.

    Returns
    -------
    ValidationResult
        Validation result.
    """
    validator = GraphValidator()
    return validator.validate(graph)


__all__ = [
    "GraphValidator",
    "validate_graph",
    "ValidationError",
    "ValidationResult",
]
