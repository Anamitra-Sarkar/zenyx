"""Zenyx memory allocator subsystem.

Exports:
    MemoryPool        — Pre-pinned three-tier memory pool (never-OOM).
    check_feasibility — Formal OOM-free bandwidth feasibility check.
    FeasibilityResult — Result of feasibility check.
    estimate_memory_budget — Transformer memory budget estimator.
    MemoryBudget      — Memory budget breakdown dataclass.
"""
from zenyx.core.allocator.feasibility import (
    FeasibilityResult,
    MemoryBudget,
    check_feasibility,
    estimate_memory_budget,
)
from zenyx.core.allocator.mem_pool import MemoryPool

__all__ = [
    "MemoryPool",
    "check_feasibility",
    "FeasibilityResult",
    "estimate_memory_budget",
    "MemoryBudget",
]
