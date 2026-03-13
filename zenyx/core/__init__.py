"""Zenyx core — HAL and allocator subsystems.

Re-exports key types from :mod:`zenyx.core.hal` and
:mod:`zenyx.core.allocator` for convenience.
"""
from zenyx.core.allocator import (
    FeasibilityResult,
    MemoryBudget,
    MemoryPool,
    check_feasibility,
    estimate_memory_budget,
)
from zenyx.core.hal import (
    CudaHAL,
    HALBase,
    HardwareInfo,
    MemBlock,
    MemTier,
    ReduceOp,
    detect_hardware,
)

__all__ = [
    # HAL
    "HALBase",
    "MemTier",
    "MemBlock",
    "ReduceOp",
    "CudaHAL",
    "detect_hardware",
    "HardwareInfo",
    # Allocator
    "MemoryPool",
    "check_feasibility",
    "FeasibilityResult",
    "estimate_memory_budget",
    "MemoryBudget",
]
