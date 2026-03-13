"""Zenyx bench — Benchmarking utilities.

Exports
-------
- :func:`memory_budget` — compute a detailed memory budget.
- :class:`BudgetReport` — formatted budget report with box-drawing table.
- :class:`HardwarePreset` — hardware specification presets.
- :data:`HARDWARE_PRESETS` — built-in presets (H100, A100, H200, etc.).
- :func:`benchmark_vs_deepspeed` — head-to-head benchmark against DeepSpeed.
- :class:`ComparisonReport` — benchmark comparison result.
"""

from zenyx.bench.memory_budget import (
    HARDWARE_PRESETS,
    BudgetReport,
    HardwarePreset,
    memory_budget,
)
from zenyx.bench.vs_deepspeed import ComparisonReport, benchmark_vs_deepspeed

__all__ = [
    "HARDWARE_PRESETS",
    "BudgetReport",
    "ComparisonReport",
    "HardwarePreset",
    "benchmark_vs_deepspeed",
    "memory_budget",
]
