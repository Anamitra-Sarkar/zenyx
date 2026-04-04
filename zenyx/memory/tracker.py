"""Lightweight memory tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int


class MemoryTracker:
    def __init__(self):
        self.snapshots: list[MemorySnapshot] = []

    def snapshot(self) -> MemorySnapshot:
        if torch.cuda.is_available():
            snap = MemorySnapshot(
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
            )
        else:
            snap = MemorySnapshot(allocated_bytes=0, reserved_bytes=0)
        self.snapshots.append(snap)
        return snap

    def possible_leak(self, threshold_bytes: int = 64 * 1024 * 1024) -> bool:
        if len(self.snapshots) < 2:
            return False
        return (self.snapshots[-1].allocated_bytes - self.snapshots[0].allocated_bytes) > threshold_bytes


__all__ = ["MemoryTracker", "MemorySnapshot"]
