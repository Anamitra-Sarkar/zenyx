"""KV cache tiering with deterministic ring-access scheduling."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class KVBlockRef:
    block_id: int
    seq_start: int
    seq_end: int


class KVTierManager:
    """T0/T1/T2 KV cache hierarchy.

    - T0: active HBM blocks (tiny)
    - T1: host DRAM recent blocks
    - T2: cold storage (disk-backed emulation)
    """

    def __init__(self, t0_capacity: int = 2, t1_capacity: int = 32):
        self.t0: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.t1: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.t2: dict[int, torch.Tensor] = {}
        self.t0_capacity = t0_capacity
        self.t1_capacity = t1_capacity

    def put(self, block_id: int, tensor: torch.Tensor) -> None:
        self.t2[block_id] = tensor.detach().cpu()

    def prefetch_to_t0(self, block_id: int, device: torch.device) -> torch.Tensor:
        if block_id in self.t0:
            self.t0.move_to_end(block_id)
            return self.t0[block_id]

        if block_id in self.t1:
            cpu_tensor = self.t1.pop(block_id)
        else:
            cpu_tensor = self.t2[block_id]

        gpu_tensor = cpu_tensor.to(device, non_blocking=(device.type == "cuda"))
        self.t0[block_id] = gpu_tensor
        self._evict_t0_if_needed()
        return gpu_tensor

    def evict_from_t0(self, block_id: int) -> None:
        tensor = self.t0.pop(block_id, None)
        if tensor is not None:
            self.t1[block_id] = tensor.detach().cpu()
            self._evict_t1_if_needed()

    def _evict_t0_if_needed(self) -> None:
        while len(self.t0) > self.t0_capacity:
            old_block, old_tensor = self.t0.popitem(last=False)
            self.t1[old_block] = old_tensor.detach().cpu()
            self._evict_t1_if_needed()

    def _evict_t1_if_needed(self) -> None:
        while len(self.t1) > self.t1_capacity:
            old_block, old_tensor = self.t1.popitem(last=False)
            self.t2[old_block] = old_tensor


def build_ring_timeline(num_blocks: int) -> dict[str, list[int]]:
    """Deterministic forward/backward access schedule for ring attention."""
    forward = list(range(num_blocks))
    backward = list(reversed(forward))
    return {"forward": forward, "backward": backward}


__all__ = ["KVTierManager", "KVBlockRef", "build_ring_timeline"]
