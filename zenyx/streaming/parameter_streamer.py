"""Layer-wise weight streaming with prefetch + double buffering."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LayerBlock:
    layer_id: str
    tensor: torch.Tensor


class ParameterStore:
    """Host-tier storage for layer weights (T1/T2 emulation)."""

    def __init__(self):
        self.blocks: dict[str, torch.Tensor] = {}

    def register(self, layer_id: str, tensor: torch.Tensor) -> None:
        self.blocks[layer_id] = tensor.detach().cpu().contiguous()

    def load(self, layer_id: str) -> torch.Tensor:
        return self.blocks[layer_id]


class LayerWeightStreamer:
    """Stream one layer at a time into device memory and evict aggressively."""

    def __init__(
        self,
        store: ParameterStore,
        device: torch.device,
        min_block_bytes: int = 5 * 1024 * 1024,
        max_resident_layers: int = 2,
    ):
        self.store = store
        self.device = device
        self.min_block_bytes = min_block_bytes
        self.max_resident_layers = max_resident_layers
        self._resident: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._prefetch: dict[str, Future[torch.Tensor]] = {}

    def prefetch(self, layer_id: str) -> None:
        if layer_id in self._resident or layer_id in self._prefetch:
            return
        self._prefetch[layer_id] = self._executor.submit(self._load_to_device, layer_id)

    def get(self, layer_id: str) -> torch.Tensor:
        if layer_id in self._resident:
            self._resident.move_to_end(layer_id)
            return self._resident[layer_id]

        if layer_id in self._prefetch:
            tensor = self._prefetch.pop(layer_id).result()
        else:
            tensor = self._load_to_device(layer_id)

        self._resident[layer_id] = tensor
        self._evict_if_needed()
        return tensor

    def evict(self, layer_id: str) -> None:
        tensor = self._resident.pop(layer_id, None)
        if tensor is not None and tensor.device.type == "cuda":
            del tensor

    def _evict_if_needed(self) -> None:
        while len(self._resident) > self.max_resident_layers:
            old_layer, tensor = self._resident.popitem(last=False)
            if tensor.device.type == "cuda":
                del tensor

    def _load_to_device(self, layer_id: str) -> torch.Tensor:
        cpu_tensor = self.store.load(layer_id)
        if cpu_tensor.numel() * cpu_tensor.element_size() < self.min_block_bytes:
            cpu_tensor = cpu_tensor.contiguous()
        if self.device.type == "cuda":
            pinned = cpu_tensor.pin_memory()
            return pinned.to(self.device, non_blocking=True)
        return cpu_tensor.to(self.device)


__all__ = ["ParameterStore", "LayerWeightStreamer", "LayerBlock"]
