"""Streaming execution engine: prefetch -> compute -> evict (forward+backward)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from zenyx.streaming.bandwidth import BandwidthAwareScheduler, BandwidthSample
from zenyx.streaming.belady import BeladyApproxCache
from zenyx.streaming.kv_tiering import KVTierManager, build_ring_timeline
from zenyx.streaming.parameter_streamer import LayerWeightStreamer, ParameterStore


@dataclass
class StreamingConfig:
    max_hbm_layers: int = 2
    kv_t0_blocks: int = 2
    kv_t1_blocks: int = 64


class StreamingExecutionEngine:
    def __init__(self, model: nn.Module, device: torch.device, config: StreamingConfig | None = None):
        self.model = model
        self.device = device
        self.config = config or StreamingConfig()

        self.store = ParameterStore()
        self.layer_names: list[str] = []
        self.layers: list[nn.Module] = []
        self._extract_layers()

        self.weight_streamer = LayerWeightStreamer(
            self.store,
            device=device,
            max_resident_layers=self.config.max_hbm_layers,
        )
        self.kv_manager = KVTierManager(
            t0_capacity=self.config.kv_t0_blocks,
            t1_capacity=self.config.kv_t1_blocks,
        )
        self.cache = BeladyApproxCache(capacity=self.config.max_hbm_layers)
        self.bandwidth = BandwidthAwareScheduler(max_fetch_to_compute_ratio=1.0)

    def _extract_layers(self) -> None:
        for name, module in self.model.named_children():
            self.layer_names.append(name)
            self.layers.append(module)
            state = module.state_dict()
            if state:
                flat = torch.cat([t.detach().reshape(-1).cpu() for t in state.values()])
            else:
                flat = torch.zeros(1)
            self.store.register(name, flat)

    def run_forward_backward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        x = x.to(self.device)
        target = target.to(self.device)

        activations: list[torch.Tensor] = [x]

        # forward: stream each layer in, compute, evict immediately if not needed
        for i, (name, layer) in enumerate(zip(self.layer_names, self.layers)):
            if i + 1 < len(self.layer_names):
                self.weight_streamer.prefetch(self.layer_names[i + 1])

            fetch_t0 = time.perf_counter()
            _weights = self.weight_streamer.get(name)
            fetch_ms = (time.perf_counter() - fetch_t0) * 1000.0

            compute_t0 = time.perf_counter()
            out = layer(activations[-1])
            compute_ms = (time.perf_counter() - compute_t0) * 1000.0

            delay_ms = self.bandwidth.throttle_ms(BandwidthSample(fetch_ms, compute_ms))
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            activations.append(out)
            self.cache.update_future(name, i + 1)
            evicted = self.cache.touch(name)
            for old in evicted:
                self.weight_streamer.evict(old)

        loss = loss_fn(activations[-1], target)
        loss.backward()

        # backward timeline for KV prefetch/evict planning (deterministic ring).
        timeline = build_ring_timeline(num_blocks=max(1, len(self.layer_names)))
        for block_id in timeline["backward"]:
            if block_id in self.kv_manager.t2:
                _ = self.kv_manager.prefetch_to_t0(block_id, self.device)
                self.kv_manager.evict_from_t0(block_id)

        return float(loss.item())


__all__ = ["StreamingExecutionEngine", "StreamingConfig"]
