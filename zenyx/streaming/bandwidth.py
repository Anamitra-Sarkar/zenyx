"""Bandwidth-aware execution throttling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BandwidthSample:
    fetch_time_ms: float
    compute_time_ms: float


class BandwidthAwareScheduler:
    """Throttle when transfer dominates compute."""

    def __init__(self, max_fetch_to_compute_ratio: float = 1.0):
        self.max_ratio = max_fetch_to_compute_ratio

    def should_throttle(self, sample: BandwidthSample) -> bool:
        if sample.compute_time_ms <= 0:
            return True
        ratio = sample.fetch_time_ms / sample.compute_time_ms
        return ratio > self.max_ratio

    def throttle_ms(self, sample: BandwidthSample) -> float:
        if not self.should_throttle(sample):
            return 0.0
        return max(0.0, sample.fetch_time_ms - sample.compute_time_ms)


__all__ = ["BandwidthSample", "BandwidthAwareScheduler"]
