"""Approximate Belady scheduler using next-use distance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccessEvent:
    tensor_id: str
    step: int


class BeladyApproxCache:
    """Evict tensors with farthest known next-use (Belady approximation)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._resident: set[str] = set()
        self._next_use: dict[str, int] = {}

    def update_future(self, tensor_id: str, next_use_step: int) -> None:
        self._next_use[tensor_id] = next_use_step

    def touch(self, tensor_id: str) -> list[str]:
        evicted: list[str] = []
        if tensor_id in self._resident:
            return evicted

        if len(self._resident) >= self.capacity:
            victim = self._select_victim()
            if victim is not None:
                self._resident.remove(victim)
                evicted.append(victim)

        self._resident.add(tensor_id)
        return evicted

    def _select_victim(self) -> str | None:
        if not self._resident:
            return None

        # farthest next use wins eviction; missing next-use treated as infinity
        victim = None
        farthest = -1
        for tid in self._resident:
            nxt = self._next_use.get(tid, 10**18)
            if nxt > farthest:
                farthest = nxt
                victim = tid
        return victim

    @property
    def resident(self) -> set[str]:
        return set(self._resident)


__all__ = ["BeladyApproxCache", "AccessEvent"]
