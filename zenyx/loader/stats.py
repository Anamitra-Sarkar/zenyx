"""Statistics for Zenyx fast model loader.

Captures throughput, timing, and configuration details for a single load
operation so that callers can log or monitor loader performance.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LoaderStats"]


@dataclass
class LoaderStats:
    """Post-load performance statistics.

    Attributes
    ----------
    bytes_loaded : int
        Total raw bytes transferred from storage to device.
    elapsed_seconds : float
        Wall-clock time for the load operation.
    throughput_mb_per_sec : float
        Effective throughput in MB/s.
    num_buffers_used : int
        Number of rotating buffers used during the load.
    gpu_direct_used : bool
        Whether GPU Direct Storage was used for the transfer.

    Time: O(1) construction.  Space: O(1).
    """

    bytes_loaded: int = 0
    elapsed_seconds: float = 0.0
    throughput_mb_per_sec: float = 0.0
    num_buffers_used: int = 0
    gpu_direct_used: bool = False

    def __repr__(self) -> str:
        return (
            f"LoaderStats(loaded={self.bytes_loaded / (1024**2):.1f}MB, "
            f"elapsed={self.elapsed_seconds:.2f}s, "
            f"throughput={self.throughput_mb_per_sec:.1f}MB/s, "
            f"buffers={self.num_buffers_used}, "
            f"gds={self.gpu_direct_used})"
        )
