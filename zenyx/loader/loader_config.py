"""Configuration for Zenyx fast model loader.

Centralises all loader knobs into a single frozen dataclass so users
can construct, serialise, and pass around loader configuration without
touching individual constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LoaderConfig"]


@dataclass
class LoaderConfig:
    """Configuration knobs for :class:`~zenyx.loader.loader.ModelLoader`.

    Attributes
    ----------
    num_buffers : int
        Number of rotating buffers for triple-buffered I/O (default 3).
    prefetch_bytes : int
        Read-ahead window in bytes (default 512 MiB).
    use_gpu_direct : bool
        Attempt to use GPU Direct Storage (cuFile / kvikio) if available.
    dtype : str
        Target data type after loading (e.g. ``"bfloat16"``).
    max_load_time_seconds : float
        Advisory timeout; logs a warning if loading exceeds this.
    verify_integrity : bool
        Validate checkpoint keys and shapes before assigning weights.

    Time: O(1) construction.  Space: O(1).
    """

    num_buffers: int = 3
    prefetch_bytes: int = 512 * 1024 * 1024
    use_gpu_direct: bool = True
    dtype: str = "bfloat16"
    max_load_time_seconds: float = 30.0
    verify_integrity: bool = True

    def __repr__(self) -> str:
        return (
            f"LoaderConfig(buffers={self.num_buffers}, "
            f"prefetch={self.prefetch_bytes / (1024**2):.0f}MB, "
            f"gds={self.use_gpu_direct}, "
            f"dtype={self.dtype!r}, "
            f"timeout={self.max_load_time_seconds:.0f}s, "
            f"verify={self.verify_integrity})"
        )
