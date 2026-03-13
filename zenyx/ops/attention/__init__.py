"""Attention implementations — ring (CUDA/TPU) and CPU fallback.

The TPU module (``RingFlashAttentionTPU``) is not imported by default to
avoid requiring JAX on non-TPU systems.  Import it explicitly::

    from zenyx.ops.attention.ring_pallas_tpu import RingFlashAttentionTPU
"""

from zenyx.ops.attention.flash_cpu import FlashAttentionCPU
from zenyx.ops.attention.ring_flash_cuda import RingFlashAttention

__all__ = [
    "FlashAttentionCPU",
    "RingFlashAttention",
]

# Lazy import for TPU module — only expose when JAX is available
def __getattr__(name: str) -> object:
    if name == "RingFlashAttentionTPU":
        from zenyx.ops.attention.ring_pallas_tpu import RingFlashAttentionTPU
        return RingFlashAttentionTPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
