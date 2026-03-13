"""Attention implementations — ring (CUDA/TPU) and CPU fallback.

The TPU module (``RingFlashAttentionTPU``) is not imported by default to
avoid requiring JAX on non-TPU systems.  Import it explicitly::

    from zenyx.ops.attention.ring_pallas_tpu import RingFlashAttentionTPU
"""

from zenyx.ops.attention.flash_cpu import FlashAttentionCPU, flash_attention_cpu
from zenyx.ops.attention.ring_flash_cuda import RingFlashAttention, RingFlashAttentionCUDA

__all__ = [
    "FlashAttentionCPU",
    "flash_attention_cpu",
    "RingFlashAttention",
    "RingFlashAttentionCUDA",
]

# Lazy import for TPU module — only expose when JAX is available
def __getattr__(name: str) -> object:
    if name in ("RingFlashAttentionTPU", "ring_attention_tpu"):
        from zenyx.ops.attention import ring_pallas_tpu
        return getattr(ring_pallas_tpu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
