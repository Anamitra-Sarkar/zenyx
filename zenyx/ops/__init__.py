"""Zenyx compute operations — kernels, communication, and attention primitives.

Sub-packages
------------
vocab     : Vocabulary-parallel cross-entropy loss.
comm      : Hardware topology detection and ring communication.
attention : Ring and local attention implementations (CUDA/TPU/CPU).
remat     : JAX Offloadable checkpoint policy for host-DRAM activation
            offloading. Fixes 55 GB XLA HBM OOM on TPU v5 lite.
"""

from zenyx.ops.vocab import VocabParallelCrossEntropy, vocab_parallel_cross_entropy
from zenyx.ops.comm import (
    DeviceInfo,
    DeviceType,
    Link,
    LinkType,
    RingCommunicator,
    Topology,
    TopologyDetector,
)
from zenyx.ops.attention import FlashAttentionCPU, RingFlashAttention
from zenyx.ops.remat import (
    offload_policy,
    make_offload_policy,
    make_offload_remat,
)

__all__ = [
    # vocab
    "VocabParallelCrossEntropy",
    "vocab_parallel_cross_entropy",
    # comm
    "DeviceInfo",
    "DeviceType",
    "Link",
    "LinkType",
    "RingCommunicator",
    "Topology",
    "TopologyDetector",
    # attention
    "FlashAttentionCPU",
    "RingFlashAttention",
    # remat
    "offload_policy",
    "make_offload_policy",
    "make_offload_remat",
]
