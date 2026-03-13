"""Zenyx compute operations — kernels, communication, and attention primitives.

Sub-packages
------------
vocab     : Vocabulary-parallel cross-entropy loss.
comm      : Hardware topology detection and ring communication.
attention : Ring and local attention implementations (CUDA/TPU/CPU).
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
]
