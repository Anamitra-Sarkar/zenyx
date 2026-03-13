"""Vocabulary-parallel cross-entropy loss."""

from zenyx.ops.vocab.vocab_parallel import (
    VocabParallelCrossEntropy,
    vocab_parallel_cross_entropy,
)

__all__ = ["VocabParallelCrossEntropy", "vocab_parallel_cross_entropy"]
