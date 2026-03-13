"""Zenyx integration test suite.

Tests the full stack end-to-end with a tiny model and fake data.
No distributed setup required — runs single-process.

Run with: python -m zenyx.bench.integration_test
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_model(
    vocab_size: int = 1000, d_model: int = 64, n_layers: int = 4, n_heads: int = 4
) -> nn.Module:
    """A minimal transformer for testing."""

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        batch_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embedding(x)
            for layer in self.layers:
                h = layer(h)
            return self.head(h)

    return TinyTransformer()


def make_fake_dataloader(
    vocab_size: int = 1000, seq_len: int = 512, batch_size: int = 2, steps: int = 20
) -> list:
    """Yields (input_ids, labels) pairs."""
    data = []
    for _ in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((input_ids, labels))
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trainer_init() -> None:
    """Test Trainer initializes without error on CPU."""
    from zenyx.train.trainer import Trainer

    model = make_tiny_model()
    dataloader = make_fake_dataloader(steps=5)
    trainer = Trainer(
        model,
        dataloader,
        total_steps=5,
        warmup_steps=2,
        log_every=1,
        checkpoint_dir="/tmp/zenyx_test_ckpt",
        checkpoint_every=100,  # don't checkpoint during test
        dtype="float32",
        selective_activation_checkpoint=False,
    )
    state = trainer.get_state()
    assert state["step"] == 0
    assert state["topology"]["world_size"] >= 1


def test_single_train_step() -> None:
    """Test one training step completes without error."""
    from zenyx.train.trainer import Trainer

    model = make_tiny_model()
    dataloader = make_fake_dataloader(steps=3)
    trainer = Trainer(
        model,
        dataloader,
        total_steps=2,
        warmup_steps=1,
        log_every=1,
        checkpoint_dir="/tmp/zenyx_test_ckpt",
        checkpoint_every=100,
        dtype="float32",
        selective_activation_checkpoint=False,
    )
    trainer.train()
    state = trainer.get_state()
    assert state["step"] >= 1, f"Expected at least 1 step, got {state['step']}"


def test_memory_pool_no_oom() -> None:
    """Test that memory pool components initialize without error."""
    from zenyx.core.allocator.reuse_heap import ReuseHeap

    heap = ReuseHeap()
    assert heap is not None
    # Basic heap operations
    heap.update_access(block_id=1, current_op_idx=0)
    heap.update_access(block_id=2, current_op_idx=1)
    heap.remove_block(block_id=1)


def test_cosine_schedule() -> None:
    """Test LR schedule values at key checkpoints."""
    from zenyx.train.lr_schedule import CosineWithWarmup

    peak_lr = 1e-4
    warmup_steps = 100
    total_steps = 1000

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    scheduler = CosineWithWarmup(
        optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps, total_steps=total_steps
    )

    # At step 0: lr == 0
    assert scheduler.get_lr() == 0.0, f"Step 0: expected 0.0, got {scheduler.get_lr()}"

    # Advance to warmup_steps
    for _ in range(warmup_steps):
        scheduler.step()

    lr_at_warmup = scheduler.get_lr()
    assert abs(lr_at_warmup - peak_lr) < 1e-10, (
        f"At warmup_steps: expected {peak_lr}, got {lr_at_warmup}"
    )

    # Advance to total_steps
    for _ in range(total_steps - warmup_steps):
        scheduler.step()

    lr_at_end = scheduler.get_lr()
    min_lr = peak_lr * 0.1
    assert abs(lr_at_end - min_lr) < 1e-10, (
        f"At total_steps: expected ~{min_lr}, got {lr_at_end}"
    )


def test_ring_attention_single_device() -> None:
    """Test RingFlashAttentionCUDA on CPU with world_size=1."""
    from zenyx.ops.attention.ring_flash_cuda import RingFlashAttentionCUDA

    # The class should instantiate without errors
    attn = RingFlashAttentionCUDA(head_dim=64, num_heads=4, num_kv_heads=4)
    assert attn is not None


def test_vocab_parallel_single_device() -> None:
    """Test VocabParallelCrossEntropy with process_group=None."""
    from zenyx.ops.vocab.vocab_parallel import VocabParallelCrossEntropy

    batch_size = 2
    seq_len = 8
    vocab_size = 100

    logits = torch.randn(batch_size * seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))

    loss = VocabParallelCrossEntropy.apply(
        logits, targets, 0, vocab_size, None
    )
    assert loss is not None
    assert loss.dim() == 0 or loss.numel() == batch_size * seq_len


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_trainer_init,
        test_single_train_step,
        test_memory_pool_no_oom,
        test_cosine_schedule,
        test_ring_attention_single_device,
        test_vocab_parallel_single_device,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"\u2705 {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"\u274c {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        raise SystemExit(1)
