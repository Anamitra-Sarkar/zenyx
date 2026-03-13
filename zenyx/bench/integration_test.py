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
# Phase 5: Agent Integration Tests
# ---------------------------------------------------------------------------


def test_async_profiler_records() -> None:
    """Test that AsyncProfiler accepts step records without crashing."""
    from zenyx.core.agent.profiler import AsyncProfiler

    profiler = AsyncProfiler(enabled=True)
    # Record a few operations
    for i in range(5):
        handle = profiler.start_op(f"op_{i}")
        profiler.end_op(handle)
    # Allow background thread to flush
    import time
    time.sleep(0.6)
    timings = profiler.get_timings()
    # All ops should have been recorded
    assert len(timings) == 5, f"Expected 5 ops, got {len(timings)}"
    for name, timing in timings.items():
        assert timing.count >= 1, f"Op {name} has count={timing.count}"
    profiler.shutdown()


def test_parallelism_planner_cpu() -> None:
    """Test that ParallelismPlanner returns a valid plan for CPU hardware."""
    from zenyx.core.hal.detector import detect_hardware
    from zenyx.core.agent.planner import ParallelismPlanner
    from zenyx.ops.comm.topology import Topology

    hw = detect_hardware()
    topo = Topology()
    planner = ParallelismPlanner(hw, topo)
    plan = planner.plan(
        model_params=1e6,
        vocab_size=32000,
        context_len=512,
        batch_size=2,
    )
    assert plan.tp_degree >= 1
    assert plan.pp_degree >= 1
    assert plan.dp_degree >= 1
    assert plan.ring_degree >= 1
    assert plan.schedule_type in ("1f1b", "braided", "gpipe")


def test_training_controller_step() -> None:
    """Test that TrainingController.step() runs without crashing."""
    from zenyx.core.hal.detector import detect_hardware
    from zenyx.core.agent.profiler import AsyncProfiler
    from zenyx.core.agent.planner import ParallelismPlanner
    from zenyx.core.agent.controller import TrainingController
    from zenyx.ops.comm.topology import Topology

    hw = detect_hardware()
    topo = Topology()
    profiler = AsyncProfiler(enabled=True)
    planner = ParallelismPlanner(hw, topo)

    controller = TrainingController(
        planner=planner,
        profiler=profiler,
        replan_interval=5,
        model_params=1e6,
        vocab_size=32000,
        batch_size=2,
    )

    # Run a few steps
    for i in range(10):
        result = controller.step(step_num=i, loss=1.0 - i * 0.05, context_len=512)

    stats = controller.get_training_stats()
    assert stats.steps_completed == 9
    profiler.shutdown()


def test_trainer_agent_state() -> None:
    """Test that Trainer.get_state() includes agent data."""
    from zenyx.train.trainer import Trainer

    model = make_tiny_model()
    dataloader = make_fake_dataloader(steps=3)
    trainer = Trainer(
        model,
        dataloader,
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        checkpoint_dir="/tmp/zenyx_test_ckpt",
        checkpoint_every=100,
        dtype="float32",
        selective_activation_checkpoint=False,
    )
    state = trainer.get_state()
    assert "parallelism_plan" in state
    assert "profiler_stats" in state
    # Plan should have valid structure
    plan = state["parallelism_plan"]
    assert plan is not None
    assert "tp_degree" in plan
    assert "dp_degree" in plan


# ---------------------------------------------------------------------------
# Phase 6: Fast Model Loader Tests
# ---------------------------------------------------------------------------


def test_loader_config_defaults() -> None:
    """Test LoaderConfig construction with defaults."""
    from zenyx.loader.loader_config import LoaderConfig

    config = LoaderConfig()
    assert config.num_buffers == 3
    assert config.prefetch_bytes == 512 * 1024 * 1024
    assert config.use_gpu_direct is True
    assert config.dtype == "bfloat16"
    assert config.max_load_time_seconds == 30.0
    assert config.verify_integrity is True


def test_model_loader_construction() -> None:
    """Test ModelLoader construction on CPU hardware."""
    from zenyx.core.hal.detector import detect_hardware
    from zenyx.loader.loader import ModelLoader

    hw = detect_hardware()
    loader = ModelLoader(hal=None, hw_info=hw)
    assert loader is not None
    stats = loader.get_stats()
    assert stats.bytes_loaded == 0  # No load yet


def test_load_model_roundtrip() -> None:
    """Test load_model on a tiny 2-layer model saved with torch.save to a temp file."""
    import tempfile
    import os

    from zenyx.loader.loader import load_model

    # Create a tiny model and save it
    model = make_tiny_model(vocab_size=100, d_model=32, n_layers=2, n_heads=2)
    state = {"model_state_dict": model.state_dict()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(state, f.name)
        tmp_path = f.name

    try:
        # Create a fresh model with same architecture and load
        fresh_model = make_tiny_model(vocab_size=100, d_model=32, n_layers=2, n_heads=2)

        # Verify weights differ before load
        old_param = next(iter(fresh_model.parameters())).clone()
        loaded_model = load_model(tmp_path, fresh_model, dtype="float32")
        assert loaded_model is fresh_model  # Same instance

        # Verify weights were actually loaded
        for name, param in model.named_parameters():
            loaded_param = dict(loaded_model.named_parameters())[name]
            assert torch.allclose(param.float(), loaded_param.float(), atol=1e-4), (
                f"Parameter {name} mismatch after load"
            )
    finally:
        os.unlink(tmp_path)


def test_loader_stats_populated() -> None:
    """Test that LoaderStats is populated correctly after a load."""
    import tempfile
    import os

    from zenyx.core.hal.detector import detect_hardware
    from zenyx.loader.loader import ModelLoader

    model = make_tiny_model(vocab_size=100, d_model=32, n_layers=2, n_heads=2)
    state = {"model_state_dict": model.state_dict()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(state, f.name)
        tmp_path = f.name

    try:
        hw = detect_hardware()
        loader = ModelLoader(hal=None, hw_info=hw, dtype="float32")
        fresh_model = make_tiny_model(vocab_size=100, d_model=32, n_layers=2, n_heads=2)
        loader.load(tmp_path, fresh_model)

        stats = loader.get_stats()
        assert stats.bytes_loaded > 0
        assert stats.elapsed_seconds > 0
        assert stats.throughput_mb_per_sec > 0
        assert stats.num_buffers_used == 3
    finally:
        os.unlink(tmp_path)


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
        # Phase 5: Agent integration
        test_async_profiler_records,
        test_parallelism_planner_cpu,
        test_training_controller_step,
        test_trainer_agent_state,
        # Phase 6: Fast model loader
        test_loader_config_defaults,
        test_model_loader_construction,
        test_load_model_roundtrip,
        test_loader_stats_populated,
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
