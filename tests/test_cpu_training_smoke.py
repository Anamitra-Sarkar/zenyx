"""CPU training smoke tests and re-audit checklist verification."""
import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_cpu_training_smoke():
    torch.manual_seed(42)
    vocab = 1000
    seq_len = 16
    batch = 4
    steps = 10

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, 64)
            self.fc = nn.Linear(64, vocab)

        def forward(self, x):
            return self.fc(self.emb(x))

    model = TinyLM()
    X = torch.randint(0, vocab, (batch * steps, seq_len))
    y = torch.randint(0, vocab, (batch * steps, seq_len))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch)

    from zenyx.train.trainer import Trainer

    trainer = Trainer(
        model,
        loader,
        total_steps=steps,
        dtype="float32",
        lr=1e-3,
        warmup_steps=2,
        checkpoint_every=1000,
        selective_activation_checkpoint=False,
    )
    trainer.train()
    state = trainer.get_state()
    assert state["step"] == steps, f"Expected {steps} steps, got {state['step']}"
    assert state["loss"] < 10.0, f"Loss too high: {state['loss']} — training may be broken"
    assert state["throughput_tokens_per_sec"] > 0
    print(f"CPU smoke test passed. Final loss={state['loss']:.4f}, steps={state['step']}")


def test_sparse_attn_warns():
    """sparse_attn=True should now emit a UserWarning about the stub."""
    model = nn.Linear(64, 64)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 16, 64), torch.randn(4, 16, 64)), batch_size=2
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            from zenyx.train.trainer import Trainer

            Trainer(
                model,
                loader,
                total_steps=1,
                dtype="float32",
                sparse_attn=True,
                selective_activation_checkpoint=False,
            )
        except Exception:
            pass
        sparse_warns = [x for x in w if "sparse" in str(x.message).lower()]
        assert len(sparse_warns) > 0, "Expected UserWarning for sparse_attn stub"


def test_curriculum_divisibility_check():
    """Bad curriculum schedule should raise ValueError at init, not mid-training."""
    from zenyx.train.ring_curriculum import RingCurriculumManager

    with pytest.raises(ValueError, match="divisible"):
        RingCurriculumManager(curriculum_schedule=[(100, 3)])  # 100 not divisible by 3


def test_gqa_formula_consistency():
    """n_kv_heads must be <= n_heads after fix."""
    from zenyx.core.agent.planner import ParallelismPlanner
    from zenyx.core.hal.detector import HardwareInfo
    from zenyx.ops.comm.topology import Topology

    hw = HardwareInfo(
        backend="cpu",
        device_count=8,
        per_device_memory_bytes=16 * (1024 ** 3),
        interconnect="none",
        bandwidth_t0_t1=50 * (1024 ** 3),
        bandwidth_t1_t2=7 * (1024 ** 3),
        compute_tflops=10.0,
        device_name="CPU-test",
    )
    planner = ParallelismPlanner(hw, Topology())
    # Should not raise and KV estimate should be positive but not astronomically large
    kv_gb = planner._estimate_kv_cache_gb(7e9, 4096, 32000)
    assert 0 < kv_gb < 100, f"KV cache estimate suspiciously large: {kv_gb:.2f} GB"


def test_convergence_check_rejects_oscillation():
    """Oscillating loss should NOT trigger should_advance()."""
    from zenyx.train.ring_curriculum import RingCurriculumManager

    mgr = RingCurriculumManager(convergence_window=10, convergence_threshold=0.01)
    # Oscillating loss: starts and ends at same value but oscillates
    history = [2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 2.0]  # range=0.5
    assert not mgr.should_advance(history), "Oscillating loss should not advance stage"
    # Truly converged loss
    history2 = [2.001, 2.000, 2.001, 2.000, 2.001, 2.000, 2.001, 2.000, 2.001, 2.000]
    assert mgr.should_advance(history2), "Converged loss should advance stage"
