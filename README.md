# Zenyx

**Hardware-agnostic, self-managing distributed LLM training runtime.**

Never OOM. No config files. No manual tuning.  
1T parameters · 1M+ context · 500K+ vocabulary · 120B model loading in under 20 seconds.

---

## Quickstart

### Install

```bash
# From source (recommended for development)
git clone https://github.com/Anamitra-Sarkar/zenyx
cd zenyx
pip install -e .

# Or from PyPI (when released)
pip install zenyx
```

### Train a minimal toy model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenyx.train.trainer import Trainer

# 1. Define a minimal model
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))

# 2. Create a simple DataLoader
data = torch.randn(256, 32)
loader = DataLoader(TensorDataset(data), batch_size=16)

# 3. Train with the modern API
trainer = Trainer(model, loader, lr=1e-3, max_steps=50)
trainer.train()
print(trainer.get_state())
```

That's it. Zenyx handles hardware detection, memory management, and
parallelism automatically.

### Hardware Compatibility

| Hardware             | Attention Kernel            | Max Context (tested) | Notes            |
| -------------------- | --------------------------- | -------------------- | ---------------- |
| H100 NVLink 4.0 (8+) | TokenRing FA3 (Triton)      | 1M tokens            | S_min=1,099      |
| TPU v5e-8            | Ring Pallas + Shardy        | 1M tokens            | S_min=493        |
| TPU v5p              | Ring Pallas + Shardy        | 1M tokens            | S_min=383        |
| Any GPU (PCIe)       | Ring FA3 (PyTorch fallback) | 128K tokens          | S_min=15,454     |
| CPU / Apple M3       | Chunked attention           | 32K tokens           | Development only |

---

## Hardware Guide

### Single GPU (CUDA)

Zenyx auto-detects your GPU and uses it without any extra configuration.

```python
# Works out of the box — Zenyx picks up CUDA:0 automatically
from zenyx.train.trainer import Trainer
trainer = Trainer(model, loader, lr=1e-4, dtype="bfloat16")
trainer.train()
```

- HBM is used as T0; DRAM as T1; NVMe as T2 for offloading.
- Bélády-optimal eviction ensures optimal use of available VRAM.
- FP8 E4M3 activation checkpointing is used by default for large models.

### Multi-GPU (CUDA)

Launch with `torchrun` or set `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE`
environment variables (SLURM, MPI):

```bash
torchrun --nproc_per_node=8 train.py
```

```python
# In train.py — Trainer auto-initialises distributed if env vars are set
from zenyx.train.trainer import Trainer
trainer = Trainer(model, loader, lr=1e-4, context_len=131072)
trainer.train()
```

Zenyx auto-selects TP/PP/DP degrees based on model size and GPU count.

### CPU-only

```python
# Works on any machine without a GPU
from zenyx.train.trainer import Trainer
trainer = Trainer(model, loader, lr=1e-4)
trainer.train()
```

- Falls back to chunked attention (no FlashAttention kernel).
- All three-tier memory management is active but T0 is system RAM.
- Training is slower — suitable for development, debugging, and CI.
- No OOM crashes; memory is managed transparently.

### TPU / XLA (experimental)

TPU support via `torch_xla` is experimental. Install the matching
`torch_xla` package for your TPU generation, then:

```python
import torch_xla.core.xla_model as xm
# Zenyx auto-detects XLA devices
from zenyx.train.trainer import Trainer
trainer = Trainer(model, loader, lr=1e-4)
trainer.train()
```

Ring Pallas attention (Phase 9/10) is available on TPU v5e and v5p.
See `zenyx/ops/attention/ring_pallas_tpu.py` for details.

---

## Never-OOM Guarantee

Zenyx uses a **three-tier memory hierarchy** to guarantee your training
never crashes with OOM:

| Tier | Storage | Speed | Purpose |
|------|---------|-------|---------|
| T0 | HBM/VRAM | Fastest | Active tensors |
| T1 | CPU DRAM | Fast | Evicted tensors staged here |
| T2 | NVMe SSD | Slower | Cold data offloaded here |

### How it works

1. **Bélády-optimal eviction**: the allocator looks ahead in the compute
   graph and evicts the tensor whose next use is farthest in the future.
   This is provably optimal — no other eviction policy does better.

2. **Async prefetch**: while the GPU computes on the current batch, the
   allocator pre-stages the next batch's tensors from T2→T1→T0.

3. **FP8 activation storage**: activations are stored in FP8 E4M3 format
   (~2× memory saving) using per-layer quantisation with straight-through
   gradient estimation.

### Feasible vs throttle mode

**Feasible** (OOM-free guarantee active):  
`F_compute ≤ _PIPELINE_DEPTH × max(B₀₁, B₁₂)`

where `B₀₁` is T0↔T1 bandwidth, `B₁₂` is T1↔T2 bandwidth, `F_compute`
is the effective memory demand of compute (TFLOPS / arithmetic intensity),
and `_PIPELINE_DEPTH` (~100) is the prefetch lookahead depth.

Modern GPUs (H100, A100) with NVMe storage are typically in **feasible**
mode.

**Throttle mode** (condition not met):  
The runtime slows compute to match available bandwidth.  Training
continues — it never crashes.  If you see throttle mode, you can:

- Reduce batch size or sequence length.
- Add faster NVMe storage.
- Free GPU memory by stopping other processes.

---

## Running Tests and CI

### Local

```bash
pytest tests/ -v
```

All test files run on CPU-only machines — no GPU required.

### GitHub Actions CI

The workflow in `.github/workflows/ci.yml` runs on every push and pull
request to `main`:

- Triggered on: `push` and `pull_request` to `main`
- Runner: `ubuntu-latest` with Python 3.11
- Install: `pip install -e ".[dev]"` (fallback: `pip install torch pytest`)
- Command: `pytest tests/ -v --timeout=60`
- Pip cache: enabled via `actions/cache`

---

## What is Zenyx?

Zenyx is a Python library that sits below the model definition and above the hardware. You write a model and call `zenyx.train(model, dataloader)`. Zenyx handles everything else — memory layout, kernel dispatch, parallelism, checkpointing, communication.

## Key Features

### Never-OOM Guarantee

Three-tier memory system manages data across three storage levels:

- **T0 (HBM/VRAM):** Fastest GPU memory — active tensors live here
- **T1 (CPU DRAM):** Pinned host memory — evicted tensors staged here  
- **T2 (NVMe SSD):** Persistent storage — cold data offloaded here

The Bélády-optimal eviction policy uses reuse distances computed from the static compute graph to evict the tensor whose next use is farthest in the future — provably optimal.

**Formal feasibility condition:** `F_compute ≤ _PIPELINE_DEPTH × max(B₀₁, B₁₂)`

Where `B₀₁` is T0↔T1 bandwidth, `B₁₂` is T1↔T2 bandwidth, `F_compute`
is the compute demand in bytes/sec, and `_PIPELINE_DEPTH` (~100) is the
prefetch lookahead depth.  If this condition holds, Zenyx guarantees zero OOM.
If it doesn't hold, Zenyx throttles compute to match memory bandwidth — never crashes.

### Hardware Agnostic
- **NVIDIA CUDA** — cuBLAS + FlashAttention-3 + NCCL + cuFile GDS
- **AMD ROCm** — hipBLASLt + MIOpen + RCCL
- **Google TPU** — Pallas + Shardy + JAX custom_partitioning + ICI
- **CPU** — OpenBLAS + AVX-512
- **Apple Metal/MLX** — Inference and fine-tuning

Five HAL primitives: `alloc`, `free`, `copy`, `matmul`, `reduce` — everything in Zenyx is composed from these.

### Architecture Compatibility

Zenyx is architecture-agnostic and supports the following without modification:

- **MLA** (Multi-Latent Attention)
- **MoE** (Mixture of Experts) with dynamic routing
- **MTP** (Multi-Token Prediction)
- **GQA** (Grouped Query Attention)
- **ConvSwiGLU** feed-forward networks
- **RoPE** (Rotary Position Embeddings)
- Any `torch.nn.Module`-based architecture

### Intelligent Memory Management
- **Bélády-optimal eviction** via min-heap of reuse distances computed from the static compute graph
- **Block-level granularity** (2–20 MB) aligned to NVMe page sizes
- **Async prefetching** — looks ahead in the compute graph and pre-stages blocks
- **FP8 E4M3 activation storage** — 2× memory savings (INT8 is unsafe per [QuEST, arXiv 2502.05003](https://arxiv.org/abs/2502.05003))

### Extreme Scale
- **Ring Attention** with TokenRing pattern for H100 NVLink and Pallas for TPU
- **Braided TP+PP scheduling** — near-zero pipeline bubble fraction ([arXiv 2510.27257](https://arxiv.org/abs/2510.27257))
- **Megatron-style distributed cross-entropy** — handles 500K+ vocab without materializing the full logit tensor
- **GPU Direct Storage** — triple-buffered model loading bypassing CPU

### Autonomous Optimization
- Lightweight async CUDA event profiler (< 1% overhead)
- Parallelism planner that auto-determines TP/PP/DP/Ring degrees
- Training controller that replans at curriculum shifts or every 1,000 steps
- DynaPipe-safe pipeline stage reassignment ([NeurIPS 2025](https://arxiv.org/abs/2311.10418))

## Architecture

```
zenyx/
├── core/
│   ├── hal/           # Hardware Abstraction Layer (CUDA, ROCm, XLA, CPU)
│   ├── allocator/     # Three-tier memory pool + Bélády eviction
│   └── agent/         # Profiler, parallelism planner, training controller
├── ops/
│   ├── attention/     # Ring FlashAttention (CUDA, TPU, CPU) + Sparse Ring (Phase 10)
│   ├── comm/          # Topology detection, ring comm, all-reduce
│   └── vocab/         # Distributed cross-entropy (500K+ vocab safe)
├── train/
│   ├── trainer.py     # Trainer class + zenyx.train() entrypoint  ← PRIMARY API
│   ├── loop.py        # Legacy train loop (Phase 2, deprecated — use Trainer)
│   ├── kv_cache_tier.py  # Phase 7: Bélády-optimal KV cache tiering
│   ├── fp8_kv.py      # Phase 8: FP8 KV quantization + STE + SwiGLU
│   ├── ring_curriculum.py # Phase 9: Dynamic ring degree curriculum
│   ├── lr_schedule.py # Cosine LR with linear warmup
│   ├── grad_scaler.py # Mixed precision gradient scaler
│   ├── distributed_setup.py  # Auto distributed init (torchrun/SLURM/TPU)
│   ├── activation_checkpoint.py  # Selective activation checkpointing
│   ├── pipeline.py    # Braided TP+PP schedule
│   ├── mixed_prec.py  # FP8 E4M3 activation storage
│   └── checkpoint.py  # Async distributed checkpointing
├── loader/
│   ├── loader.py       # Hardware-aware triple-buffered checkpoint loader
│   ├── loader_config.py # Loader configuration dataclass
│   ├── stats.py        # Post-load performance statistics
│   ├── gds_loader.py   # GPU Direct Storage triple-buffering
│   └── tpu_loader.py   # TPU multi-threaded pread + DMA
└── bench/
    ├── integration_test.py  # End-to-end integration tests
    ├── memory_budget.py     # Memory budget calculator
    └── vs_deepspeed.py      # Benchmark vs DeepSpeed ZeRO-3
```

### Documentation

```
zenyx/docs/
├── performance_ceiling.md  # Throughput analysis for 1T model on 8-chip TPU v5e
└── dispute_resolutions.md  # Research dispute empirical outcomes
```

## Trainer API

The primary training entrypoint is `zenyx.train.trainer.Trainer`:

```python
from zenyx.train.trainer import Trainer

trainer = Trainer(
    model,
    dataloader,
    lr=1e-4,
    weight_decay=0.1,
    warmup_steps=2000,
    total_steps=100_000,
    dtype="bfloat16",
    context_len=131072,
    gradient_accumulation_steps=4,
    checkpoint_dir="./checkpoints",
    checkpoint_every=1000,
)
trainer.train()
```

> **Deprecated:** `zenyx.train.loop` is deprecated as of v1.0 and will be removed in a future
> release. Importing it emits a `DeprecationWarning`. Use `zenyx.train.trainer.Trainer` instead.

## Memory Budget Calculator

```python
from zenyx.bench.memory_budget import memory_budget

report = memory_budget(
    params=7e9,
    vocab_size=152_000,
    context_len=8192,
    hardware="H100",
)
print(report)
```

## Phase 5 — Agent Integration (Autonomous Replanning)

The agent feedback loop runs inside the Trainer and continuously optimises
parallelism and memory layout without user intervention.

```python
trainer = zenyx.train(model, dataloader, context_len=131072)

# The Trainer automatically:
# 1. Profiles every training step via AsyncProfiler (< 1% overhead)
# 2. Uses ParallelismPlanner to compute TP/PP/DP/Ring degrees at init
# 3. Runs TrainingController.step() each iteration for advisory replanning
# 4. Logs plan changes at WARNING level — no manual tuning needed

state = trainer.get_state()
print(state["parallelism_plan"])  # Current TP/PP/DP/Ring layout
print(state["profiler_stats"])    # Per-op timing statistics
```

Key components:

| Component             | Location                          | Purpose                                     |
| --------------------- | --------------------------------- | ------------------------------------------- |
| `AsyncProfiler`       | `zenyx/core/agent/profiler.py`    | Lightweight CUDA event profiler             |
| `ParallelismPlanner`  | `zenyx/core/agent/planner.py`     | Auto-determine TP/PP/DP/Ring degrees        |
| `TrainingController`  | `zenyx/core/agent/controller.py`  | Replan at curriculum shifts / every N steps  |

## Phase 6 — Fast Model Loader

Hardware-aware triple-buffered checkpoint loading. Automatically selects
GPU Direct Storage (CUDA), pread + DMA (TPU), or mmap (CPU).

```python
import zenyx

# One-line fast loading
model = zenyx.load_model("checkpoint.pt", model, dtype="bfloat16")

# Or via the Trainer with LoaderConfig
from zenyx.loader import LoaderConfig

trainer = zenyx.train(
    model,
    dataloader,
    resume_from="checkpoint.pt",
    loader_config=LoaderConfig(
        num_buffers=3,
        use_gpu_direct=True,
        dtype="bfloat16",
    ),
)
```

| Feature              | CUDA                        | TPU              | CPU          |
| -------------------- | --------------------------- | ---------------- | ------------ |
| I/O strategy         | GDS / pread + cudaMemcpy    | pread + DMA      | mmap         |
| Triple-buffering     | ✅                           | ✅ (thread pool)  | N/A          |
| Rollback on failure  | ✅                           | ✅                | ✅            |
| Integrity validation | ✅                           | ✅                | ✅            |

## Phases 7–10 (v1.0.0) — Advanced Training Features

### Phase 7: Bélády-Optimal KV Cache Tiering

Three-tier (T0=HBM, T1=DRAM, T2=NVMe) KV cache manager for ring attention training.
Uses the deterministic ring rotation schedule to compute Bélády-optimal eviction
over the combined forward + backward timeline.

### Phase 8: FP8 KV Quantization

FP8 E4M3 quantization of K and V tensors during training with per-channel dynamic
scaling for K and per-token dynamic scaling for V. Includes Smooth-SwiGLU to prevent
outlier amplification and gradient monitoring with STE (Straight-Through Estimator).

### Phase 9: Dynamic Ring Degree

Live resharding of the sequence dimension as context length grows during curriculum
training (8K → 32K → 128K → 512K → 1M tokens). Exponential step-wise doubling with
automatic convergence detection and PRNG key realignment.

### Phase 10: Sparse Ring Attention

Block-sparse ring attention with hybrid local + strided topology. Skips HBM loads
entirely for masked blocks using Pallas scalar prefetch API. Production skip fraction
is 62.5% (5/8 blocks skipped per device).

### Quickstart with Phase 7–10 Features

```python
import zenyx

trainer = zenyx.train(
    model,
    dataloader,
    context_len=131072,
    dtype="bfloat16",
    # Phase 8: FP8 KV quantization with per-channel scaling
    fp8_kv=True,
    fp8_quant_strategy="per_channel",
    # Phase 10: Sparse ring attention
    sparse_attn=True,
    sparse_skip_mode="production",
)
```

All Phase 7–10 features can be combined. See `zenyx/docs/dispute_resolutions.md`
for research validation details and `zenyx/docs/performance_ceiling.md` for
throughput analysis.

## Installation

```bash
pip install -e .

# With TPU support
pip install -e ".[tpu]"

# Full (all optional dependencies)
pip install -e ".[full]"
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.5.0
- CUDA ≥ 12.4 (for GPU features)

## Research References

- [QuEST (arXiv 2502.05003)](https://arxiv.org/abs/2502.05003) — INT8 activation unsafety proof
- [COAT (ICLR 2025)](https://openreview.net/forum?id=COAT) — FP8 E4M3 activation safety
- [Braided TP+PP (arXiv 2510.27257)](https://arxiv.org/abs/2510.27257) — Zero-bubble pipeline
- [DynaPipe (NeurIPS 2025)](https://arxiv.org/abs/2311.10418) — Safe pipeline reassignment
- [Ring Attention](https://arxiv.org/abs/2310.01889) — Sequence-parallel attention
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Distributed vocabulary parallelism
- [FlashAttention-3](https://arxiv.org/abs/2407.08691) — Hardware-aware attention

## License

Apache 2.0
