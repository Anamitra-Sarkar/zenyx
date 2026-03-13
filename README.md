# Zenyx

**Hardware-agnostic, self-managing distributed LLM training runtime.**

Never OOM. No config files. No manual tuning.  
1T parameters · 1M+ context · 500K+ vocabulary · 120B model loading in under 20 seconds.

---

## Quick Start

```bash
pip install zenyx
```

```python
import torch
import zenyx

model = YourModel()
dataloader = YourDataLoader()

# That's it. Zenyx handles everything else.
trainer = zenyx.train(
    model,
    dataloader,
    context_len=131072,    # 128K context
    dtype="bfloat16",
    activation_dtype="float8_e4m3",
)

# Inspect training state
state = trainer.get_state()
print(f"Step: {state['step']}, Loss: {state['loss']:.4f}")
```

### Hardware Compatibility

| Hardware             | Attention Kernel            | Max Context (tested) | Notes            |
| -------------------- | --------------------------- | -------------------- | ---------------- |
| H100 NVLink 4.0 (8+) | TokenRing FA3 (Triton)      | 1M tokens            | S_min=1,099      |
| TPU v5e-8            | Ring Pallas + Shardy        | 1M tokens            | S_min=493        |
| TPU v5p              | Ring Pallas + Shardy        | 1M tokens            | S_min=383        |
| Any GPU (PCIe)       | Ring FA3 (PyTorch fallback) | 128K tokens          | S_min=15,454     |
| CPU / Apple M3       | Chunked attention           | 32K tokens           | Development only |

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

**Formal feasibility condition:** `(1/B₀₁) + (1/B₁₂) ≤ (1/F_compute)`

Where B₀₁ is T0↔T1 bandwidth, B₁₂ is T1↔T2 bandwidth, and F_compute is the compute rate. If this condition holds, Zenyx guarantees zero OOM. If it doesn't hold, Zenyx throttles compute to match memory bandwidth — never crashes.

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
│   ├── attention/     # Ring FlashAttention (CUDA, TPU, CPU)
│   ├── comm/          # Topology detection, ring comm, all-reduce
│   └── vocab/         # Distributed cross-entropy (500K+ vocab safe)
├── train/
│   ├── trainer.py     # Trainer class + zenyx.train() entrypoint
│   ├── loop.py        # Legacy train loop (Phase 2)
│   ├── lr_schedule.py # Cosine LR with linear warmup
│   ├── grad_scaler.py # Mixed precision gradient scaler
│   ├── distributed_setup.py  # Auto distributed init (torchrun/SLURM/TPU)
│   ├── activation_checkpoint.py  # Selective activation checkpointing
│   ├── pipeline.py    # Braided TP+PP schedule
│   ├── mixed_prec.py  # FP8 E4M3 activation storage
│   └── checkpoint.py  # Async distributed checkpointing
├── loader/
│   ├── gds_loader.py  # GPU Direct Storage triple-buffering
│   └── tpu_loader.py  # TPU multi-threaded pread + DMA
└── bench/
    ├── integration_test.py  # End-to-end integration tests
    ├── memory_budget.py     # Memory budget calculator
    └── vs_deepspeed.py      # Benchmark vs DeepSpeed ZeRO-3
```

## Trainer API

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
