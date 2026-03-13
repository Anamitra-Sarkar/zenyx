# Zenyx

**Hardware-agnostic, self-managing distributed LLM training runtime.**

Never OOM. No config files. No manual tuning.  
1T parameters · 1M+ context · 500K+ vocabulary · 120B model loading in under 20 seconds.

---

## What is Zenyx?

Zenyx is a Python library that sits below the model definition and above the hardware. You write a model and call `zenyx.train(model, dataloader)`. Zenyx handles everything else — memory layout, kernel dispatch, parallelism, checkpointing, communication.

```python
import zenyx

# That's it. Zenyx handles memory, parallelism, everything.
zenyx.train(model, dataloader)
```

## Key Features

### Never-OOM Guarantee
Three-tier memory system (VRAM → CPU DRAM → NVMe SSD) with a formal feasibility condition. If the hardware can theoretically fit the workload, Zenyx will train it without crashing. If it can't, Zenyx throttles — never crashes.

**Formal condition:** `(1/B₀₁) + (1/B₁₂) ≤ (1/F_compute)`

### Hardware Agnostic
- **NVIDIA CUDA** — cuBLAS + FlashAttention-3 + NCCL + cuFile GDS
- **AMD ROCm** — hipBLASLt + MIOpen + RCCL
- **Google TPU** — Pallas + Shardy + JAX custom_partitioning + ICI
- **CPU** — OpenBLAS + AVX-512
- **Apple Metal/MLX** — Inference and fine-tuning

Five HAL primitives: `alloc`, `free`, `copy`, `matmul`, `reduce` — everything in Zenyx is composed from these.

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
│   ├── loop.py        # Main train() entrypoint
│   ├── pipeline.py    # Braided TP+PP schedule
│   ├── mixed_prec.py  # FP8 E4M3 activation storage
│   └── checkpoint.py  # Async distributed checkpointing
├── loader/
│   ├── gds_loader.py  # GPU Direct Storage triple-buffering
│   └── tpu_loader.py  # TPU multi-threaded pread + DMA
└── bench/
    ├── memory_budget.py  # Memory budget calculator
    └── vs_deepspeed.py   # Benchmark vs DeepSpeed ZeRO-3
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

Output:
```
╔══════════════════════════════════════════════════╗
║           Zenyx Memory Budget Report             ║
╠══════════════════════════════════════════════════╣
║ Model: 7.0B params                               ║
║ Vocab: 152,000 | Context: 8,192                  ║
║ Hardware: H100 (80 GB HBM3)                      ║
╠══════════════════════════════════════════════════╣
║ Component         │ Size (GB)                     ║
║ Weights           │ 14.00                         ║
║ Activations       │ 28.00                         ║
║ KV Cache          │  X.XX                         ║
║ Optimizer States  │ 28.00                         ║
╠══════════════════════════════════════════════════╣
║ Load time: X.Xs | Throughput: X,XXX tok/s         ║
║ OOM-free: ✅                                      ║
╚══════════════════════════════════════════════════╝
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
