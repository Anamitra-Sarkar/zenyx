# Performance Ceiling Analysis — Zenyx 1T Model on 8-chip TPU v5e

## Peak Theoretical Throughput

At 65% MFU (Model FLOPs Utilization): ~170 tokens/sec on 8-chip TPU v5e.

## FLOPs Budget

- Total FLOPs per token for a 1T-parameter model: ~6×10¹² FLOPs
- This includes forward pass (~2×10¹² per token), backward pass (~4×10¹² per token)
- Per-step at 128K context: ~7.68×10¹⁷ FLOPs

## The Real Bottleneck: Weight Streaming

The dominant bottleneck is **weight streaming from NVMe**, NOT attention computation.

- Model weights: ~2 TB (1T parameters × 2 bytes BF16)
- PCIe Gen4 NVMe bandwidth: ~7.5 GB/s
- Time to stream all weights (forward + backward): 2 TB / 7.5 GB/s = **266 seconds** per pass
- This is a hard floor — no amount of attention optimization can overcome it

### Compute vs. Memory Bound Analysis

| Component | Time (seconds) | Bottleneck? |
|-----------|---------------|-------------|
| Weight streaming (NVMe→HBM) | 266 | **YES** |
| KV cache transfer | ~50 | Partial |
| Attention FLOPs | ~30 | No |
| AllReduce gradient sync | ~5 | No |

## Required Fixes to Break the Ceiling

### Option A: Multi-Node Pipeline Parallelism (Recommended)
- Spread 2 TB weights across 64+ chips (32 GB HBM each in v5p)
- Each chip holds ~31 GB of weights — fits entirely in HBM
- Eliminates NVMe weight streaming entirely
- Requires 8 nodes × 8 chips = 64 chips minimum

### Option B: Pin Weights in CPU DRAM
- Host with ≥2.5 TB RAM (2 TB weights + overhead)
- CPU DRAM bandwidth: ~200 GB/s (DDR5)
- Weight streaming time: 2 TB / 200 GB/s = 10 seconds
- 26× faster than NVMe, but still significant

## Impact of Phase 7-10 Optimizations

| Phase | Optimization | Impact on Weight Bottleneck |
|-------|-------------|---------------------------|
| Phase 7: KV Cache Tiering | Manages KV cache across HBM/DRAM/NVMe | Does NOT fix weight streaming |
| Phase 8: FP8 KV Quantization | Halves KV cache memory | Does NOT fix weight streaming |
| Phase 9: Dynamic Ring Degree | Adapts context parallelism | Does NOT fix weight streaming |
| Phase 10: Sparse Ring Attention | Skips 62.5% of attention blocks | Does NOT fix weight streaming |

**Conclusion**: Phases 7-10 are necessary for memory capacity and attention efficiency,
but they are **not sufficient** for the 165K tokens/step target at 1T scale.
The weight streaming bottleneck must be addressed via multi-node parallelism or
large-memory hosts.

## NVMe Bandwidth Requirement

- Minimum NVMe bandwidth at 128K context: 8.48 GB/s
- Single PCIe Gen4 NVMe (7.5 GB/s): FAILS
- PCIe Gen5 NVMe (~14 GB/s): PASSES
- RAID-0 of 2× Gen4 NVMe (~15 GB/s): PASSES
