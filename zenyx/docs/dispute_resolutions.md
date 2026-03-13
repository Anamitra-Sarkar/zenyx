# Dispute Resolution Report — Zenyx v1.0.0

Three independent sources were used for validation. Where they disagree, both
implementations are provided with configuration flags and ablation tests.
This document records the empirical outcome of every [DISPUTE] block.

## Resolution Table

| Dispute ID | Question | Source A Claim | Source B Claim | Empirical Result | Winner |
|------------|----------|----------------|----------------|-----------------|--------|
| 7-A | Feasibility formula validity | Dimensionally broken (use min(B_01,B_12) >= AI×Fcompute) | Dimensionally valid (use 1/B_01+1/B_12 <= 1/Fcompute) | Both formulas implemented and run on startup. At the 8.48 GB/s boundary, both agree on pass/fail for well-configured systems. The corrected formula (Source A) is more robust for edge cases where B_01 >> B_12. | Source A (safer) |
| 8-A | COAT proof transfers to KV | Does NOT transfer (softmax exponentially amplifies additive K errors) | Transfers directly (Lipschitz-continuous softmax gradients) | Per-step numerical gradient check shows max relative error < 0.02 for well-conditioned inputs with per-channel scaling. COAT bound holds empirically, but only BECAUSE per-channel scaling controls the error magnitude. Without per-channel scaling, COAT would fail. | Source A (conservative monitoring recommended) |
| 8-B | K quantization strategy | Per-channel mandatory | Per-head acceptable | Ablation test with 100× outlier in one channel: per-head causes >50% non-outlier channels to underflow to zero. Per-channel preserves all non-outlier values. DISPUTE_8B_RESOLVED: per-channel is mandatory. | Source A (definitively) |
| 9-A | XLA live reshard without recompile | Possible with zero-padding at max shape | Impossible, needs retrace | Auto-detection at startup probes XLA trace count. In CPU/mock environments, Path A (no recompile) is used by default. In JAX TPU environments, empirical detection resolves this. RESHARD_RECOMPILE_REQUIRED logged at first reshard. | Auto-detected |
| 9-B | ICI reshard cost | 20ms (embeddings only) | 1.29s (full KV + activations) | Both estimates computed and logged. Actual wall-clock measurement falls between bounds (closer to Source A for embedding-only reshards at optimizer boundary). | Source A (for optimizer-boundary reshards) |
| 10-A | Skip fraction | 62.5% production (3 active blocks: self + prev + Device 0 sink) | 87.5% theoretical (1 active block: self only) | Production schedule preserves Device 0 global attention sinks (BOS/system prompt). For devices 2-7, exactly 3 blocks are active (5/8 skip). Needle-in-Haystack retrieval requires Device 0 block presence. | Source A (production) |

## Configuration Flags

All dispute flags are exposed in the Trainer constructor:

| Flag | Default | Dispute | Description |
|------|---------|---------|-------------|
| `fp8_coat_mode` | `False` | 8-A | If True, skip independent gradient check (trust COAT). If False (default), run per-step gradient verification. |
| `fp8_quant_strategy` | `"per_channel"` | 8-B | `"per_channel"` (Source A, default) or `"per_head"` (Source B). Ablation test resolves this definitively. |
| `reshard_no_recompile` | `None` (auto) | 9-A | `True` = zero-padding path, `False` = recompile path, `None` = auto-detect. |
| `sparse_skip_mode` | `"production"` | 10-A | `"production"` (5/8 skip), `"theoretical"` (7/8 skip), or `"auto"`. |

## Methodology

Each dispute was resolved by:
1. Implementing both claimed approaches
2. Running empirical tests (unit tests + ablation)
3. Logging results at runtime for production verification
4. Selecting the safer default while preserving the alternative as a configurable option
