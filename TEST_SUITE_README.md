# Zenyx Hardware Training Validation

This directory contains comprehensive tests to validate Zenyx's training capabilities across different hardware backends.

## What is Zenyx?

Zenyx is a **hardware-agnostic, self-managing distributed LLM training runtime** that claims:

- ✅ **1T parameters** • **1M+ context** • **500K+ vocabulary**
- ✅ **Never-OOM guarantee** via three-tier memory hierarchy (HBM/DRAM/NVMe)
- ✅ **Hardware agnostic**: CUDA, ROCm, TPU/XLA, CPU
- ✅ **Auto-parallelism planning**: Tensor Parallel (TP) + Pipeline Parallel (PP) + Data Parallel (DP)
- ✅ **Ring Attention** for extreme sequence lengths

---

## Test Suite

### 1. CPU Training Test ✅

**File:** `test_cpu_8k_context.py`

**What it tests:**
- CPU-only training with up to 8K token context
- Per README: "CPU / Apple M3: Chunked attention, 32K tokens - Development only"
- Validates that Zenyx uses chunked attention fallback on CPU
- Tests context scaling: 1K → 2K → 4K → 8K tokens

**How to run:**
```bash
python test_cpu_8k_context.py
```

**Expected output:**
```
✅ CPU (8K context)
  Time: ~6 seconds
  Steps: 3/3 completed
  Loss: Training loss tracked
  Parallelism: TP=1 PP=1 DP=1 (single CPU)
```

**Status:** ✅ **PASSING** - Verified that CPU training works with chunked attention

---

### 2. GPU Training Test 🔷

**File:** `test_gpu_128k_context.py`

**What it tests:**
- Distributed training on **2x NVIDIA Tesla T4 GPUs**
- Per README: "Any GPU (PCIe): Ring FA3 (PyTorch fallback), 128K tokens"
- Tests ring attention with 128K context
- Auto-parallelism planning across 2 GPUs

**How to run:**
```bash
# Single GPU test
python test_gpu_128k_context.py

# Multi-GPU test (2xT4)
torchrun --nproc_per_node=2 test_gpu_128k_context.py
```

**Requirements:**
- CUDA 12.4+
- 2x NVIDIA T4 GPUs (24GB VRAM each)

**Expected output:**
```
✅ GPU (128K context, 2xT4)
  Time: <60 seconds
  Steps: 5/5 completed
  Context: 131,072 tokens
  Parallelism: TP=? PP=? DP=?
  Throughput: High
```

**Status:** ⏭️ **NOT TESTED YET** - Requires NVIDIA GPU + CUDA

---

### 3. TPU Training Test 🚀

**File:** `test_tpu_1m_context.py`

**What it tests:**
- Extreme-scale training on **Google Cloud TPU v5e-8**
- Per README: "TPU v5e-8: Ring Pallas + Shardy, 1M tokens"
- Claims: **1T parameters with 1M context length**
- Validates Ring Pallas attention and Shardy collective ops

**How to run:**
```bash
# On TPU v5e-8 instance with torch_xla
python test_tpu_1m_context.py
```

**Requirements:**
- Google Cloud TPU v5e-8 (8 cores)
- `torch_xla` installed: `pip install torch_xla[tpu]`
- XLA devices available

**Expected output:**
```
✅ TPU (1M context, v5e-8)
  Time: <5 minutes
  Steps: 5/5 completed
  Context: 1,000,000 tokens
  Model Params: ~1T (1 trillion)
  Vocab: 262K
  Throughput: 500K+ tokens/sec
  Parallelism: Ring Degree adjusts dynamically
```

**Status:** ⏭️ **NOT TESTED YET** - Requires Google Cloud TPU + torch_xla

---

## Running All Tests

### Quick validation (CPU only):
```bash
python validate_hardware.py
```

### Full suite:
Run each test individually on the appropriate hardware:
```bash
# CPU (always available)
python test_cpu_8k_context.py

# GPU (requires 2xT4)
python test_gpu_128k_context.py

# TPU (requires v5e-8)
python test_tpu_1m_context.py
```

---

## Key Claims Validated

| Capability | Hardware | Context | Parameters | Status | Test |
|-----------|----------|---------|-----------|--------|------|
| Chunked Attention | CPU | 8K | 6.5M | ✅ Pass | `test_cpu_8k_context.py` |
| Ring Attention (FA3) | GPU (PCIe) | 128K | - | ⏭️ Pending | `test_gpu_128k_context.py` |
| Ring Pallas + Shardy | TPU v5e-8 | 1M | **1T** | ⏭️ Pending | `test_tpu_1m_context.py` |

---

## Memory Management Validation

All tests verify Zenyx's **never-OOM guarantee**:

1. **Three-tier memory hierarchy**:
   - T0: HBM/VRAM (fastest)
   - T1: CPU DRAM (medium)
   - T2: NVMe SSD (slowest)

2. **Bélády-optimal eviction**: Evicts tensor whose next use is farthest in future

3. **Async prefetching**: Overlaps computation with memory transfers

4. **FP8 activation storage**: 2x memory savings with per-layer quantization

Tests validate that no `OutOfMemoryError` occurs even when:
- Extreme context lengths (up to 1M tokens)
- Massive batch sizes implied by parallelism
- Deep networks with billions of parameters

---

## Integration with CI/CD

These tests can be integrated into CI/CD:

```yaml
# Example GitHub Actions
- name: CPU Training Test
  run: python test_cpu_8k_context.py

- name: GPU Training Test (if available)
  run: python test_gpu_128k_context.py
  if: ${{ runner.gpu }}

- name: TPU Training Test (if available)
  run: python test_tpu_1m_context.py
  if: ${{ runner.tpu }}
```

---

## Interpretation of Results

### ✅ PASS
- Training completed without errors
- Loss computed and reduced
- Parallelism plan auto-generated
- Memory stayed within tier bounds
- No OOM crashes

### ⏭️ SKIPPED
- Hardware not available (normal for CPU-only machines)
- Required dependencies not installed (e.g., `torch_xla` for TPU)

### ❌ FAIL
- Training crashed or timed out
- OOM or memory errors
- Loss computation failed
- Parallelism planner failed

---

## Debugging Failed Tests

If a test fails:

1. **Check hardware setup:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
   ```

2. **Check dependencies:**
   ```bash
   python -c "import torch_xla"  # For TPU
   python -c "import zenyx; print(zenyx.__version__)"
   ```

3. **Run with full logging:**
   ```bash
   python test_gpu_128k_context.py 2>&1 | tee debug.log
   ```

4. **Check Zenyx logs:**
   - Look for parallelism planner warnings
   - Check memory tier transitions
   - Validate attention kernel selection

---

## References

- **Zenyx GitHub**: https://github.com/Anamitra-Sarkar/zenyx
- **README**: See `../README.md` for full documentation
- **Performance Analysis**: See `../zenyx/docs/performance_ceiling.md`
- **Validation Details**: See `../zenyx/docs/dispute_resolutions.md`
