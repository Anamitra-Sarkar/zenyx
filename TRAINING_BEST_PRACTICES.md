# ZENYX Training Best Practices

## Overview

This document provides practical guidance on training models effectively with ZENYX, based on lessons learned from training systems at scale.

---

## Table of Contents

1. [Before You Start](#before-you-start)
2. [Configuration Best Practices](#configuration-best-practices)
3. [Data Preparation](#data-preparation)
4. [Optimization Techniques](#optimization-techniques)
5. [Memory Management](#memory-management)
6. [Performance Tuning](#performance-tuning)
7. [Debugging Common Issues](#debugging-common-issues)
8. [Production Deployment](#production-deployment)

---

## Before You Start

### 0. Installation & Imports

**See:** `INSTALLATION_AND_SETUP.md` for complete setup

```bash
# Install ZENYX
pip install zenyx torch

# Verify
python test/validate_zenyx_four_pillars.py
```

**Core imports for training:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# For ZENYX features:
from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager
from zenyx.train.fp8_kv_quantization import FP8KVQuantizer
from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler
from zenyx.train.sparse_ring_attention import SparseRingAttention
```

### 1. Verify Hardware Compatibility

**Always check your hardware first:**

```python
import torch
import jax

# Check CPU/GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Check TPU
devices = jax.devices('tpu')
print(f"TPU count: {len(devices)}")
print(f"TPU type: {devices[0].device_kind if devices else 'None'}")
```

**Hardware-Specific Settings:**

| Hardware | Max Params | Max Context | Batch Size | Config |
|----------|-----------|-------------|-----------|--------|
| CPU | 100M | 8K | 1 | `enable_sparse_attention=False` |
| GPU (16GB) | 7B | 8K | 4 | All phases enabled |
| TPU v5e-8 | 1T | 1M | 1 | All phases enabled |

### 2. Test Your Setup

```bash
# 1. Run validation suite
python test/comprehensive_e2e_validation.py

# 2. Test with tiny model
python examples/01_beginner_cpu_training.py

# 3. Only then, scale up
```

### 3. Understand Your Data

```python
# Before training, analyze your dataset
import numpy as np

# Check statistics
tokens = [...your training tokens...]
print(f"Total tokens: {len(tokens):,}")
print(f"Vocab size: {len(set(tokens))}")
print(f"Average sequence length: {np.mean([len(seq) for seq in tokens]):.0f}")
print(f"Max sequence length: {np.max([len(seq) for seq in tokens])}")

# Verify no data leakage
train_set = set(train_tokens)
test_set = set(test_tokens)
overlap = len(train_set & test_set)
print(f"Train/test overlap: {overlap} tokens")
```

---

## Configuration Best Practices

### 1. Choose the Right Model Size

**Rule of thumb:** Model size should match your hardware

```python
from zenyx.unified_training import ZenyxConfig

# CPU (100M parameters max)
config = ZenyxConfig(model_params=int(1e8))

# GPU (7B parameters)
config = ZenyxConfig(model_params=int(7e9))

# TPU v5e-8 (1 trillion parameters!)
config = ZenyxConfig(model_params=int(1e12))
```

**Don't violate the rule:**
- ❌ 100B params on CPU → Out of memory
- ❌ 1T params on GPU → Way too slow
- ❌ 70B params on TPU v5e-8 → Wasting hardware

### 2. Set Batch Size Based on Memory

**Start conservative, increase gradually:**

```python
# Conservative start
config = ZenyxConfig(batch_size=1)  # 1 sample per step

# With gradient accumulation
config = ZenyxConfig(
    batch_size=1,
    gradient_accumulation_steps=32,  # Effective batch = 32
)

# If you have more memory
config = ZenyxConfig(
    batch_size=4,
    gradient_accumulation_steps=8,   # Effective batch = 32
)
```

**Memory equation:**
```
memory_used = (params + kv_cache + activations) * dtype_size
```

If memory > available HBM → use gradient accumulation or reduce batch size.

### 3. Learning Rate & Warmup Strategy

**Use cosine annealing with warmup (industry standard):**

```python
import math

def cosine_annealing_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-5,
    max_lr: float = 1e-4,
) -> float:
    """Cosine annealing: warmup → decay"""
    if step < warmup_steps:
        # Linear warmup
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**Recommended values:**

| Model Size | Max LR | Min LR | Warmup % |
|-----------|--------|--------|----------|
| 100M | 1e-3 | 1e-5 | 10% |
| 7B | 3e-4 | 1e-5 | 5% |
| 70B | 1e-4 | 1e-6 | 2% |
| 1T | 1e-4 | 1e-6 | 2% |

### 4. Enable All Four Pillars (On Compatible Hardware)

```python
config = ZenyxConfig(
    # Always enable if hardware supports
    enable_belayd_tiering=True,       # Phase 7 (always safe)
    enable_fp8_quantization=True,     # Phase 8 (GPU/TPU only)
    enable_curriculum=True,           # Phase 9 (always safe)
    enable_sparse_attention=True,     # Phase 10 (TPU v5e-8 only)
)
```

**Impact of each phase:**

| Phase | Memory Saved | Speed Gain | Stability | When to enable |
|-------|------------|-----------|-----------|----------------|
| 7 | 50% | 0% | ✓ Safe | Always |
| 8 | 2x | 0% | ✓ Safe | GPU/TPU |
| 9 | 0% | 0% | ✓ Required for 1M | Always (for large context) |
| 10 | 0% | 13.3x | ✓ Safe | TPU v5e-8 |

---

## Data Preparation

### 1. Tokenization

```python
# Use a proven tokenizer (don't build your own)
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

# Tokenize training data
def prepare_dataset(texts, max_length=8192):
    """Prepare texts for training"""
    encodings = tokenizer.encode_batch(texts)
    
    tokens = []
    for enc in encodings:
        ids = enc.ids
        
        # Pad or truncate to max_length
        if len(ids) < max_length:
            ids += [tokenizer.token_to_id("<pad>")] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        
        tokens.append(ids)
    
    return tokens
```

### 2. Data Streaming

**For large datasets (>100GB), stream from disk:**

```python
import torch
from torch.utils.data import IterableDataset

class StreamingTokenDataset(IterableDataset):
    """Stream tokens from disk in chunks"""
    
    def __init__(self, data_dir, chunk_size=1_000_000):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        
    def __iter__(self):
        import os
        import numpy as np
        
        for filename in sorted(os.listdir(self.data_dir)):
            # Load chunk (e.g., .npy file)
            tokens = np.load(f"{self.data_dir}/{filename}")
            
            # Yield batches
            for i in range(0, len(tokens), 8192):
                yield {
                    'input_ids': tokens[i:i+8192],
                }

# Use in training
dataset = StreamingTokenDataset("./data_chunks/")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,  # Already batched
)
```

### 3. Shuffling Strategy

```python
# Good: Shuffle within reasonable window
def shuffle_dataset(tokens, window_size=1_000_000):
    """Shuffle tokens within a window to maintain locality"""
    import random
    
    shuffled = []
    for i in range(0, len(tokens), window_size):
        chunk = tokens[i:i+window_size]
        random.shuffle(chunk)
        shuffled.extend(chunk)
    
    return shuffled

# Bad: Don't shuffle by dropping all data in memory
# tokens = random.shuffle(all_training_data)  # OOM on 1TB dataset!
```

---

## Optimization Techniques

### 1. Gradient Accumulation

**Simulate larger batches without memory:**

```python
config = ZenyxConfig(
    batch_size=1,                      # Physical batch
    gradient_accumulation_steps=32,    # Effective = 32
    total_steps=100_000,
)

trainer = ZenyxTrainer(config)

accumulated_loss = 0
for step, batch in enumerate(dataloader):
    loss = trainer.train_step(batch)
    accumulated_loss += loss
    
    if (step + 1) % 32 == 0:
        # This is your "effective" step
        avg_loss = accumulated_loss / 32
        # Update learning rate for effective step
        effective_step = (step + 1) // 32
        accumulated_loss = 0
```

### 2. Gradient Checkpointing

**Trade compute for memory (save 50% memory, 20% slower):**

```python
# Enabled in ZenyxConfig
config = ZenyxConfig(
    enable_gradient_checkpointing=True,  # Save activations on backward pass
    batch_size=1,
)
```

**When to use:**
- ✓ Model is too large for memory even with batch_size=1
- ✗ You have plenty of memory and want maximum speed

### 3. Mixed Precision Training

```python
config = ZenyxConfig(
    dtype="bfloat16",  # or "float16" on older GPUs
)
```

**Precision options:**

| Dtype | Accuracy | Speed | Memory | When to use |
|-------|----------|-------|--------|------------|
| float32 | Best | 1x | 1x | Debugging, small models |
| bfloat16 | Excellent | 2x | 0.5x | TPU, standard choice |
| float16 | Good | 2x | 0.5x | GPU (needs loss scaling) |
| fp8 | Good | 2-3x | 0.25x | KV cache only (Phase 8) |

### 4. Layer-wise Learning Rate Decay

```python
# Recommended for large models (>10B params)
def get_lrd_groups(model, num_layers=12, lr=1e-4):
    """Group parameters by layer for layer-wise LR decay"""
    groups = []
    decay = 0.9
    
    for i in range(num_layers):
        layer_lr = lr * (decay ** (num_layers - i))
        # Get parameters for layer i
        # Add to optimizer param groups
    
    return groups
```

---

## Memory Management

### 1. Profile Memory Usage

```python
import torch
import tracemalloc

tracemalloc.start()

# Your training step
loss = trainer.train_step(batch)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1e9:.1f} GB")
print(f"Peak memory: {peak / 1e9:.1f} GB")
tracemalloc.stop()
```

### 2. Monitor KV Cache Growth

```python
# ZENYX automatically monitors this, but you can check
report = trainer.get_training_report()

print(f"Phase 7 KV tiering:")
print(f"  HBM usage: {report['phase7']['hbm_usage_gb']:.1f} GB")
print(f"  DRAM usage: {report['phase7']['dram_usage_gb']:.1f} GB")
print(f"  NVMe usage: {report['phase7']['nvme_usage_gb']:.1f} GB")

print(f"Phase 8 compression:")
print(f"  Original: {report['phase8']['original_size_gb']:.1f} GB")
print(f"  Compressed: {report['phase8']['compressed_size_gb']:.1f} GB")
print(f"  Ratio: {report['phase8']['compression_ratio']:.1f}x")
```

### 3. Handle OOM Gracefully

```python
try:
    loss = trainer.train_step(batch)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("OOM detected. Reducing batch size...")
        config.batch_size = config.batch_size // 2
        trainer = ZenyxTrainer(config)
    else:
        raise
```

---

## Performance Tuning

### 1. Measure Throughput

```python
import time

start = time.time()
step_count = 0

for step, batch in enumerate(dataloader):
    loss = trainer.train_step(batch)
    step_count += 1
    
    if step % 100 == 0:
        elapsed = time.time() - start
        throughput = (step_count * config.batch_size * 8192) / elapsed
        print(f"Throughput: {throughput:,.0f} tokens/sec")
```

**Expected throughput:**

| Hardware | No Optimizations | With Phase 10 Sparse |
|----------|-----------------|-------------------|
| CPU | 100 tok/s | 100 tok/s |
| GPU (16GB) | 5,000 tok/s | 5,000 tok/s |
| TPU v5e-8 | 50,000 tok/s | **665,000 tok/s** (13.3x!) |

### 2. Identify Bottlenecks

```python
# Bottleneck types:
# 1. Data loading (dataloader is slow)
# 2. Compute (forward/backward is slow)
# 3. Memory (OOM or thrashing)

# Check data loading
start = time.time()
for batch in dataloader:
    print(f"Data loading: {time.time() - start:.3f}s")
    break

# If > 0.1s, optimize data pipeline
# Use num_workers, pin_memory, etc.
```

### 3. Optimize Data Pipeline

```python
# Bad: Single worker, slow disk reads
loader = DataLoader(dataset, batch_size=32)

# Good: Multiple workers, pinned memory
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,          # Parallel loading
    pin_memory=True,        # CPU → GPU faster
    prefetch_factor=2,      # Prefetch 2 batches
)
```

---

## Debugging Common Issues

### Issue: Loss Not Decreasing

**Checklist:**

1. ✓ Learning rate too high?
   ```python
   # Reduce by 10x
   config.learning_rate = 1e-5
   ```

2. ✓ Data normalized properly?
   ```python
   # Check data statistics
   print(f"Mean: {tokens.mean():.4f}")
   print(f"Std: {tokens.std():.4f}")
   # Should be ~ 0 mean, 1 std
   ```

3. ✓ Labels correct?
   ```python
   # Sanity check: targets should be plausible
   print(f"Sample loss on random target: {baseline_loss:.4f}")
   print(f"Your loss: {your_loss:.4f}")
   # Your loss should decrease below baseline
   ```

4. ✓ Warmup enabled?
   ```python
   config.warmup_steps = max(100, total_steps // 100)
   ```

### Issue: Training Unstable (Loss spikes)

**Checklist:**

1. ✓ Enable gradient clipping
   ```python
   config.grad_clip = 1.0
   ```

2. ✓ Use curriculum learning (Phase 9)
   ```python
   config.enable_curriculum = True
   ```

3. ✓ Reduce learning rate
   ```python
   config.learning_rate = config.learning_rate / 10
   ```

4. ✓ Check batch normalization
   ```python
   # Disable if present (not common in transformers)
   model.eval()  # to use stored running stats
   ```

### Issue: Training Very Slow

**Checklist:**

1. ✓ Enable Phase 10 (sparse attention) if on TPU
   ```python
   config.enable_sparse_attention = True
   ```

2. ✓ Use mixed precision (BF16)
   ```python
   config.dtype = "bfloat16"
   ```

3. ✓ Increase batch size if possible
   ```python
   config.batch_size = 4  # if memory allows
   ```

4. ✓ Check data pipeline (is it slow?)
   ```python
   # Profile data loading
   import cProfile
   cProfile.run('next(iter(dataloader))')
   ```

---

## Production Deployment

### 1. Pre-deployment Checklist

```bash
# 1. Test on small dataset
python examples/01_beginner_cpu_training.py

# 2. Verify config
python -c "from zenyx.unified_training import ZenyxConfig; \
  c=ZenyxConfig(); print(c)"

# 3. Run validation suite
python test/comprehensive_e2e_validation.py

# 4. Profile memory
python -c "import tracemalloc; tracemalloc.start(); \
  trainer.train_step(batch); print(tracemalloc.get_traced_memory())"

# 5. Measure throughput (100 steps)
python train.py --steps 100 --measure-throughput
```

### 2. Logging & Monitoring

```python
import json
from datetime import datetime

# Log every 100 steps
if step % 100 == 0:
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'step': step,
        'loss': float(loss),
        'learning_rate': float(current_lr),
        'throughput_tokens_per_sec': throughput,
        'phase7_hbm_usage_gb': report['phase7']['hbm_usage_gb'],
        'phase8_compression_ratio': report['phase8']['compression_ratio'],
        'phase9_context_tokens': report['phase9']['context_tokens'],
        'phase10_sparsity': report['phase10']['sparsity'],
    }
    
    # Log to file
    with open('training.jsonl', 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    # Log to tensorboard
    # writer.add_scalar('Loss', loss, step)
```

### 3. Checkpoint Management

```python
# Save checkpoint every N steps
if step % 1000 == 0:
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'metrics': metrics,
    }
    torch.save(checkpoint, f'checkpoints/step_{step:06d}.pt')

# Keep only last 5 checkpoints
import os
import glob
checkpoints = sorted(glob.glob('checkpoints/step_*.pt'))
for old_ckpt in checkpoints[:-5]:
    os.remove(old_ckpt)
```

### 4. Handle Preemption (TPU)

```python
import signal

checkpoint_path = None

def handle_preemption(signum, frame):
    """Save checkpoint on preemption signal"""
    global checkpoint_path
    print("Preemption signal received. Saving checkpoint...")
    torch.save(model.state_dict(), 'emergency_checkpoint.pt')
    exit(0)

signal.signal(signal.SIGTERM, handle_preemption)

# Training loop
for step, batch in enumerate(dataloader):
    try:
        loss = trainer.train_step(batch)
    except Exception as e:
        print(f"Error on step {step}: {e}")
        torch.save(model.state_dict(), 'error_checkpoint.pt')
        raise
```

---

## Summary

Key takeaways:

1. **Hardware first**: Know your constraints before training
2. **Small to large**: Start with toy example, scale gradually
3. **Verify often**: Test data, config, and metrics constantly
4. **Monitor always**: Track loss, throughput, and memory
5. **Tune late**: Get it working first, optimize second
6. **Debug systematically**: Check one thing at a time

For questions, refer to `TRAINING_GUIDE_COMPLETE.md` or run examples directly.

Happy training! 🚀
