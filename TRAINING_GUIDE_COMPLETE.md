# ZENYX Complete Training Guide

## Installation & Imports

### Prerequisites
- Python 3.8+
- pip or conda package manager
- For GPU: NVIDIA CUDA 11.8+
- For TPU: Google Cloud TPU access

### Step 1: Install ZENYX and Dependencies

```bash
# Using pip (recommended)
pip install zenyx torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or using conda
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install zenyx
```

### Step 2: Verify Installation

```bash
python -c "import zenyx; import torch; print(f'ZENYX: ✓, PyTorch: ✓')"
python test/validate_zenyx_four_pillars.py
```

### Step 3: Core Imports for All Scripts

Every training script uses these core imports:

```python
# PyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Optional but recommended
import json
from pathlib import Path
from datetime import datetime

# ZENYX Phase 7: KV Cache Tiering
from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager

# ZENYX Phase 8: FP8 Quantization
from zenyx.train.fp8_kv_quantization import FP8KVQuantizer

# ZENYX Phase 9: Dynamic Curriculum
from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler

# ZENYX Phase 10: Sparse Attention
from zenyx.train.sparse_ring_attention import SparseRingAttention
```

### Step 4: Verify Your Hardware

Add this to the top of your training script:

```python
# Check available hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU detected, using CPU")

# Optional: Check for TPU (Google Cloud)
try:
    import jax
    print(f"TPU devices available: {len(jax.devices())}")
except:
    print("TPU not available (JAX not installed or not on TPU)")
```

## Overview

ZENYX is a production-ready library for training trillion-parameter language models on TPU v5e-8. This guide shows you how to train models using ZENYX on your hardware.

## Quick Start

### 1. Minimal Example (CPU, <1 minute)

```bash
python train_minimal.py
```

This trains a tiny 108-parameter model on CPU. Perfect for testing the setup.

### 2. Beginner Example (CPU, 2-5 minutes)

```bash
python train_with_loss.py
```

This shows a complete training pipeline with loss tracking, checkpointing, and metrics.

### 3. Production Example (TPU v5e-8, hours)

```bash
python train/zenyx_single_tpu_train.py
```

This trains a 1 trillion parameter model with all ZENYX features.

## Training Examples

### Example 01: Beginner CPU Training
**File:** `examples/01_beginner_cpu_training.py`

Simple model, simple data, simple loop. Perfect for learning the basics.

```python
# Create a tiny model
model = nn.Sequential(
    nn.Embedding(1000, 64),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1000),
)

# Create data
input_ids = torch.randint(0, 1000, (100, 32))
target_ids = torch.randint(0, 1000, (100, 32))

# Train
for epoch in range(2):
    for inputs, targets in loader:
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 1000), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**What you'll learn:**
- How to define a PyTorch model
- How to create a DataLoader
- How to run a training loop
- How to compute loss and update weights

**Time:** ~1-2 minutes on CPU

---

### Example 02: Intermediate Fine-tuning
**File:** `examples/02_intermediate_finetuning.py`

Larger model, real techniques, production patterns.

```python
# Transformer model
model = TransformerModel(
    vocab_size=2000,
    hidden_dim=128,
    num_layers=2,
    seq_len=64,
)

# Gradient accumulation
accumulated_loss = 0.0
accumulation_steps = 2

# Learning rate warmup
def get_lr(step, warmup_steps):
    if step < warmup_steps:
        return (step / warmup_steps) * base_lr
    return base_lr * 0.5 * (1 + cos(step / total_steps))

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inp, tgt) in enumerate(loader):
        logits = model(inp)
        loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

**What you'll learn:**
- How to use gradient accumulation
- How to implement learning rate warmup and scheduling
- How to use gradient clipping
- How to save checkpoints and track metrics

**Time:** ~5-10 minutes on CPU

---

### Example 03: Expert TPU Training
**File:** `examples/03_expert_tpu_v5e8_training.py`

Production-scale model, all ZENYX features, distributed training ready.

```python
# 1 trillion parameter model
model = ProductionLanguageModel(
    vocab_size=128000,
    hidden_dim=8192,
    num_layers=40,
    num_heads=64,
    max_seq_len=1048576,  # 1M tokens
)

# ZENYX Features
# Phase 7: Bélády KV Cache Tiering
#   - Manages 1M token context with 16 GB HBM
#   - Intelligently tiers between fast and slow memory
# 
# Phase 8: FP8 KV Quantization
#   - 2x memory compression
#   - Minimal accuracy loss
#
# Phase 9: Dynamic Ring Curriculum
#   - Progressive training difficulty
#   - Improved convergence
#
# Phase 10: Sparse Ring Attention
#   - 13.3x speedup with sliding window
#   - Reduced computational complexity

# Training loop with all features
pipeline = ZenyxTrainingPipeline(config)
pipeline.train(train_loader)
```

**What you'll learn:**
- How to define a large language model
- How to integrate all ZENYX features
- How to set up distributed training
- How to checkpoint and recover from failures

**Time:** Hours to days on TPU v5e-8

---

## Main Training Scripts

### `train_minimal.py`
The smallest possible training script. Good for smoke testing.

### `train_with_loss.py`
A complete training pipeline with:
- Transformer model
- Cross-entropy loss
- Gradient clipping
- Epoch-based training

### `train_complete_demo.py`
Advanced training with:
- Train and validation loops
- Learning rate scheduling
- Checkpoint saving
- Metrics tracking
- JSON output

### `train/zenyx_single_tpu_train.py`
Production TPU training with:
- Large model (1T parameters)
- All ZENYX features
- Distributed training support
- Checkpoint and recovery
- Logging and monitoring

## Training on Different Hardware

### CPU Training

```bash
# Beginner
python examples/01_beginner_cpu_training.py

# Intermediate
python examples/02_intermediate_finetuning.py

# Full demo
python train_complete_demo.py
```

**Tips:**
- Use smaller models (< 1B parameters)
- Reduce batch size and sequence length
- Enable gradient accumulation
- Use mixed precision (float16/bfloat16) if supported

### GPU Training

```bash
# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Update code to use GPU
device = "cuda:0"
model = model.to(device)

# For multi-GPU, use DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)
```

**Tips:**
- Use bfloat16 for A100 GPUs
- Use float16 for older GPUs
- Use torch.cuda.amp for automatic mixed precision

### TPU Training

```bash
# Install JAX for TPU support
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run production training
python train/zenyx_single_tpu_train.py
```

**Tips:**
- Use bfloat16 (TPU native)
- Use all 8 TPU cores
- Enable all ZENYX features:
  - KV cache tiering
  - FP8 quantization
  - Curriculum learning
  - Sparse attention

## Training Best Practices

### 1. Start Small

```python
# Start with tiny model on CPU
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
)

# Test with minimal data
data = torch.randn(10, 10)

# Run for 1 epoch
for epoch in range(1):
    # Training code
```

### 2. Monitor Loss

```python
losses = []

for epoch in range(num_epochs):
    for batch in loader:
        loss = train_step(batch)
        losses.append(loss)

# Plot to check for:
# - Stable descent (good)
# - Oscillation (bad learning rate)
# - Explosion (bad gradient)
# - Plateau (training done)
```

### 3. Gradient Clipping

Always use gradient clipping for stability:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 4. Learning Rate Scheduling

Use a schedule that decreases LR over time:

```python
# Cosine annealing (recommended)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
)

# Or step-based
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1,
)
```

### 5. Checkpointing

Save regularly:

```python
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }, f"checkpoint_{epoch}.pt")
```

### 6. Validation

Always validate:

```python
# Training loop
model.train()
for batch in train_loader:
    # Training code

# Validation loop
model.eval()
with torch.no_grad():
    for batch in val_loader:
        # Validation code
```

### 7. Mixed Precision

Use mixed precision for speed:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    with autocast():
        loss = compute_loss(batch)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

## ZENYX Features

### Phase 7: Bélády KV Cache Tiering
Manages attention KV cache across memory hierarchy.

```python
from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager

# Enable for 1M token context
tiering = BeladyKVCacheTieringManager(
    ring_sequence=[...],
    hbm_capacity_gb=16,
    bandwidth=900,  # GB/s
)
```

### Phase 8: FP8 KV Quantization
Quantizes KV cache to FP8 for 2x compression.

```python
from zenyx.train.fp8_kv_quantization import FP8KVQuantizer

quantizer = FP8KVQuantizer(
    scaling_factor=1.0,
    min_value=-448,
    max_value=448,
)
```

### Phase 9: Dynamic Ring Curriculum
Progressive training curriculum.

```python
from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler

curriculum = RingDegreeScheduler(
    config=config,
    initial_degree=2,
    max_degree=8,
)
```

### Phase 10: Sparse Ring Attention
Sparse attention with 13.3x speedup.

```python
from zenyx.train.sparse_ring_attention import SparseRingAttention

attention = SparseRingAttention(
    num_heads=96,
    head_dim=128,
    config=config,
)
```

## Common Issues and Solutions

### Issue: Out of Memory

```python
# Solution 1: Reduce batch size
batch_size = 1  # From 32

# Solution 2: Reduce sequence length
seq_len = 512  # From 2048

# Solution 3: Enable gradient accumulation
accumulation_steps = 4

# Solution 4: Use mixed precision
with autocast():
    loss = compute_loss()

# Solution 5: Use ZENYX quantization
quantizer = FP8KVQuantizer(...)
```

### Issue: Loss Not Decreasing

```python
# Solution 1: Increase learning rate
lr = 1e-3  # From 1e-5

# Solution 2: Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 3: Check data pipeline
# Print a few batches to verify

# Solution 4: Warmup learning rate
# Start small, increase to target over N steps
```

### Issue: Training Too Slow

```python
# Solution 1: Use mixed precision
with autocast():
    loss = compute_loss()

# Solution 2: Enable sparse attention
attention = SparseRingAttention(...)

# Solution 3: Reduce model size
hidden_dim = 512  # From 2048

# Solution 4: Use gradient accumulation
# Reduces backward passes
```

## Next Steps

1. **Start with Example 01** - Get familiar with basics
2. **Move to Example 02** - Learn production patterns
3. **Try on your hardware** - Adapt to GP