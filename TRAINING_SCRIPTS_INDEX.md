# ZENYX Training Scripts - Complete Index

## Training Scripts Overview

All training scripts in the ZENYX library are designed to help you train models on your hardware (CPU, GPU, or TPU).

## Main Training Scripts

### 1. `train_minimal.py`
**Purpose:** Quickest way to test ZENYX installation

- Model: 108 parameters
- Hardware: CPU
- Time: <1 minute
- Data: Synthetic
- Features: Basic training loop

**Run:**
```bash
python train_minimal.py
```

**Output:**
- Training loss trajectory
- Parameter count
- Training completion status

---

### 2. `train_with_loss.py`
**Purpose:** Learn complete training pipeline

- Model: Transformer with embeddings (52K parameters)
- Hardware: CPU
- Time: 2-5 minutes
- Data: Synthetic language modeling
- Features:
  - Gradient clipping
  - Cross-entropy loss
  - Multiple epochs
  - Loss tracking

**Run:**
```bash
python train_with_loss.py
```

**Output:**
- Per-epoch average loss
- Training completion status
- Model parameter count

---

### 3. `train_complete_demo.py`
**Purpose:** Production-ready training pipeline

- Model: Feed-forward with residuals (1.5M parameters)
- Hardware: CPU
- Time: 5-10 minutes
- Data: Synthetic language modeling
- Features:
  - Train + validation loops
  - Learning rate scheduling (CosineAnnealing)
  - Checkpoint saving
  - Metrics JSON export
  - Gradient accumulation

**Run:**
```bash
python train_complete_demo.py
```

**Output:**
- Per-epoch train and validation loss
- Learning rate adjustments
- Checkpoint files in `./checkpoints`
- Metrics in `training_metrics_demo.json`

---

### 4. `train_demo.py`
**Purpose:** Reference implementation for custom training

- Model: Customizable transformer
- Hardware: CPU
- Time: 5-15 minutes
- Data: Synthetic
- Features: Flexible architecture, easy to modify

---

### 5. `train/zenyx_single_tpu_train.py`
**Purpose:** Production TPU training with all ZENYX features

- Model: 1 trillion parameters (configurable)
- Hardware: TPU v5e-8 (or CPU for testing)
- Time: Hours to weeks depending on scale
- Data: Synthetic (use real data in production)
- Features:
  - Large-scale model definition
  - All 4 ZENYX pillars integrated
  - Distributed training support
  - Advanced checkpointing
  - Production logging
  - Error handling and recovery

**Run:**
```bash
python train/zenyx_single_tpu_train.py
```

**Output:**
- Training progress logs
- Checkpoint files
- Metrics JSON
- Final model state

---

## Interactive Examples

### `examples/01_beginner_cpu_training.py`
**Target:** Complete beginners

- Explains each step
- Runs quickly (<1 min)
- Good for learning flow

```bash
python examples/01_beginner_cpu_training.py
```

---

### `examples/02_intermediate_finetuning.py`
**Target:** Intermediate users

- Gradient accumulation
- Learning rate warmup
- Proper checkpoint saving
- Metrics tracking

```bash
python examples/02_intermediate_finetuning.py
```

---

### `examples/03_expert_tpu_v5e8_training.py`
**Target:** Advanced users

- Production-scale model
- All ZENYX features
- Distributed training patterns
- Enterprise logging

```bash
python examples/03_expert_tpu_v5e8_training.py
```

---

## Documentation Files

### `TRAINING_GUIDE_COMPLETE.md` ⭐ START HERE
Complete guide covering:
- Quick start for all skill levels
- Step-by-step examples
- Training on different hardware (CPU/GPU/TPU)
- Best practices
- ZENYX features explanation
- Troubleshooting guide

**Read:** First! This is the master guide.

---

### `TRAINING_QUICK_REFERENCE.md`
Cheat sheet with:
- Running all scripts
- Key code patterns
- ZENYX features quick examples
- Common commands
- Troubleshooting table

**Use:** For quick lookups and copy-paste patterns.

---

### `TRAINING_BEST_PRACTICES.md`
Deep dive into:
- Learning rate scheduling strategies
- Gradient clipping and normalization
- Checkpointing and recovery
- Distributed training setup
- Mixed precision training
- Memory optimization
- Debugging strategies

**Read:** When optimizing training.

---

## Quick Start Guide

### For Beginners
```bash
# 1. Minimal test (validates installation)
python train_minimal.py

# 2. See training in action
python examples/01_beginner_cpu_training.py

# 3. Read complete guide
cat TRAINING_GUIDE_COMPLETE.md
```

### For Intermediate Users
```bash
# 1. Run complete demo
python train_complete_demo.py

# 2. Study intermediate example
python examples/02_intermediate_finetuning.py

# 3. Check best practices
cat TRAINING_BEST_PRACTICES.md
```

### For Expert Users
```bash
# 1. Study production code
python examples/03_expert_tpu_v5e8_training.py

# 2. Run on TPU
python train/zenyx_single_tpu_train.py

# 3. Customize for your needs
# Edit train/zenyx_single_tpu_train.py
```

---

## Training Workflow

### Step 1: Understand the Basics
```
📖 Read TRAINING_GUIDE_COMPLETE.md section 1-2
🏃 Run examples/01_beginner_cpu_training.py
✅ Understand the training loop
```

### Step 2: Learn Production Patterns
```
📖 Read TRAINING_GUIDE_COMPLETE.md section 3-4
🏃 Run train_complete_demo.py
✅ Understand checkpointing and metrics
```

### Step 3: Add ZENYX Features
```
📖 Read TRAINING_GUIDE_COMPLETE.md section 5 (ZENYX Features)
🏃 Review examples/03_expert_tpu_v5e8_training.py
✅ Understand all four pillars
```

### Step 4: Scale to Production
```
📖 Read TRAINING_BEST_PRACTICES.md
🏃 Run train/zenyx_single_tpu_train.py
✅ Deploy with confidence
```

---

## Feature Comparison

| Feature | train_minimal | train_with_loss | train_complete_demo | zenyx_single_tpu |
|---------|---|---|---|---|
| Gradient clipping | ❌ | ✅ | ✅ | ✅ |
| LR scheduling | ❌ | ❌ | ✅ | ✅ |
| Checkpointing | ❌ | ❌ | ✅ | ✅ |
| Validation loop | ❌ | ❌ | ✅ | ❌ |
| Metrics export | ❌ | ❌ | ✅ | ✅ |
| Multi-GPU support | ❌ | ❌ | ❌ | ✅ |
| TPU support | ❌ | ❌ | ❌ | ✅ |
| ZENYX integration | ❌ | ❌ | ❌ | ✅ |
| Model size | 108 | 52K | 1.5M | 1T |
| Training time | <1 min | 2-5 min | 5-10 min | Hours+ |

---

## Hardware Support

### CPU
**Scripts:** `train_minimal.py`, `train_with_loss.py`, `train_complete_demo.py`, `examples/*`

```bash
# All scripts work on CPU
python train_with_loss.py
```

### GPU (NVIDIA A100)
**Scripts:** `train_complete_demo.py`, `train/zenyx_single_tpu_train.py`

```bash
# Install GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run with GPU
python train_complete_demo.py  # Will auto-detect GPU
```

### TPU v5e-8
**Scripts:** `train/zenyx_single_tpu_train.py`, `examples/03_expert_tpu_v5e8_training.py`

```bash
# Install JAX for TPU
pip install jax[tpu]

# Run on TPU
python train/zenyx_single_tpu_train.py
```

---

## Customization Guide

### Change Model Size
```python
# In train_complete_demo.py
config = {
    "hidden_dim": 512,      # Change this
    "num_layers": 6,        # Change this
    # ...
}
```

### Change Dataset
```python
# In train_complete_demo.py
# Replace create_data_loaders() function with your data
def create_data_loaders(config):
    # Load your data
    # Return train_loader, val_loader
```

### Change Optimizer
```python
# In train_complete_demo.py
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# or
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
```

### Change Learning Rate Schedule
```python
# In train_complete_demo.py
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# or
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

---

## Testing and Validation

### Validate ZENYX Installation
```bash
python test/validate_zenyx_four_pillars.py
```

### End-to-End Testing
```bash
python test/comprehensive_e2e_validation.py
```

### Test Specific Components
```bash
# Test KV cache tiering
python -c "from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager; print('✓ Phase 7 OK')"

# Test FP8 quantization
python -c "from zenyx.train.fp8_kv_quantization import FP8KVQuantizer; print('✓ Phase 8 OK')"

# Test curriculum learning
python -c "from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler; print('✓ Phase 9 OK')"

# Test sparse attention
python -c "from zenyx.train.sparse_ring_attention import SparseRingAttention; print('✓ Phase 10 OK')"
```

---

## Troubleshooting

### Script Not Running?
```bash
# Check Python version
python --version  # Need 3.8+

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check ZENYX installation
python -c "import zenyx; print('OK')"
```

### Out of Memory?
Edit script and:
```python
config = {
    "batch_size": 2,        # Reduce from 8
    "seq_len": 256,         # Reduce from 512
    "hidden_dim": 256,      # Reduce from 512
}
```

### Loss Not Decreasing?
```python
# Increase learning rate
config["learning_rate"] = 1e-3  # From 1e-4

# Add warmup
# Check your data pipeline
# Verify model is trainable
```

### Training Too Slow?
```python
# Use GPU/TPU instead of CPU
# Enable mixed precision
# Reduce model size
# Reduce dataset size
```

---

## Next Steps

1. **Start:** Run `python examples/01_beginner_cpu_training.py`
2. **Learn:** Read `TRAINING_GUIDE_COMPLETE.md`
3. **Practice:** Modify `train_complete_demo.py` for your task
4. **Scale:** Move to `train/zenyx_single_tpu_train.py`
5. **Deploy:** Use checkpoints for inference

---

## Support

- **Full Guide:** `TRAINING_GUIDE_COMPLETE.md`
- **Quick Ref:** `TRAINING_QUICK_REFERENCE.md`
- **Best Practices:** `TRAINING_BEST_PRACTICES.md`
- **Examples:** `examples/*.py`
- **Tests:** `test/*.py`
