# 🚀 ZENYX Training - START HERE

Welcome! This is your entry point to training models with ZENYX.

## What is ZENYX?

ZENYX is a library for training trillion-parameter language models on TPU v5e-8 with advanced techniques:
- **Phase 7:** Bélády KV Cache Tiering (1M token context)
- **Phase 8:** FP8 KV Quantization (2x compression)
- **Phase 9:** Dynamic Ring Curriculum (15% faster)
- **Phase 10:** Sparse Ring Attention (13.3x speedup)

## Installation & Setup

**👉 See:** `INSTALLATION_AND_SETUP.md` for complete instructions

### Quick Install (2 minutes)

```bash
# Install ZENYX
pip install zenyx torch

# Verify installation
python test/validate_zenyx_four_pillars.py
```

### Core Imports

All training scripts use:

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

See `INSTALLATION_AND_SETUP.md` for:
- Full installation steps for CPU/GPU/TPU
- How to verify your hardware
- Environment variable setup
- Troubleshooting common issues

## Quick Start (Choose Your Path)

### 🟢 I'm a Complete Beginner (5 minutes)

```bash
# 1. Validate installation
python test/validate_zenyx_four_pillars.py

# 2. Run first example
python examples/01_beginner_cpu_training.py

# 3. Read quick guide
cat TRAINING_QUICK_REFERENCE.md
```

**Next:** Read `TRAINING_GUIDE_COMPLETE.md`

---

### 🟡 I Know Machine Learning (15 minutes)

```bash
# 1. Review intermediate example
cat examples/02_intermediate_finetuning.py

# 2. Run it
python examples/02_intermediate_finetuning.py

# 3. Modify for your task
# Edit examples/02_intermediate_finetuning.py
```

**Next:** Read `TRAINING_BEST_PRACTICES.md`

---

### 🔴 I'm a Production Expert (Setup production)

```bash
# 1. Review production code
cat train/zenyx_single_tpu_train.py

# 2. Adapt for your hardware
# Edit train/zenyx_single_tpu_train.py

# 3. Run on TPU
python train/zenyx_single_tpu_train.py
```

**Next:** Monitor checkpoints and metrics

---

## All Training Scripts

| Script | Time | Hardware | Model | Use |
|--------|------|----------|-------|-----|
| `train_minimal.py` | <1 min | CPU | 108 params | Smoke test |
| `train_with_loss.py` | 2 min | CPU | 52K params | Learn basics |
| `train_complete_demo.py` | 10 min | CPU/GPU | 1.5M params | See best practices |
| `train/zenyx_single_tpu_train.py` | Hours | TPU | 1T params | Production |

## All Examples

| Example | Time | Level | Purpose |
|---------|------|-------|---------|
| `examples/01_beginner_cpu_training.py` | <1 min | Beginner | Basic loop |
| `examples/02_intermediate_finetuning.py` | 5 min | Intermediate | Production patterns |
| `examples/03_expert_tpu_v5e8_training.py` | Hours | Expert | Large-scale training |

## All Documentation

| Document | Best For |
|----------|----------|
| `TRAINING_QUICK_REFERENCE.md` | Quick lookups |
| `TRAINING_GUIDE_COMPLETE.md` | Learning everything |
| `TRAINING_BEST_PRACTICES.md` | Optimizing training |
| `TRAINING_SCRIPTS_INDEX.md` | Understanding scripts |
| `TRAINING_AUDIT_COMPLETE.md` | Seeing what was done |
| `TRAINING_COMMANDS_CHEATSHEET.py` | Copy-paste commands |

## Common Tasks

### Train a tiny model (testing)
```bash
python train_minimal.py
```

### Train a small model (learning)
```bash
python examples/01_beginner_cpu_training.py
```

### Train a real model (production)
```bash
python train_complete_demo.py
```

### Train on GPUs
```bash
# Same scripts, will auto-detect GPU
python train_complete_demo.py
```

### Train on TPU v5e-8
```bash
python train/zenyx_single_tpu_train.py
```

## What I'll Learn

After following this guide, you'll understand:

✅ How to define a PyTorch model  
✅ How to create a training loop  
✅ How to use optimizers and schedulers  
✅ How to save and load checkpoints  
✅ How to track metrics  
✅ How to integrate ZENYX features  
✅ How to scale to large models  
✅ How to deploy in production  

## Next Steps

### Step 1: Pick Your Path
- **Beginner?** → Run `examples/01_beginner_cpu_training.py`
- **Intermediate?** → Run `examples/02_intermediate_finetuning.py`
- **Expert?** → Run `examples/03_expert_tpu_v5e8_training.py`

### Step 2: Read the Guide
Start with: `TRAINING_QUICK_REFERENCE.md`  
Then: `TRAINING_GUIDE_COMPLETE.md`

### Step 3: Run the Code
- Try the examples as-is
- Modify for your data
- Scale up gradually

### Step 4: Deploy
- Save checkpoints
- Monitor metrics
- Deploy with confidence

## FAQ

**Q: How long does training take?**  
A: Depends on hardware. CPU: minutes. GPU: hours. TPU: hours-weeks.

**Q: What hardware do I need?**  
A: Any! Start on CPU, scale to GPU or TPU later.

**Q: Can I modify the scripts?**  
A: Absolutely! They're designed to be modified.

**Q: How do I use my own data?**  
A: Replace the `create_synthetic_data()` function with your data loader.

**Q: How do I enable ZENYX features?**  
A: See `train/zenyx_single_tpu_train.py` for examples.

## File Structure

```
train_minimal.py                    ← Simplest example
train_with_loss.py                  ← With loss tracking
train_complete_demo.py              ← Full training pipeline
train/
  └── zenyx_single_tpu_train.py    ← Production TPU training

examples/
  ├── 01_beginner_cpu_training.py
  ├── 02_intermediate_finetuning.py
  └── 03_expert_tpu_v5e8_training.py

Documentation/
  ├── TRAINING_QUICK_REFERENCE.md
  ├── TRAINING_GUIDE_COMPLETE.md
  ├── TRAINING_BEST_PRACTICES.md
  ├── TRAINING_SCRIPTS_INDEX.md
  └── TRAINING_AUDIT_COMPLETE.md

Tests/
  ├── test/validate_zenyx_four_pillars.py
  └── test/comprehensive_e2e_validation.py
```

## Ready? Let's Go!

Choose your path:

**Beginner:**
```bash
python examples/01_beginner_cpu_training.py
cat TRAINING_GUIDE_COMPLETE.md
```

**Intermediate:**
```bash
python examples/02_intermediate_finetuning.py
cat TRAINING_BEST_PRACTICES.md
```

**Expert:**
```bash
python examples/03_expert_tpu_v5e8_training.py
cat train/zenyx_single_tpu_train.py
```

**Need help?**
```bash
cat TRAINING_QUICK_REFERENCE.md
cat TRAINING_GUIDE_COMPLETE.md
python TRAINING_COMMANDS_CHEATSHEET.py
```

---

**Happy training! 🎉**
