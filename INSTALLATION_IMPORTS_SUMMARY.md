# ZENYX Installation & Imports - Complete Summary

## Quick Reference

### Installation (2 commands)
```bash
pip install zenyx torch
python test/validate_zenyx_four_pillars.py
```

### Imports (Copy-Paste)
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

---

## Documentation Files

### 1. **INSTALLATION_AND_SETUP.md** ⭐ Primary Reference
Complete installation guide including:
- Installation options (pip, conda, source)
- Verification steps
- Full import examples for all use cases
- Hardware setup (CPU/GPU/TPU)
- Environment variables
- Troubleshooting guide

**When to use:** First-time setup, hardware issues, import errors

### 2. **TRAINING_START_HERE.md** ⭐ Entry Point
Quick setup and path selection:
- Quick install (2 min)
- Core imports overview
- Links to INSTALLATION_AND_SETUP.md
- Choose your learning path

**When to use:** First time, getting started, choosing difficulty level

### 3. **TRAINING_QUICK_REFERENCE.md** ⭐ Cheat Sheet
Quick lookups with copy-paste code:
- Installation summary
- Core imports (ready to copy)
- Setup verification code
- Quick training commands

**When to use:** Need quick copy-paste code, forgotten syntax

### 4. **TRAINING_BEST_PRACTICES.md** ⭐ Advanced
In-depth optimization guide:
- Installation & imports section
- Hardware-specific configurations
- Optimization techniques
- Performance tuning
- Debugging & deployment

**When to use:** Optimizing training, production deployment, debugging

### 5. **TRAINING_GUIDE_COMPLETE.md** 
Comprehensive learning guide:
- Will include installation section (in progress)
- Detailed explanations
- Complete examples with comments
- Hardware-specific guidance

**When to use:** Learning from scratch, detailed understanding

---

## Installation Paths

### Path 1: Quick Start (5 minutes)
1. **Install:** `pip install zenyx torch`
2. **Verify:** `python test/validate_zenyx_four_pillars.py`
3. **Copy imports:** See TRAINING_QUICK_REFERENCE.md
4. **Run example:** `python examples/01_beginner_cpu_training.py`

### Path 2: Complete Setup (30 minutes)
1. **Read:** INSTALLATION_AND_SETUP.md (full)
2. **Install:** Choose your platform (CPU/GPU/TPU)
3. **Verify:** Run validation tests
4. **Learn:** TRAINING_GUIDE_COMPLETE.md
5. **Run:** examples/02_intermediate_finetuning.py

### Path 3: Production Deployment (1 hour)
1. **Read:** INSTALLATION_AND_SETUP.md (full)
2. **Setup:** Hardware configuration for GPU/TPU
3. **Learn:** TRAINING_BEST_PRACTICES.md
4. **Deploy:** train/zenyx_single_tpu_train.py
5. **Monitor:** Checkpoints and metrics

---

## Key Information by Topic

### Installation Methods

| Method | Command | Best For |
|--------|---------|----------|
| pip (CPU) | `pip install zenyx torch` | Quick start on CPU |
| pip (GPU) | `pip install zenyx torch --index-url https://download.pytorch.org/whl/cu118` | GPU training |
| conda | `conda install pytorch::pytorch` + `pip install zenyx` | Clean environment |
| source | `git clone ... && pip install -e .` | Development |

### Core Imports by Use Case

**Minimal (Just PyTorch)**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
```

**Standard (Production)**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
```

**Full (With ZENYX)**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager
from zenyx.train.fp8_kv_quantization import FP8KVQuantizer
from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler
from zenyx.train.sparse_ring_attention import SparseRingAttention
```

### Hardware Verification

**Check what you have:**
```python
import torch

# Basic
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")

# Detailed
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Set device:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

---

## Common Issues & Solutions

| Issue | Solution | Reference |
|-------|----------|-----------|
| `ModuleNotFoundError: zenyx` | `pip install zenyx --force-reinstall` | INSTALLATION_AND_SETUP.md |
| GPU not detected | `pip install torch --index-url https://download.pytorch.org/whl/cu118` | INSTALLATION_AND_SETUP.md |
| CUDA out of memory | Reduce batch size, use mixed precision | TRAINING_BEST_PRACTICES.md |
| Import errors | `python test/validate_zenyx_four_pillars.py` | INSTALLATION_AND_SETUP.md |
| Slow training | Enable mixed precision, use GPU/TPU | TRAINING_BEST_PRACTICES.md |

---

## Next Steps

1. **Install:**
   ```bash
   pip install zenyx torch
   python test/validate_zenyx_four_pillars.py
   ```

2. **Choose your path:**
   - **Beginner:** Read TRAINING_START_HERE.md
   - **Intermediate:** Read TRAINING_QUICK_REFERENCE.md
   - **Advanced:** Read INSTALLATION_AND_SETUP.md + TRAINING_BEST_PRACTICES.md

3. **Run first example:**
   ```bash
   python examples/01_beginner_cpu_training.py
   ```

4. **Learn & Train:**
   - TRAINING_GUIDE_COMPLETE.md for detailed learning
   - TRAINING_BEST_PRACTICES.md for optimization
   - train/zenyx_single_tpu_train.py for production

---

## Documentation Hierarchy

```
START HERE
  ├─ TRAINING_START_HERE.md (choose path)
  │
  ├─ QUICK PATH (5 min)
  │  ├─ TRAINING_QUICK_REFERENCE.md
  │  └─ examples/01_beginner_cpu_training.py
  │
  ├─ COMPLETE PATH (30 min)
  │  ├─ INSTALLATION_AND_SETUP.md (full)
  │  ├─ TRAINING_GUIDE_COMPLETE.md
  │  └─ examples/02_intermediate_finetuning.py
  │
  └─ PRODUCTION PATH (1 hour)
     ├─ INSTALLATION_AND_SETUP.md (full)
     ├─ TRAINING_BEST_PRACTICES.md
     └─ train/zenyx_single_tpu_train.py
```

---

**Happy training! 🎉**

For installation help: See **INSTALLATION_AND_SETUP.md**  
For quick start: See **TRAINING_QUICK_REFERENCE.md**  
For learning: See **TRAINING_GUIDE_COMPLETE.md**  
For optimization: See **TRAINING_BEST_PRACTICES.md**
