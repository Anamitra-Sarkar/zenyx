# ZENYX Installation & Setup Guide

## Quick Install

```bash
# Install ZENYX with PyTorch
pip install zenyx torch

# Verify
python test/validate_zenyx_four_pillars.py
```

---

## Full Installation Guide

### Prerequisites

- Python 3.8+
- pip or conda
- For GPU: NVIDIA CUDA 11.8+
- For TPU: Google Cloud TPU access

### Step 1: Install via pip

```bash
# CPU-only
pip install zenyx torch

# GPU (NVIDIA CUDA)
pip install zenyx torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# From source
git clone https://github.com/zenyx/zenyx.git && cd zenyx && pip install -e .
```

### Step 2: Install via conda

```bash
conda create -n zenyx python=3.10
conda activate zenyx
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install zenyx
```

### Step 3: Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check ZENYX
python -c "import zenyx; print('ZENYX ✓')"

# Full validation
python test/validate_zenyx_four_pillars.py
```

---

## Core Imports

### Minimal (Basic Training)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
```

### Standard (Production)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import json
from pathlib import Path
from datetime import datetime
```

### Full (With ZENYX Features)

```python
# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Utilities
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

---

## Hardware Setup

### CPU

```python
import torch
device = torch.device("cpu")
print(f"Using CPU with {torch.get_num_threads()} threads")
```

### GPU (NVIDIA)

```python
import torch

# Check GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
```

### TPU (Google Cloud)

```bash
pip install jax[tpu] jaxlib
python -c "import jax; print(f'TPU devices: {jax.devices()}')"
```

---

## Environment Variables

```bash
# Optimal performance settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Persistent: add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
export OMP_NUM_THREADS=8
EOF
source ~/.bashrc
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'zenyx'` | `pip install zenyx --force-reinstall` |
| GPU not detected | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| CUDA out of memory | Reduce batch size, use mixed precision, enable gradient checkpointing |
| Import errors | `python test/validate_zenyx_four_pillars.py` |

---

## Next Steps

1. **Verify:** `python test/validate_zenyx_four_pillars.py`
2. **Train:** `python examples/01_beginner_cpu_training.py`
3. **Learn:** Read `TRAINING_GUIDE_COMPLETE.md`
4. **Optimize:** See `TRAINING_BEST_PRACTICES.md`

Happy training! 🎉
