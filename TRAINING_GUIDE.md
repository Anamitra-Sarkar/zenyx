# Training Models with Zenyx

## Quick Start

Train a model on CPU using Zenyx's simple API:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenyx.train.trainer import Trainer

# 1. Define your model
model = nn.Sequential(
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
)

# 2. Prepare data
data = torch.randn(128, 8, dtype=torch.bfloat16)
targets = torch.randn(128, 4, dtype=torch.bfloat16)
loader = DataLoader(TensorDataset(data, targets), batch_size=8)

# 3. Wrap model to return loss
class TrainableModel(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
    
    def forward(self, x, y=None):
        out = self.model(x)
        if y is not None:
            return self.loss_fn(out, y)
        return out

trainable_model = TrainableModel(model, nn.MSELoss())

# 4. Train with Zenyx
trainer = Trainer(
    trainable_model,
    loader,
    lr=1e-3,
    total_steps=100,
)
trainer.train()
```

## What Zenyx Does Automatically

✅ **Hardware Detection**: CPU, GPU (CUDA), AMD (ROCm), TPU  
✅ **Memory Management**: 3-tier hierarchy (HBM/VRAM → DRAM → NVMe) with Bélády-optimal eviction  
✅ **Parallelism Planning**: Auto-determines TP/PP/DP degrees based on model size and hardware  
✅ **Attention Kernel**: Selects FlashAttention-3 (H100), Ring-Pallas (TPU), or chunked attention (CPU)  
✅ **Optimizer Setup**: Adam with mixed precision, gradient scaling, and clipping  
✅ **Checkpointing**: Activation checkpointing + distributed checkpoint saving  

## Running the Training Demos

Three example scripts are provided:

### 1. Basic Training (`train_demo.py`)
Simple training loop with metrics output.
```bash
python train_demo.py
```

### 2. Loss Tracking (`train_with_loss.py`)
Training with final loss and throughput metrics.
```bash
python train_with_loss.py
```

### 3. Complete Demo (`train_complete_demo.py`)
Comprehensive training pipeline with all metrics and hardware info.
```bash
python train_complete_demo.py
```

## Key Trainer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate |
| `total_steps` | 100,000 | Total training steps |
| `warmup_steps` | 2,000 | Learning rate warmup |
| `dtype` | bfloat16 | Model dtype (bfloat16, float16, float32) |
| `context_len` | 4,096 | Maximum context length |
| `gradient_accumulation_steps` | 1 | Gradient accumulation for larger effective batches |
| `checkpoint_every` | 1,000 | Save checkpoint frequency |
| `weight_decay` | 0.1 | L2 regularization |
| `grad_clip` | 1.0 | Gradient clipping norm |

## Advanced Features

### FP8 KV Cache Quantization
```python
trainer = Trainer(
    model, loader,
    fp8_kv=True,
    fp8_quant_strategy="per_channel",
)
```

### Dynamic Ring Degree (Curriculum Learning)
```python
trainer = Trainer(
    model, loader,
    context_len=1048576,  # 1M tokens
)
```

### Sparse Ring Attention
```python
trainer = Trainer(
    model, loader,
    sparse_attn=True,
    sparse_skip_mode="production",
)
```

## Troubleshooting

**Issue**: "Global batch X tokens is dangerously small"  
**Solution**: Increase batch size or use `gradient_accumulation_steps`

**Issue**: Model dtype mismatch error  
**Solution**: Ensure model weights match the Trainer's dtype (default: bfloat16)

**Issue**: Low throughput on CPU  
**Expected**: CPU training is slower (suitable for development/CI). Use GPU for production.

## References

- **Paper**: Ring Attention (https://arxiv.org/abs/2310.01889)
- **Kernel**: FlashAttention-3 (https://arxiv.org/abs/2407.08691)
- **Memory**: Bélády's algorithm for cache replacement
- **Parallelism**: Megatron-LM + DynaPipe
