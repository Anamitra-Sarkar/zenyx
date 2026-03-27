# ZENYX Training Quick Reference

## Installation & Setup

**Full guide:** See `INSTALLATION_AND_SETUP.md`

### Quick Install
```bash
pip install zenyx torch
python test/validate_zenyx_four_pillars.py
```

### Core Imports (Copy & Paste)
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# ZENYX features (optional)
from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager
from zenyx.train.fp8_kv_quantization import FP8KVQuantizer
from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler
from zenyx.train.sparse_ring_attention import SparseRingAttention
```

### Check Your Setup
```python
import torch

# CPU/GPU info
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Running Training Scripts

### Option 1: Minimal (Testing, <1 min)
```bash
python train_minimal.py
```
- 108-parameter model
- CPU training
- Perfect for testing setup

### Option 2: Beginner (Learning, 2-5 min)
```bash
python train_with_loss.py
```
- 2K vocab language model
- Full training pipeline
- Loss tracking & metrics

### Option 3: Complete Demo (Advanced, 5-10 min)
```bash
python train_complete_demo.py
```
- Transformer model
- Train + validation
- Learning rate scheduling
- Checkpoint saving

### Option 4: Production TPU (Scale, hours)
```bash
python train/zenyx_single_tpu_train.py
```
- 1T parameter model
- TPU v5e-8 optimized
- All ZENYX features
- Distributed training ready

### Option 5: Interactive Examples
```bash
python examples/01_beginner_cpu_training.py      # 1 min
python examples/02_intermediate_finetuning.py    # 5 min
python examples/03_expert_tpu_v5e8_training.py   # hours
```

## Key Code Patterns

### Creating a Model
```python
model = nn.Sequential(
    nn.Embedding(vocab_size, hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, vocab_size),
)
```

### Creating a DataLoader
```python
from torch.utils.data import DataLoader, TensorDataset

input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
target_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

dataset = TensorDataset(input_ids, target_ids)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### Training Loop
```python
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    # Training loop...
    scheduler.step()
```

### Checkpointing
```python
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

### Validation Loop
```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch)
        loss = loss_fn(outputs, targets)
```

## ZENYX Features

### Phase 7: KV Cache Tiering
```python
from zenyx.train.belayd_kv_cache_tiering import BeladyKVCacheTieringManager

tiering = BeladyKVCacheTieringManager(
    ring_sequence=[...],
    hbm_capacity_gb=16,
)
```

### Phase 8: FP8 Quantization
```python
from zenyx.train.fp8_kv_quantization import FP8KVQuantizer

quantizer = FP8KVQuantizer(
    scaling_factor=1.0,
    min_value=-448,
    max_value=448,
)
```

### Phase 9: Curriculum Learning
```python
from zenyx.train.dynamic_ring_curriculum import RingDegreeScheduler

curriculum = RingDegreeScheduler(
    config=config,
    initial_degree=2,
    max_degree=8,
)
```

### Phase 10: Sparse Attention
```python
from zenyx.train.sparse_ring_attention import SparseRingAttention

attention = SparseRingAttention(
    num_heads=96,
    head_dim=128,
)
```

## Hardware Requirements

### CPU
- 8+ GB RAM
- 2+ cores
- Training time: Minutes-hours

### GPU (A100)
- 80 GB VRAM for 70B model
- 8+ cores
- bfloat16 support
- Training time: Hours-days

### TPU v5e-8
- 16 GB HBM per core (128 GB total)
- 8 cores
- bfloat16 native
- Training time: Hours-weeks

## Optimization Tips

### 1. Memory Usage
- Reduce batch size
- Reduce sequence length
- Enable gradient accumulation
- Use mixed precision
- Enable quantization

### 2. Training Speed
- Use mixed precision
- Enable sparse attention
- Use gradient accumulation
- Batch process data
- Use multiple GPUs/TPUs

### 3. Convergence
- Use learning rate warmup
- Use learning rate scheduling
- Use gradient clipping
- Use batch normalization
- Use curriculum learning

## Common Commands

```bash
# Run minimal example
python train_minimal.py

# Run with loss tracking
python train_with_loss.py

# Run full demo
python train_complete_demo.py

# Run TPU training
python train/zenyx_single_tpu_train.py

# Run beginner example
python examples/01_beginner_cpu_training.py

# Run intermediate example
python examples/02_intermediate_finetuning.py

# Run expert example
python examples/03_expert_tpu_v5e8_training.py

# Validate installation
python test/validate_zenyx_four_pillars.py

# Test all features
python test/comprehensive_e2e_validation.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce batch size, sequence length, use quantization |
| Loss not decreasing | Increase learning rate, check data, use warmup |
| NaN loss | Check gradients, clip gradients, reduce learning rate |
| Too slow | Use mixed precision, enable sparse attention, use GPU/TPU |
| Model not converging | Use learning rate scheduling, curriculum learning |

## Next Steps

1. Start with `examples/01_beginner_cpu_training.py`
2. Read `TRAINING_GUIDE_COMPLETE.md`
3. Review best practices in `TRAINING_BEST_PRACTICES.md`
4. Try on your hardware (CPU/GPU/TPU)
5. Integrate ZENYX features
6. Scale up model and data size
7. Deploy with torchscript or ONNX
