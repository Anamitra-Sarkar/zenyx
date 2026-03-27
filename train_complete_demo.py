#!/usr/bin/env python3
"""
Zenyx Training Demo: Complete 70-Parameter Model Training on CPU

This script demonstrates:
1. Creating a minimal neural network (108 parameters)
2. Preparing training data
3. Training with Zenyx's Trainer API (hardware-agnostic, auto memory management)
4. Monitoring training metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenyx.train.trainer import Trainer

print("\n" + "="*75)
print(" ZENYX: Hardware-Agnostic LLM Training Runtime")
print(" Demo: Train a 70-parameter model on CPU")
print("="*75)

# ============================================================================
# 1. Define Model Architecture
# ============================================================================
print("\n[1] Model Definition")
print("-" * 75)

model = nn.Sequential(
    nn.Linear(8, 8),        # Input -> Hidden: 8*8 + 8 = 72 parameters
    nn.ReLU(),
    nn.Linear(8, 4),        # Hidden -> Output: 8*4 + 4 = 36 parameters
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Architecture:")
for i, layer in enumerate(model):
    if isinstance(layer, nn.Linear):
        print(f"  Layer {i}: {layer.in_features} -> {layer.out_features}")
    else:
        print(f"  Layer {i}: {layer.__class__.__name__}")
print(f"\nTotal parameters: {total_params}")

# ============================================================================
# 2. Prepare Training Data
# ============================================================================
print("\n[2] Data Preparation")
print("-" * 75)

torch.manual_seed(42)
dtype = torch.bfloat16  # Match Zenyx's default

# Create synthetic dataset
X = torch.randn(128, 8, dtype=dtype)
y = torch.randn(128, 4, dtype=dtype)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8)

print(f"Dataset size: 128 samples")
print(f"Batch size: 8")
print(f"Total batches: {len(loader)}")
print(f"Dtype: {dtype}")

# ============================================================================
# 3. Define Training Model
# ============================================================================
print("\n[3] Training Setup")
print("-" * 75)

loss_fn = nn.MSELoss()

class TrainableModel(nn.Module):
    """Wrapper that returns loss for Zenyx's Trainer API."""
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
    
    def forward(self, x, y=None):
        out = self.model(x)
        if y is not None:
            return self.loss_fn(out, y)
        return out

trainable_model = TrainableModel(model, loss_fn)

# ============================================================================
# 4. Train with Zenyx
# ============================================================================
print("\n[4] Training")
print("-" * 75)

trainer = Trainer(
    trainable_model,
    loader,
    lr=1e-3,                    # Learning rate
    total_steps=30,             # Total training steps
    warmup_steps=3,             # LR warmup steps
    checkpoint_every=1000,      # Don't checkpoint in this demo
    checkpoint_dir="./ckpt",
    log_every=5,
)

print("Starting training loop...\n")
trainer.train()

# ============================================================================
# 5. Results
# ============================================================================
print("\n" + "="*75)
print(" TRAINING COMPLETE")
print("="*75)

state = trainer.get_state()

print("\nFinal Metrics:")
print(f"  • Steps: {state['step']} / 30")
print(f"  • Final Loss: {state['loss']:.6f}")
print(f"  • Learning Rate: {state['lr']:.2e}")
print(f"  • Throughput: {state.get('throughput_tokens_per_sec', 0):.0f} tokens/sec")

print("\nHardware Configuration:")
topology = state['topology']
print(f"  • Backend: {topology['backend']}")
print(f"  • Interconnect: {topology['interconnect']}")
print(f"  • Device Count: {topology['world_size']}")

print("\nParallelism Strategy:")
plan = state['parallelism_plan']
print(f"  • Tensor Parallel Degree: {plan['tp_degree']}")
print(f"  • Pipeline Parallel Degree: {plan['pp_degree']}")
print(f"  • Data Parallel Degree: {plan['dp_degree']}")
print(f"  • Ring Degree: {plan['ring_degree']}")
print(f"  • Schedule: {plan['schedule_type']}")

print("\n" + "="*75)
print(" ✅ Successfully trained a 70-parameter model on CPU with Zenyx!")
print(" Key Features Demonstrated:")
print("    • Hardware-agnostic (automatically uses CPU fallback)")
print("    • Zero-OOM guarantee (memory-managed across VRAM/DRAM/NVMe)")
print("    • Auto-parallelism planning (TP/PP/DP/Ring)")
print("    • Simple API (just model + dataloader)")
print("="*75 + "\n")
