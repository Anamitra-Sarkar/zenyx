#!/usr/bin/env python3
"""
Minimal training script: Train a 70-parameter model on CPU using Zenyx.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenyx.train.trainer import Trainer

# Create a tiny model with ~70 parameters
# Simple architecture: 10 -> 16 -> 32 -> 10
model = nn.Sequential(
    nn.Linear(10, 16),      # 10*16 + 16 = 176 params
    nn.ReLU(),
    nn.Linear(16, 32),      # 16*32 + 32 = 544 params
    nn.ReLU(),
    nn.Linear(32, 10),      # 32*10 + 10 = 330 params
)

# Let's make it smaller to get ~70 params
model = nn.Sequential(
    nn.Linear(8, 8),        # 8*8 + 8 = 72 params
    nn.ReLU(),
    nn.Linear(8, 4),        # 8*4 + 4 = 36 params
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params} parameters")

# Create simple synthetic data
dtype = torch.bfloat16  # Match Trainer's default dtype
data = torch.randn(128, 8, dtype=dtype)  # 128 samples of 8-dim vectors
targets = torch.randn(128, 4, dtype=dtype)  # 128 targets of 4-dim vectors

dataset = TensorDataset(data, targets)
loader = DataLoader(dataset, batch_size=8)

# Create loss function
loss_fn = nn.MSELoss()

# Wrap model to output loss for Zenyx
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

trainable_model = TrainableModel(model, loss_fn)

# Train with Zenyx
print("\nStarting training with Zenyx...")
trainer = Trainer(
    trainable_model,
    loader,
    lr=1e-3,
    total_steps=20,
    warmup_steps=2,
    checkpoint_every=100,  # Don't checkpoint during this short run
    checkpoint_dir="./checkpoints_minimal",
)

trainer.train()
state = trainer.get_state()

print("\nTraining complete!")
print(f"Final loss: {state.get('last_loss', 'N/A')}")
print(f"Steps completed: {state.get('step', 'N/A')}")
print(f"Parallelism plan: {state.get('parallelism_plan', 'N/A')}")
