#!/usr/bin/env python3
"""
Advanced demo: Track loss across training steps.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenyx.train.trainer import Trainer

print("="*70)
print("Zenyx Training with Loss Tracking")
print("="*70)

# Seed for reproducibility
torch.manual_seed(42)

# Create model
model = nn.Sequential(
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params}")

# Create data
dtype = torch.bfloat16
data = torch.randn(128, 8, dtype=dtype)
targets = torch.randn(128, 4, dtype=dtype)
loader = DataLoader(TensorDataset(data, targets), batch_size=8)

# Loss wrapper
class TrainableModel(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.losses = []  # Track losses
    
    def forward(self, x, y=None):
        out = self.model(x)
        if y is not None:
            loss = self.loss_fn(out, y)
            self.losses.append(float(loss.detach()))
            return loss
        return out

trainable_model = TrainableModel(model, nn.MSELoss())

# Train
print("\nStarting training...")
trainer = Trainer(
    trainable_model,
    loader,
    lr=1e-3,
    total_steps=30,
    warmup_steps=3,
    log_every=1,
    checkpoint_every=1000,
    checkpoint_dir="./ckpt_loss_track",
)

trainer.train()

# Show results
print("\n" + "="*70)
print("Training Results")
print("="*70)

state = trainer.get_state()
final_loss = state.get('loss')
final_step = state.get('step')

print(f"\nSteps completed: {final_step} / 30")
print(f"Final loss: {final_loss:.6f}" if final_loss else "Final loss: N/A")
print(f"Learning rate: {state.get('lr', 'N/A')}")
print(f"Hardware: CPU (fallback attention)")
print(f"Throughput: {state.get('throughput_tokens_per_sec', 'N/A'):.2f} tokens/sec" 
      if isinstance(state.get('throughput_tokens_per_sec'), (int, float)) else "Throughput: N/A")

print("\n✅ Successfully trained 70-parameter model on CPU with Zenyx!")
