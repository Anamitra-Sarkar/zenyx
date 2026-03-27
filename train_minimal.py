#!/usr/bin/env python3
"""
Minimal Training Script: Train a small model on CPU using Zenyx.

This script demonstrates the simplest way to get started with Zenyx:
- A tiny neural network (~108 parameters)
- Synthetic data on CPU
- Standard PyTorch training loop
- Complete in under 1 minute

Use this as a quickstart template for your own training scripts.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_tiny_model():
    """Create a small model suitable for CPU training."""
    return nn.Sequential(
        nn.Linear(8, 8),      # 8*8 + 8 = 72 params
        nn.ReLU(),
        nn.Linear(8, 4),      # 8*4 + 4 = 36 params
    )


def create_synthetic_data(num_samples=128, input_dim=8, output_dim=4, batch_size=8):
    """Create synthetic training data."""
    # Use float32 for CPU compatibility (bfloat16 only on TPU/specific GPUs)
    data = torch.randn(num_samples, input_dim, dtype=torch.float32)
    targets = torch.randn(num_samples, output_dim, dtype=torch.float32)
    
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def main():
    """Main training function."""
    print("=" * 80)
    print("ZENYX MINIMAL TRAINING EXAMPLE - CPU")
    print("=" * 80)
    
    # Step 1: Create model
    print("\n[Step 1] Creating model...")
    model = create_tiny_model()
    model = model.to(torch.float32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params} parameters")
    print(f"  Architecture: Linear(8→8) → ReLU → Linear(8→4)")
    
    # Step 2: Create data
    print("\n[Step 2] Preparing synthetic data...")
    train_loader = create_synthetic_data(num_samples=128, batch_size=8)
    print(f"✓ Created {len(train_loader)} batches of size 8")
    
    # Step 3: Setup training
    print("\n[Step 3] Setting up training...")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 2
    print(f"✓ Optimizer: Adam (lr=1e-3)")
    print(f"✓ Loss: MSELoss")
    print(f"✓ Epochs: {num_epochs}")
    
    # Step 4: Training loop
    print("\n[Step 4] Training...")
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 4 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        total_loss += epoch_loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete - Avg Loss: {avg_epoch_loss:.6f}")
    
    # Step 5: Summary
    print("\n[Step 5] Training complete!")
    print(f"✓ Average loss: {total_loss / (num_epochs * len(train_loader)):.6f}")
    print(f"✓ Total batches processed: {num_batches}")
    print(f"✓ Model ready for evaluation or fine-tuning")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("  1. See examples/01_beginner_cpu_training.py for a beginner-friendly guide")
    print("  2. See examples/02_intermediate_finetuning.py to add Zenyx features")
    print("  3. See examples/03_expert_tpu_v5e8_training.py for advanced TPU training")
    print("=" * 80)


if __name__ == "__main__":
    main()
