#!/usr/bin/env python3
"""
Example 01: Beginner CPU Training

This example shows the absolute basics:
- Create a small model on CPU
- Load toy data
- Run a simple training loop
- Save results

Perfect for understanding the fundamentals before moving to advanced features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    """Main training function."""
    print("=" * 80)
    print("EXAMPLE 01: BEGINNER CPU TRAINING")
    print("=" * 80)
    
    # Configuration
    print("\n[Configuration]")
    vocab_size = 1000
    hidden_dim = 64
    seq_len = 32
    batch_size = 4
    num_epochs = 2
    print(f"  • Vocab size: {vocab_size}")
    print(f"  • Hidden dim: {hidden_dim}")
    print(f"  • Sequence length: {seq_len}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Num epochs: {num_epochs}")
    
    # Step 1: Create a simple model
    print("\n[Step 1] Creating model...")
    model = nn.Sequential(
        nn.Embedding(vocab_size, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {params:,} parameters")
    
    # Step 2: Create synthetic data
    print("\n[Step 2] Creating synthetic data...")
    input_ids = torch.randint(0, vocab_size, (100, seq_len))
    target_ids = torch.randint(0, vocab_size, (100, seq_len))
    
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"✓ Created {len(loader)} batches")
    
    # Step 3: Setup training
    print("\n[Step 3] Setting up training...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"✓ Ready to train!")
    
    # Step 4: Training loop
    print("\n[Step 4] Training...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (inp, tgt) in enumerate(loader):
            # Forward
            logits = model(inp)
            loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    print("\n✓ Training complete!")
    print("\nWhat you learned:")
    print("  1. How to create a PyTorch model")
    print("  2. How to create a DataLoader")
    print("  3. How to run a training loop")
    print("  4. How to compute loss and update weights")


if __name__ == "__main__":
    main()
