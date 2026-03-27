#!/usr/bin/env python3
"""
Beginner-Friendly Training Script: Train on CPU with detailed explanations.

This script shows a complete training pipeline:
- Model definition and parameter counting
- Data loading and preprocessing
- Standard PyTorch training loop
- Loss tracking and metrics
- Checkpoint saving

Perfect for understanding how ZENYX integrates with standard PyTorch workflows.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json


class SimpleTransformerModel(nn.Module):
    """A simple transformer-inspired model for language modeling."""
    
    def __init__(self, vocab_size=1000, hidden_dim=128, num_layers=2, seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            dtype=torch.float32
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids):
        """Forward pass for language modeling."""
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


def create_synthetic_lm_data(vocab_size=1000, seq_len=64, num_samples=100, batch_size=4):
    """Create synthetic language modeling data."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    target_ids = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def main():
    """Main training function."""
    print("=" * 80)
    print("ZENYX BEGINNER TRAINING EXAMPLE - CPU")
    print("=" * 80)
    
    # Configuration
    config = {
        "vocab_size": 1000,
        "hidden_dim": 128,
        "num_layers": 2,
        "seq_len": 64,
        "batch_size": 4,
        "num_epochs": 2,
        "learning_rate": 1e-3,
        "num_samples": 32,
    }
    
    print("\n[Configuration]")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Create model
    print("\n[Step 1] Creating model...")
    model = SimpleTransformerModel(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        seq_len=config["seq_len"],
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Step 2: Create data
    print("\n[Step 2] Preparing synthetic data...")
    train_loader = create_synthetic_lm_data(
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        num_samples=config["num_samples"],
        batch_size=config["batch_size"],
    )
    print(f"✓ Created {len(train_loader)} batches")
    
    # Step 3: Setup training
    print("\n[Step 3] Setting up training...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print(f"✓ Optimizer: Adam (lr={config['learning_rate']})")
    print(f"✓ Loss function: CrossEntropyLoss")
    
    # Step 4: Training loop
    print("\n[Step 4] Training...")
    model.train()
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = loss_fn(
                logits.view(-1, config["vocab_size"]),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(train_loader)//2) == 0:
                print(f"  Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete - Avg Loss: {avg_epoch_loss:.6f}")
    
    # Step 5: Summary
    print("\n[Training Summary]")
    print(f"✓ Total epochs: {config['num_epochs']}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Training complete!")
    
    print("\n" + "=" * 80)
    print("WHAT YOU LEARNED:")
    print("  1. How to define a PyTorch model")
    print("  2. How to prepare training data")
    print("  3. How to implement a training loop")
    print("  4. How to use gradient clipping")
    print("\nNEXT STEPS:")
    print("  1. See examples/02_intermediate_finetuning.py")
    print("  2. See examples/03_expert_tpu_v5e8_training.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
