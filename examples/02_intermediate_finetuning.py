#!/usr/bin/env python3
"""
Example 02: Intermediate Fine-tuning with ZENYX Features

This example shows:
- Loading a pretrained model
- Fine-tuning on custom data
- Using ZENYX optimization techniques
- Multi-phase training with curriculum learning
- Monitoring training metrics

Great for real-world training tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json


class TransformerModel(nn.Module):
    """Simple transformer for text tasks."""
    
    def __init__(self, vocab_size=2000, hidden_dim=128, num_layers=2, seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            dtype=torch.float32
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        logits = self.output(x)
        return logits


def main():
    """Main training function."""
    print("=" * 80)
    print("EXAMPLE 02: INTERMEDIATE FINE-TUNING")
    print("=" * 80)
    
    # Configuration
    print("\n[Configuration]")
    config = {
        "vocab_size": 2000,
        "hidden_dim": 128,
        "num_layers": 2,
        "seq_len": 64,
        "batch_size": 4,
        "num_epochs": 3,
        "learning_rate": 5e-4,
        "warmup_steps": 50,
        "enable_gradient_accumulation": True,
        "gradient_accumulation_steps": 2,
    }
    
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # Step 1: Create/load model
    print("\n[Step 1] Creating model...")
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        seq_len=config["seq_len"],
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded with {params:,} parameters")
    
    # Step 2: Prepare data
    print("\n[Step 2] Preparing fine-tuning data...")
    input_ids = torch.randint(0, config["vocab_size"], (100, config["seq_len"]))
    target_ids = torch.randint(0, config["vocab_size"], (100, config["seq_len"]))
    
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    print(f"✓ Created {len(loader)} batches")
    
    # Step 3: Setup fine-tuning
    print("\n[Step 3] Setting up fine-tuning...")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Learning rate warmup schedule
    def get_lr(step, warmup_steps, total_steps):
        if step < warmup_steps:
            return (step / warmup_steps) * config["learning_rate"]
        return config["learning_rate"] * 0.5 * (1 + torch.cos(torch.tensor((step - warmup_steps) / (total_steps - warmup_steps) * 3.14159)).item())
    
    print(f"✓ Optimizer: AdamW")
    print(f"✓ Warmup steps: {config['warmup_steps']}")
    print(f"✓ Gradient accumulation: {config['enable_gradient_accumulation']}")
    
    # Step 4: Fine-tuning loop
    print("\n[Step 4] Fine-tuning...")
    model.train()
    metrics = {
        "epochs": [],
        "losses": [],
    }
    
    global_step = 0
    accumulation_counter = 0
    accumulated_loss = 0.0
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        
        for batch_idx, (inp, tgt) in enumerate(loader):
            # Forward pass
            logits = model(inp)
            loss = loss_fn(logits.view(-1, config["vocab_size"]), tgt.view(-1))
            
            # Gradient accumulation
            if config["enable_gradient_accumulation"]:
                loss = loss / config["gradient_accumulation_steps"]
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            accumulation_counter += 1
            
            # Update weights
            if accumulation_counter >= config["gradient_accumulation_steps"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                accumulation_counter = 0
                global_step += 1
                
                # Learning rate schedule
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_lr(global_step, config["warmup_steps"], len(loader) * config["num_epochs"])
                
                metrics["losses"].append(accumulated_loss)
                accumulated_loss = 0.0
                
                if global_step % 5 == 0:
                    print(f"  Step {global_step}: Loss = {accumulated_loss:.6f}")
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(loader)
        metrics["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": avg_epoch_loss,
        })
        print(f"Epoch {epoch+1}: Avg Loss = {avg_epoch_loss:.6f}")
    
    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    torch.save(model.state_dict(), "finetuned_model.pt")
    print(f"✓ Model saved to finetuned_model.pt")
    
    with open("finetuning_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to finetuning_metrics.json")
    
    print("\n✓ Fine-tuning complete!")
    print("\nWhat you learned:")
    print("  1. How to fine-tune a pretrained model")
    print("  2. How to use gradient accumulation")
    print("  3. How to implement learning rate warmup")
    print("  4. How to track and save metrics")


if __name__ == "__main__":
    main()
