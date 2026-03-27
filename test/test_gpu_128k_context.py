#!/usr/bin/env python3
"""
GPU Training Test: 2xT4 with 128K Context
Tests distributed training on 2 NVIDIA Tesla T4 GPUs (128K context per README)
Run with: torchrun --nproc_per_node=2 test_gpu_128k_context.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from zenyx.train.trainer import Trainer
import time
import os
import sys

class RandomTokenDataset(IterableDataset):
    """Generate random token sequences"""
    def __init__(self, seq_len: int, vocab_size: int = 50257, num_batches: int = 10):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
            yield (tokens,)

class SimpleModel(nn.Module):
    """Simple feedforward model"""
    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
              for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        x = self.embed(tokens)
        x = self.layers(x)
        logits = self.lm_head(x)
        return logits.squeeze(0) if logits.shape[0] == 1 else logits


def test_gpu_128k_context():
    """Test GPU training with 128K context on 2xT4"""
    
    print("\n" + "="*80)
    print("GPU TRAINING TEST: 2xT4 with 128K Context")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ ERROR: CUDA not available!")
        print("This test requires NVIDIA GPU support.")
        return False
    
    print(f"\nGPU Setup:")
    print(f"  Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    if torch.cuda.device_count() < 2:
        print(f"\n⚠️  WARNING: This test expects 2 GPUs, found {torch.cuda.device_count()}")
        print("Continuing with available GPUs...")
    
    # Configuration
    CONTEXT_LENGTH = 131072  # 128K tokens
    VOCAB_SIZE = 50257
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    LEARNING_RATE = 1e-4
    NUM_TRAINING_STEPS = 5
    WARMUP_STEPS = 1
    
    print(f"\nConfiguration:")
    print(f"  Context Length: {CONTEXT_LENGTH:,} tokens")
    print(f"  Model: {HIDDEN_DIM}D, {NUM_LAYERS} layers")
    print(f"  Training Steps: {NUM_TRAINING_STEPS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # Create model
    print(f"\n[1/4] Creating model...")
    model = SimpleModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model: {num_params:,} parameters ({num_params * 2 / 1e9:.2f} GB bfloat16)")
    
    # Create data loader
    print(f"\n[2/4] Creating data loader...")
    dataset = RandomTokenDataset(
        seq_len=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        num_batches=NUM_TRAINING_STEPS,
    )
    loader = DataLoader(dataset, batch_size=1)
    print(f"  ✓ Dataset ready ({CONTEXT_LENGTH:,} tokens per batch)")
    
    # Create trainer
    print(f"\n[3/4] Creating Zenyx Trainer...")
    trainer = Trainer(
        model=model,
        dataloader=loader,
        lr=LEARNING_RATE,
        total_steps=NUM_TRAINING_STEPS,
        warmup_steps=WARMUP_STEPS,
        dtype="bfloat16",
        context_len=CONTEXT_LENGTH,
        checkpoint_dir="/tmp/zenyx_gpu_128k",
        checkpoint_every=NUM_TRAINING_STEPS + 1,
    )
    print(f"  ✓ Trainer configured")
    
    # Train
    print(f"\n[4/4] Training on GPU...")
    start_time = time.time()
    try:
        trainer.train()
        elapsed = time.time() - start_time
        state = trainer.get_state()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"  Training Time: {elapsed:.2f}s")
        print(f"  Steps Completed: {state.get('step', 'N/A')}/{NUM_TRAINING_STEPS}")
        print(f"  Final Loss: {state.get('loss', 'N/A'):.4f}")
        
        # Validate parallelism plan
        plan = state.get("parallelism_plan", {})
        print(f"\nParallelism Plan:")
        print(f"  Tensor Parallelism: {plan.get('tp_degree', 1)}")
        print(f"  Pipeline Parallelism: {plan.get('pp_degree', 1)}")
        print(f"  Data Parallelism: {plan.get('dp_degree', 1)}")
        print(f"  Ring Degree: {plan.get('ring_degree', 1)}")
        
        print(f"\n{'='*80}")
        print(f"✅ GPU 128K CONTEXT TEST PASSED!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gpu_128k_context()
    sys.exit(0 if success else 1)
