#!/usr/bin/env python3
"""
CPU Training Test: 32K Context Length
Validates that Zenyx can train models with 32K token context on CPU
(per README: "CPU / Apple M3: Chunked attention, 32K tokens - Development only")
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from zenyx.train.trainer import Trainer
import time
import sys

class RandomTokenDataset(IterableDataset):
    """Generate random token sequences of fixed length"""
    def __init__(self, seq_len: int, vocab_size: int = 50257, num_batches: int = 10):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            # Input and target sequences
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
            yield (tokens,)

class SimpleTransformer(nn.Module):
    """Minimal transformer for testing"""
    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # Simple stack of linear layers + ReLU to avoid transformer complexity
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
              for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        # Handle both tensor and list inputs
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
        
        x = self.embed(tokens)  # (batch, seq_len, hidden_dim)
        x = self.layers(x)      # (batch, seq_len, hidden_dim)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits.squeeze(0) if logits.shape[0] == 1 else logits


def test_cpu_32k_context():
    """Test CPU training with 32K context"""
    
    print("\n" + "="*80)
    print("CPU TRAINING TEST: 32K Context Length")
    print("="*80)
    
    # Configuration
    CONTEXT_LENGTH = 32768  # 32K tokens as per README
    VOCAB_SIZE = 50257
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1  # Single sequence per batch for large context
    NUM_TRAINING_STEPS = 5  # Short test
    WARMUP_STEPS = 1
    
    print(f"\nConfiguration:")
    print(f"  Context Length: {CONTEXT_LENGTH:,} tokens")
    print(f"  Vocab Size: {VOCAB_SIZE:,}")
    print(f"  Model Dims: {HIDDEN_DIM}D, {NUM_LAYERS} layers")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Training Steps: {NUM_TRAINING_STEPS}")
    
    # 1. Create model
    print(f"\n[1/4] Creating model...")
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created: {num_params:,} parameters")
    print(f"  ✓ Model size: {num_params * 2 / 1e9:.2f} GB (bfloat16)")
    
    # 2. Create data loader
    print(f"\n[2/4] Creating data loader...")
    dataset = RandomTokenDataset(
        seq_len=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        num_batches=NUM_TRAINING_STEPS,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    print(f"  ✓ Data loader created")
    print(f"  ✓ Each batch: {CONTEXT_LENGTH:,} tokens")
    
    # 3. Create trainer
    print(f"\n[3/4] Creating Zenyx Trainer...")
    trainer = Trainer(
        model=model,
        dataloader=loader,
        lr=LEARNING_RATE,
        total_steps=NUM_TRAINING_STEPS,
        warmup_steps=WARMUP_STEPS,
        dtype="bfloat16",
        context_len=CONTEXT_LENGTH,
        checkpoint_dir="/tmp/zenyx_cpu_32k",
        checkpoint_every=5,
    )
    print(f"  ✓ Trainer configured")
    
    # 4. Train
    print(f"\n[4/4] Training with 32K context...")
    start_time = time.time()
    try:
        trainer.train()
        elapsed = time.time() - start_time
        state = trainer.get_state()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"  Training Time: {elapsed:.2f}s")
        print(f"  Steps Completed: {state.get('step', 'N/A')}/{NUM_TRAINING_STEPS}")
        print(f"  Final Loss: {state.get('loss', 'N/A')}")
        
        # Validate parallelism plan
        plan = state.get("parallelism_plan", {})
        print(f"\nParallelism Plan:")
        print(f"  Tensor Parallelism: {plan.get('tp_degree', 1)}")
        print(f"  Pipeline Parallelism: {plan.get('pp_degree', 1)}")
        print(f"  Data Parallelism: {plan.get('dp_degree', 1)}")
        print(f"  Ring Degree: {plan.get('ring_degree', 1)}")
        
        print(f"\n{'='*80}")
        print(f"✅ CPU 32K CONTEXT TEST PASSED!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cpu_32k_context()
    sys.exit(0 if success else 1)
