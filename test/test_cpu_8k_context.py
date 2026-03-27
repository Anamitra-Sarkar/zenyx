#!/usr/bin/env python3
"""
CPU Training Test: 8K Context (Scale Test towards 32K)
Validates that Zenyx can train models with increasing context on CPU
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
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
            yield (tokens,)

class SimpleModel(nn.Module):
    """Simple feedforward model"""
    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 128, num_layers: int = 2):
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


def run_test(context_len, num_steps=5):
    """Run a single test with given context length"""
    print(f"\n{'='*80}")
    print(f"Testing CPU Training: {context_len:,} token context")
    print(f"{'='*80}")
    
    VOCAB_SIZE = 50257
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 1
    
    print(f"\nConfig: {HIDDEN_DIM}D model, {num_steps} steps, {context_len:,} context")
    
    # Create model
    print(f"Creating model...")
    model = SimpleModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ {num_params:,} parameters")
    
    # Create data loader
    print(f"Creating data loader...")
    dataset = RandomTokenDataset(seq_len=context_len, vocab_size=VOCAB_SIZE, num_batches=num_steps)
    loader = DataLoader(dataset, batch_size=1)
    print(f"  ✓ Dataset ready")
    
    # Create trainer
    print(f"Creating Zenyx Trainer...")
    trainer = Trainer(
        model=model,
        dataloader=loader,
        lr=LEARNING_RATE,
        total_steps=num_steps,
        warmup_steps=WARMUP_STEPS,
        dtype="bfloat16",
        context_len=context_len,
        checkpoint_dir=f"/tmp/zenyx_cpu_{context_len}",
        checkpoint_every=num_steps + 1,
    )
    print(f"  ✓ Trainer ready")
    
    # Train
    print(f"\nTraining...")
    start = time.time()
    try:
        trainer.train()
        elapsed = time.time() - start
        state = trainer.get_state()
        
        print(f"\n✅ SUCCESS!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps: {state.get('step', 'N/A')}/{num_steps}")
        print(f"  Loss: {state.get('loss', 'N/A'):.4f}")
        
        plan = state.get("parallelism_plan", {})
        print(f"\nParallelism: TP={plan.get('tp_degree', 1)} PP={plan.get('pp_degree', 1)} DP={plan.get('dp_degree', 1)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


def main():
    """Test CPU training at different context lengths"""
    print("\n" + "="*80)
    print("CPU CONTEXT LENGTH SCALING TEST")
    print("README claims: CPU can handle 32K context with chunked attention")
    print("="*80)
    
    results = {}
    
    # Test increasing context lengths
    context_lengths = [1024, 2048, 4096, 8192]
    
    for ctx_len in context_lengths:
        success = run_test(ctx_len, num_steps=3)
        results[ctx_len] = success
        if not success:
            print(f"\n⚠️  Training failed at {ctx_len} context - stopping")
            break
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for ctx_len, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {ctx_len:6,} tokens: {status}")
    
    if all(results.values()):
        print(f"\n✅ ALL TESTS PASSED")
        return True
    else:
        print(f"\n⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
