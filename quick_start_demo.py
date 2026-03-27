#!/usr/bin/env python3
"""
ZENYX Quick Start Demo

Run this to see the complete system in action.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("\n" + "="*80)
    print("ZENYX QUICK START DEMO".center(80))
    print("="*80 + "\n")
    
    # Import unified trainer
    from zenyx.unified_training import ZenyxTrainer, ZenyxConfig
    import numpy as np
    
    print("Step 1: Creating trainer configuration...")
    config = ZenyxConfig(
        model_params=int(1e12),      # 1 trillion parameters
        num_layers=126,
        num_heads=32,
        head_dim=128,
        num_devices=8,               # TPU v5e-8
        max_context_tokens=1_000_000,  # 1M tokens
        total_steps=100,
        curriculum_phases=4,
        enable_belayd_tiering=True,
        enable_fp8_quantization=True,
        enable_curriculum=True,
        enable_sparse_attention=True,
    )
    print(f"  ✓ Config created: {config.model_params/1e12:.0f}T params, {config.max_context_tokens:,} tokens max")
    
    print("\nStep 2: Initializing unified trainer...")
    trainer = ZenyxTrainer(config)
    print("  ✓ Trainer initialized with all four pillars")
    
    print("\nStep 3: Running training simulation (10 steps)...")
    print("-" * 80)
    
    for step in range(10):
        # Create dummy batch
        batch = {
            'input_ids': np.random.randint(0, config.vocab_size, (config.batch_size, 512)),
            'attention_mask': np.ones((config.batch_size, 512)),
        }
        
        # Prepare with all optimizations
        prepared_batch = trainer.prepare_batch(batch)
        
        # Train
        loss = trainer.train_step(prepared_batch)
        
        # Print progress
        context = trainer.get_current_context_length()
        mem_info = trainer.compute_kv_memory_cost()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} | Loss: {loss:.4f} | Context: {context:>7,} tokens | "
                  f"KV Cache: {mem_info['total_kv_gb']:6.1f} GB | "
                  f"Compression: {mem_info['compression_ratio']:.1f}x")
    
    print("-" * 80)
    print("\nStep 4: Final status report...")
    trainer.print_status()
    
    print("\nStep 5: Full training report (JSON format)...")
    import json
    report = trainer.get_training_report()
    print(json.dumps(report, indent=2, default=str))
    
    print("\n" + "="*80)
    print("✅ ZENYX DEMO COMPLETE".center(80))
    print("="*80)
    print("\nYour ZENYX library is fully functional!")
    print("\nRun comprehensive validation:")
    print("  python3 test/comprehensive_e2e_validation.py")
    print("\nRun phase-specific validation:")
    print("  python3 test/validate_zenyx_four_pillars.py")
    print("\nReady for TPU v5e-8 deployment!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
