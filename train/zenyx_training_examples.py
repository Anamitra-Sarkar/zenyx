#!/usr/bin/env python3
"""
ZENYX Unified Training Examples

This script demonstrates how to use ZENYX for training on different TPUs.

EXAMPLES:
=========

# 1. List all available configurations
python3 train/zenyx_training_examples.py --list-configs

# 2. Train with a specific config template
python3 train/zenyx_training_examples.py --config v5e8-1t

# 3. Train 7B model on v5e-1
python3 train/zenyx_training_examples.py --config v5e1-7b

# 4. Fine-tune 70B model on v5e-4
python3 train/zenyx_training_examples.py --config finetune-v5e4-70b

# 5. Multi-pod training (8T on 4x TPU v5e-8)
python3 train/zenyx_training_examples.py --config multipod-v5e8x4-8t

# 6. Ultra-long context (500B with 2M token context)
python3 train/zenyx_training_examples.py --config pretrain-v5e8-500b-2m
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from zenyx_config_templates import CONFIGS, get_config, list_configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_training():
    """Example 1: Basic training with default v5e-8 config."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Training on TPU v5e-8")
    logger.info("=" * 80)
    
    config = get_config("v5e8-1t")
    
    print("\nConfiguration:")
    for key, value in config.items():
        if not callable(value):
            print(f"  {key}: {value}")
    
    print("\n✓ To run this training:")
    print("  python3 train/zenyx_unified_tpu_training.py \\")
    print("    --model-size 1e12 \\")
    print("    --tpu-version v5e-8 \\")
    print("    --batch-size 8 \\")
    print("    --learning-rate 1e-4")


def example_small_device():
    """Example 2: Training on small device (v5e-1)."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Training 7B Model on TPU v5e-1 (2GB HBM)")
    logger.info("=" * 80)
    
    config = get_config("v5e1-7b")
    
    print("\nConfiguration:")
    print(f"  Model: {config['model_size_params']/1e9:.0f}B parameters")
    print(f"  TPU: {config['tpu_version']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Max sequence: {config['max_seq_len']:,} tokens")
    
    print("\n✓ Features enabled:")
    print("  • Phase 7: KV cache tiering (8K context)")
    print("  • Phase 8: FP8 quantization")
    print("  • Phase 10: Sparse attention")
    
    print("\n✓ To run this training:")
    print("  python3 train/zenyx_unified_tpu_training.py \\")
    print("    --config v5e1-7b")


def example_multipod():
    """Example 3: Multi-pod distributed training."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Multi-Pod Training (8T on 4x TPU v5e-8)")
    logger.info("=" * 80)
    
    config = get_config("multipod-v5e8x4-8t")
    
    print("\nConfiguration:")
    print(f"  Model: {config['model_size_params']/1e12:.0f}T parameters")
    print(f"  Hardware: {config['num_tpu_pods']}x {config['tpu_version']}")
    print(f"  Total cores: {config['num_tpu_pods'] * 8}")
    print(f"  Batch size per pod: {config['batch_size']}")
    print(f"  Global batch size: {config['batch_size'] * config['num_tpu_pods']}")
    
    print("\n✓ Distributed training features:")
    print("  • Ring All-Reduce for gradient synchronization")
    print("  • Distributed data loading across pods")
    print("  • Automatic load balancing")
    print("  • All ZENYX phases enabled")
    
    print("\n✓ To run this training:")
    print("  python3 train/zenyx_unified_tpu_training.py \\")
    print("    --config multipod-v5e8x4-8t")


def example_finetuning():
    """Example 4: Fine-tuning large model."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Fine-Tuning 70B Model on TPU v5e-4")
    logger.info("=" * 80)
    
    config = get_config("finetune-v5e4-70b")
    
    print("\nConfiguration:")
    print(f"  Base model: {config['model_size_params']/1e9:.0f}B parameters")
    print(f"  TPU: {config['tpu_version']}")
    print(f"  LoRA rank: {config.get('lora_rank', 'N/A')}")
    print(f"  Learning rate: {config['learning_rate']:.0e} (lower for fine-tuning)")
    print(f"  Training steps: {config['total_steps']:,}")
    
    print("\n✓ Optimization features:")
    print("  • LoRA adapters (low-rank fine-tuning)")
    print("  • FP8 quantization")
    print("  • Lower learning rate")
    print("  • Shorter training duration")
    
    print("\n✓ To run this training:")
    print("  python3 train/zenyx_unified_tpu_training.py \\")
    print("    --config finetune-v5e4-70b")


def example_long_context():
    """Example 5: Ultra-long context training."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Ultra-Long Context (500B with 2M token context)")
    logger.info("=" * 80)
    
    config = get_config("pretrain-v5e8-500b-2m")
    
    print("\nConfiguration:")
    print(f"  Model: {config['model_size_params']/1e9:.0f}B parameters")
    print(f"  TPU: {config['tpu_version']}")
    print(f"  Max context: {config['max_seq_len']:,} tokens (2M)")
    
    print("\n✓ Phase 7 extreme context setup:")
    print("  • Three-tier memory (HBM → DRAM → NVMe)")
    print("  • Bélády-optimal page replacement")
    print("  • NVMe caching enabled")
    
    print("\n✓ Phase 9 curriculum:")
    print("  • Start: 4K tokens")
    print("  • Max: 2M tokens")
    print("  • Phases: 6 (gradual increase)")
    
    print("\n✓ To run this training:")
    print("  python3 train/zenyx_unified_tpu_training.py \\")
    print("    --config pretrain-v5e8-500b-2m")


def print_all_examples():
    """Print all examples."""
    print("\n" + "=" * 80)
    print("ZENYX UNIFIED TRAINING EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("Basic Training", "Train 1T model on TPU v5e-8", example_basic_training),
        ("Small Device", "Train 7B model on TPU v5e-1 (2GB)", example_small_device),
        ("Multi-Pod", "Train 8T model on 4x TPU v5e-8", example_multipod),
        ("Fine-Tuning", "Fine-tune 70B model on TPU v5e-4", example_finetuning),
        ("Long Context", "Train with 2M token context", example_long_context),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, desc, _) in enumerate(examples, 1):
        print(f"  {i}. {name}: {desc}")
    
    print("\n" + "=" * 80)
    print("To run an example:")
    print("  python3 train/zenyx_training_examples.py --example N")
    print("\nOr to see all configurations:")
    print("  python3 train/zenyx_training_examples.py --list-configs")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="ZENYX Unified Training Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a specific example"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations"
    )
    parser.add_argument(
        "--export-config",
        type=str,
        help="Export a configuration as JSON"
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_configs()
    elif args.export_config:
        config = get_config(args.export_config)
        print(json.dumps(config, indent=2, default=str))
    elif args.example:
        examples = [
            example_basic_training,
            example_small_device,
            example_multipod,
            example_finetuning,
            example_long_context,
        ]
        examples[args.example - 1]()
    else:
        print_all_examples()


if __name__ == "__main__":
    main()
