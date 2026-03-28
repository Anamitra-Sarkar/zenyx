"""
ZENYX GPU Configuration Templates
==================================

Pre-built configurations for various GPU setups and model sizes.
Includes single GPU, multi-GPU, and multi-node configurations.
"""

import json
from dataclasses import asdict
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUConfigTemplates:
    """Pre-built GPU training configurations"""

    # Single GPU Configurations
    SINGLE_GPU_H100_7B = {
        "gpu_type": "H100",
        "num_gpus": 1,
        "model_size": 7e9,
        "hidden_dim": 4096,
        "num_layers": 32,
        "seq_length": 4096,
        "batch_size": 32,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    SINGLE_GPU_H100_30B = {
        "gpu_type": "H100",
        "num_gpus": 1,
        "model_size": 30e9,
        "hidden_dim": 6144,
        "num_layers": 48,
        "seq_length": 4096,
        "batch_size": 8,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-5,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": True,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    SINGLE_GPU_H100_70B = {
        "gpu_type": "H100",
        "num_gpus": 1,
        "model_size": 70e9,
        "hidden_dim": 8192,
        "num_layers": 80,
        "seq_length": 4096,
        "batch_size": 2,
        "gradient_accumulation_steps": 32,
        "learning_rate": 3e-5,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    SINGLE_GPU_A100_7B = {
        "gpu_type": "A100",
        "num_gpus": 1,
        "model_size": 7e9,
        "hidden_dim": 4096,
        "num_layers": 32,
        "seq_length": 4096,
        "batch_size": 16,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    SINGLE_GPU_RTX4090_7B = {
        "gpu_type": "RTX4090",
        "num_gpus": 1,
        "model_size": 7e9,
        "hidden_dim": 4096,
        "num_layers": 32,
        "seq_length": 2048,
        "batch_size": 8,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    # Multi-GPU Configurations (Data Parallel)
    MULTI_GPU_2X_H100_13B = {
        "gpu_type": "H100",
        "num_gpus": 2,
        "model_size": 13e9,
        "hidden_dim": 5120,
        "num_layers": 40,
        "seq_length": 4096,
        "batch_size": 64,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    MULTI_GPU_4X_H100_34B = {
        "gpu_type": "H100",
        "num_gpus": 4,
        "model_size": 34e9,
        "hidden_dim": 6144,
        "num_layers": 48,
        "seq_length": 4096,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    MULTI_GPU_8X_H100_70B = {
        "gpu_type": "H100",
        "num_gpus": 8,
        "model_size": 70e9,
        "hidden_dim": 8192,
        "num_layers": 80,
        "seq_length": 4096,
        "batch_size": 256,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    MULTI_GPU_8X_A100_70B = {
        "gpu_type": "A100",
        "num_gpus": 8,
        "model_size": 70e9,
        "hidden_dim": 8192,
        "num_layers": 80,
        "seq_length": 4096,
        "batch_size": 128,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    # Multi-Node Configurations
    MULTI_NODE_4X8_H100_140B = {
        "gpu_type": "H100",
        "num_gpus": 8,
        "num_nodes": 4,
        "model_size": 140e9,
        "hidden_dim": 8192,
        "num_layers": 80,
        "seq_length": 8192,
        "batch_size": 512,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    MULTI_NODE_8X8_H100_280B = {
        "gpu_type": "H100",
        "num_gpus": 8,
        "num_nodes": 8,
        "model_size": 280e9,
        "hidden_dim": 10240,
        "num_layers": 96,
        "seq_length": 16384,
        "batch_size": 1024,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    # Long-Context Training (with Phase 7 Cache Tiering)
    LONG_CONTEXT_H100_7B_128K = {
        "gpu_type": "H100",
        "num_gpus": 8,
        "model_size": 7e9,
        "hidden_dim": 4096,
        "num_layers": 32,
        "seq_length": 131072,  # 128K tokens
        "batch_size": 2,
        "gradient_accumulation_steps": 64,
        "learning_rate": 1e-4,
        "max_steps": 100000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    LONG_CONTEXT_H100_7B_1M = {
        "gpu_type": "H100",
        "num_gpus": 8,
        "num_nodes": 4,
        "model_size": 7e9,
        "hidden_dim": 4096,
        "num_layers": 32,
        "seq_length": 1048576,  # 1M tokens
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "learning_rate": 5e-5,
        "max_steps": 50000,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": True,
        "enable_fp8": True,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    # Fine-tuning Configurations
    FINETUNE_H100_7B = {
        "gpu_type": "H100",
        "num_gpus": 1,
        "model_size": 7e9,
        "hidden_dim": 4096,
        "num_layers": 32,
        "seq_length": 4096,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "max_steps": 5000,
        "warmup_steps": 100,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    FINETUNE_2X_H100_30B = {
        "gpu_type": "H100",
        "num_gpus": 2,
        "model_size": 30e9,
        "hidden_dim": 6144,
        "num_layers": 48,
        "seq_length": 4096,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-5,
        "max_steps": 5000,
        "warmup_steps": 100,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "enable_cache_tiering": False,
        "enable_fp8": True,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """List all available templates"""
        templates = {}
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper() and attr_name != "list_templates":
                attr = getattr(cls, attr_name)
                if isinstance(attr, dict):
                    description = f"{attr['num_gpus']}x {attr['gpu_type']}"
                    if attr['num_gpus'] > 1 and attr.get('num_nodes', 1) > 1:
                        description += f" ({attr['num_nodes']} nodes)"
                    description += f" - {attr['model_size']/1e9:.0f}B"
                    templates[attr_name] = description
        return templates

    @classmethod
    def get_template(cls, template_name: str) -> Dict[str, Any]:
        """Get a specific template"""
        if hasattr(cls, template_name):
            template = getattr(cls, template_name)
            if isinstance(template, dict):
                return template.copy()
        raise ValueError(f"Template {template_name} not found")

    @classmethod
    def print_all_templates(cls):
        """Print all available templates"""
        print("\n" + "=" * 100)
        print("ZENYX GPU Training Configuration Templates")
        print("=" * 100)

        templates = cls.list_templates()
        categories = {
            "Single GPU": [],
            "Multi-GPU": [],
            "Multi-Node": [],
            "Long-Context": [],
            "Fine-tuning": [],
        }

        for name, desc in templates.items():
            if "SINGLE_GPU" in name:
                categories["Single GPU"].append((name, desc))
            elif "MULTI_NODE" in name:
                categories["Multi-Node"].append((name, desc))
            elif "LONG_CONTEXT" in name:
                categories["Long-Context"].append((name, desc))
            elif "FINETUNE" in name:
                categories["Fine-tuning"].append((name, desc))
            else:
                categories["Multi-GPU"].append((name, desc))

        for category, items in categories.items():
            if items:
                print(f"\n{category}:")
                print("-" * 100)
                for name, desc in items:
                    print(f"  {name:<40} {desc}")

        print("\n" + "=" * 100)
        print("Usage: python zenyx_distributed_gpu_training.py --template <template_name> [--override-key value]")
        print("=" * 100 + "\n")


def print_template_details(template_name: str):
    """Print detailed information about a specific template"""
    try:
        config = GPUConfigTemplates.get_template(template_name)
        print(f"\n{template_name} Configuration:")
        print("=" * 80)
        for key, value in sorted(config.items()):
            if isinstance(value, float):
                if value >= 1e9:
                    print(f"  {key:<30} {value/1e9:.0f}B")
                elif value < 0.1:
                    print(f"  {key:<30} {value:.2e}")
                else:
                    print(f"  {key:<30} {value:.4f}")
            else:
                print(f"  {key:<30} {value}")
        print("=" * 80 + "\n")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Print all templates
    GPUConfigTemplates.print_all_templates()

    # Print example template
    print("\nExample: Detailed view of MULTI_GPU_8X_H100_70B")
    print_template_details("MULTI_GPU_8X_H100_70B")
