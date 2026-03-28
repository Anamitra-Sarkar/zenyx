"""
ZENYX CPU Configuration Templates
==================================

Pre-built configurations for CPU training on single and multi-worker setups.
Optimized for efficient multi-core utilization.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CPUConfigTemplates:
    """Pre-built CPU training configurations"""

    # Single Core Configurations (minimal memory, slow)
    SINGLE_CORE_200M = {
        "num_workers": 1,
        "model_size": 200e6,
        "hidden_dim": 768,
        "num_layers": 12,
        "seq_length": 512,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    SINGLE_CORE_500M = {
        "num_workers": 1,
        "model_size": 500e6,
        "hidden_dim": 1024,
        "num_layers": 18,
        "seq_length": 1024,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    SINGLE_CORE_1B = {
        "num_workers": 1,
        "model_size": 1e9,
        "hidden_dim": 2048,
        "num_layers": 24,
        "seq_length": 512,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-5,
        "max_steps": 30000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    # Multi-Core Configurations (better throughput)
    DUAL_CORE_200M = {
        "num_workers": 2,
        "num_threads_per_worker": 2,
        "model_size": 200e6,
        "hidden_dim": 768,
        "num_layers": 12,
        "seq_length": 1024,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    QUAD_CORE_500M = {
        "num_workers": 4,
        "num_threads_per_worker": 2,
        "model_size": 500e6,
        "hidden_dim": 1024,
        "num_layers": 18,
        "seq_length": 1024,
        "batch_size": 64,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    OCTA_CORE_700M = {
        "num_workers": 8,
        "num_threads_per_worker": 2,
        "model_size": 700e6,
        "hidden_dim": 1536,
        "num_layers": 20,
        "seq_length": 1024,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-5,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    OCTA_CORE_1B = {
        "num_workers": 8,
        "num_threads_per_worker": 2,
        "model_size": 1e9,
        "hidden_dim": 2048,
        "num_layers": 24,
        "seq_length": 2048,
        "batch_size": 64,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    # High Core Count Configurations (16+ cores)
    HIGH_CORE_16CORE_500M = {
        "num_workers": 16,
        "num_threads_per_worker": 1,
        "model_size": 500e6,
        "hidden_dim": 1024,
        "num_layers": 18,
        "seq_length": 1024,
        "batch_size": 256,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    HIGH_CORE_32CORE_1B = {
        "num_workers": 32,
        "num_threads_per_worker": 1,
        "model_size": 1e9,
        "hidden_dim": 2048,
        "num_layers": 24,
        "seq_length": 2048,
        "batch_size": 512,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "max_steps": 50000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": False,
    }

    # Fine-tuning Configurations
    FINETUNE_SINGLE_CORE_200M = {
        "num_workers": 1,
        "model_size": 200e6,
        "hidden_dim": 768,
        "num_layers": 12,
        "seq_length": 512,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "max_steps": 5000,
        "warmup_steps": 100,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    FINETUNE_QUAD_CORE_500M = {
        "num_workers": 4,
        "num_threads_per_worker": 2,
        "model_size": 500e6,
        "hidden_dim": 1024,
        "num_layers": 18,
        "seq_length": 1024,
        "batch_size": 64,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-5,
        "max_steps": 5000,
        "warmup_steps": 100,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": False,
        "enable_fp8": False,
        "enable_curriculum": False,
        "enable_sparse_attention": False,
    }

    # Long-Context (requires cache tiering)
    LONG_CONTEXT_OCTA_CORE_200M = {
        "num_workers": 8,
        "num_threads_per_worker": 2,
        "model_size": 200e6,
        "hidden_dim": 768,
        "num_layers": 12,
        "seq_length": 32768,  # 32K tokens
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "max_steps": 25000,
        "use_gradient_checkpointing": True,
        "enable_cache_tiering": True,
        "enable_fp8": False,
        "enable_curriculum": True,
        "enable_sparse_attention": True,
    }

    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """List all available templates"""
        templates = {}
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper() and attr_name != "list_templates":
                attr = getattr(cls, attr_name)
                if isinstance(attr, dict):
                    description = f"{attr['num_workers']} worker(s)"
                    description += f" - {attr['model_size']/1e6:.0f}M"
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
        print("ZENYX CPU Training Configuration Templates")
        print("=" * 100)

        templates = cls.list_templates()
        categories = {
            "Single Core": [],
            "Multi-Core (2-8)": [],
            "High Core Count (16+)": [],
            "Fine-tuning": [],
            "Long-Context": [],
        }

        for name, desc in templates.items():
            if "SINGLE_CORE" in name:
                categories["Single Core"].append((name, desc))
            elif "LONG_CONTEXT" in name:
                categories["Long-Context"].append((name, desc))
            elif "FINETUNE" in name:
                categories["Fine-tuning"].append((name, desc))
            elif any(x in name for x in ["DUAL_CORE", "QUAD_CORE", "OCTA_CORE"]):
                categories["Multi-Core (2-8)"].append((name, desc))
            else:
                categories["High Core Count (16+)"].append((name, desc))

        for category, items in categories.items():
            if items:
                print(f"\n{category}:")
                print("-" * 100)
                for name, desc in items:
                    print(f"  {name:<40} {desc}")

        print("\n" + "=" * 100)
        print("Usage: python zenyx_cpu_training.py --template <template_name> [--override-key value]")
        print("=" * 100 + "\n")


def print_template_details(template_name: str):
    """Print detailed information about a specific template"""
    try:
        config = CPUConfigTemplates.get_template(template_name)
        print(f"\n{template_name} Configuration:")
        print("=" * 80)
        for key, value in sorted(config.items()):
            if isinstance(value, float):
                if value >= 1e6:
                    print(f"  {key:<30} {value/1e6:.0f}M")
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
    CPUConfigTemplates.print_all_templates()

    # Print example template
    print("\nExample: Detailed view of OCTA_CORE_1B")
    print_template_details("OCTA_CORE_1B")
