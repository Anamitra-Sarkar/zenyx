#!/usr/bin/env python3
"""
ZENYX Configuration Templates for Different TPU Scenarios

This file contains pre-built configurations for common training setups:
- Single device training (v5e-1, v5e-4, v5e-8)
- Multi-pod training (v5e-8 x2, x4, x8)
- Different model sizes
- Different training objectives

Each configuration is optimized for maximum efficiency and memory usage.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import json


# ============================================================================
# SINGLE DEVICE CONFIGURATIONS
# ============================================================================

@dataclass
class ConfigV5e1_7B:
    """
    Configuration for 7B parameter model on single TPU v5e-1 (2GB HBM)
    
    Model: LLaMA-7B equivalent
    Hardware: 1x TPU v5e-1 (2GB HBM, 1 core)
    Context: 8K tokens
    Batch size: 1
    
    Features:
    - Phase 7: KV cache tiering (8K context)
    - Phase 8: FP8 quantization enabled
    - Phase 9: Curriculum disabled (too small)
    - Phase 10: Sparse attention for efficiency
    """
    
    # Model
    model_size_params: int = int(7e9)
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    max_seq_len: int = 8192
    
    # Hardware
    tpu_version: str = "v5e-1"
    num_tpu_pods: int = 1
    batch_size: int = 1
    
    # Training
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    total_steps: int = 10000
    
    # ZENYX Features
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = False  # Too small
    enable_phase10_sparse_attention: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_v5e1_7b"
    checkpoint_every_steps: int = 100


@dataclass
class ConfigV5e4_30B:
    """
    Configuration for 30B parameter model on single TPU v5e-4 (8GB HBM)
    
    Model: LLaMA-30B equivalent
    Hardware: 1x TPU v5e-4 (8GB HBM, 4 cores)
    Context: 32K tokens
    Batch size: 4
    
    Features:
    - Phase 7: KV cache tiering (32K context)
    - Phase 8: FP8 quantization enabled
    - Phase 9: Basic curriculum learning
    - Phase 10: Full sparse attention
    """
    
    # Model
    model_size_params: int = int(30e9)
    vocab_size: int = 32000
    hidden_dim: int = 6656
    num_layers: int = 60
    num_heads: int = 52
    max_seq_len: int = 32768
    
    # Hardware
    tpu_version: str = "v5e-4"
    num_tpu_pods: int = 1
    batch_size: int = 4
    
    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 2000
    total_steps: int = 50000
    
    # ZENYX Features
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = True
    enable_phase10_sparse_attention: bool = True
    
    # Phase 9 Config
    curriculum_start_context: int = 4096
    curriculum_max_context: int = 32768
    curriculum_phases: int = 3
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_v5e4_30b"
    checkpoint_every_steps: int = 500


@dataclass
class ConfigV5e8_1T:
    """
    Configuration for 1 Trillion parameter model on single TPU v5e-8 (16GB HBM)
    
    This is the ZENYX flagship configuration, demonstrating the full power
    of all four pillars working together.
    
    Model: 1T parameters (distributed across HBM/DRAM/NVMe)
    Hardware: 1x TPU v5e-8 (16GB HBM, 8 cores)
    Context: 1M tokens
    Batch size: 8
    
    Features:
    - Phase 7: Full Bélády KV cache tiering (1M context via HBM/DRAM/NVMe)
    - Phase 8: FP8 quantization (2x compression)
    - Phase 9: Dynamic curriculum (8K → 1M progressively)
    - Phase 10: Full sparse ring attention (13.3x speedup)
    
    This configuration pushes the limits of what's possible on single TPU.
    """
    
    # Model
    model_size_params: int = int(1e12)
    vocab_size: int = 128000
    hidden_dim: int = 12288
    num_layers: int = 80
    num_heads: int = 96
    max_seq_len: int = 1048576  # 1M
    
    # Hardware
    tpu_version: str = "v5e-8"
    num_tpu_pods: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Training
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 10000
    total_steps: int = 100000
    
    # ZENYX Features (ALL ENABLED)
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = True
    enable_phase10_sparse_attention: bool = True
    
    # Phase 7 Config
    use_nvm_tiering: bool = True  # Enable NVMe tiering for extreme context
    nvm_cache_dir: str = "/tmp/zenyx_1t_cache"
    
    # Phase 9 Config
    curriculum_start_context: int = 8000
    curriculum_max_context: int = 1048576
    curriculum_phases: int = 4
    curriculum_warmup_ratio: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_v5e8_1t"
    checkpoint_every_steps: int = 500
    save_metrics_every_steps: int = 100


# ============================================================================
# MULTI-POD CONFIGURATIONS
# ============================================================================

@dataclass
class ConfigMultiPodV5e8x2_4T:
    """
    Configuration for 4 Trillion parameter model on 2x TPU v5e-8 pods
    
    This demonstrates distributed training across multiple TPU pods.
    
    Model: 4T parameters
    Hardware: 2x TPU v5e-8 (32GB total HBM, 16 cores)
    Context: 1M tokens
    Batch size: 16 (per pod) → 32 global
    
    Features:
    - Ring All-Reduce for gradient synchronization
    - Distributed data loading
    - Phase 7/8/9/10 all enabled
    - Automatic load balancing
    """
    
    # Model
    model_size_params: int = int(4e12)
    vocab_size: int = 128000
    hidden_dim: int = 16384
    num_layers: int = 100
    num_heads: int = 128
    max_seq_len: int = 1048576  # 1M
    
    # Hardware
    tpu_version: str = "v5e-8"
    num_tpu_pods: int = 2
    batch_size: int = 16  # Per pod
    gradient_accumulation_steps: int = 2
    
    # Training
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 20000
    total_steps: int = 200000
    
    # ZENYX Features
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = True
    enable_phase10_sparse_attention: bool = True
    
    # Distributed Training
    use_ring_allreduce: bool = True
    distributed_data_loading: bool = True
    
    # Phase 9 Config
    curriculum_start_context: int = 8000
    curriculum_max_context: int = 1048576
    curriculum_phases: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_multipod_4t"
    checkpoint_every_steps: int = 1000


@dataclass
class ConfigMultiPodV5e8x4_8T:
    """
    Configuration for 8 Trillion parameter model on 4x TPU v5e-8 pods
    
    Large-scale distributed training across 4 TPU pods.
    
    Model: 8T parameters
    Hardware: 4x TPU v5e-8 (64GB total HBM, 32 cores)
    Context: 1M tokens
    Batch size: 32 (per pod) → 128 global
    """
    
    # Model
    model_size_params: int = int(8e12)
    vocab_size: int = 128000
    hidden_dim: int = 20480
    num_layers: int = 120
    num_heads: int = 160
    max_seq_len: int = 1048576  # 1M
    
    # Hardware
    tpu_version: str = "v5e-8"
    num_tpu_pods: int = 4
    batch_size: int = 32  # Per pod
    gradient_accumulation_steps: int = 1
    
    # Training
    learning_rate: float = 1.5e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 40000
    total_steps: int = 400000
    
    # ZENYX Features
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = True
    enable_phase10_sparse_attention: bool = True
    
    # Distributed Training
    use_ring_allreduce: bool = True
    distributed_data_loading: bool = True
    use_tensor_parallel: bool = True  # Shard large tensors
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_multipod_8t"
    checkpoint_every_steps: int = 1000


@dataclass
class ConfigMultiPodV5e8x8_16T:
    """
    Configuration for 16 Trillion parameter model on 8x TPU v5e-8 pods
    
    Cutting-edge large-scale training.
    
    Model: 16T parameters
    Hardware: 8x TPU v5e-8 (128GB total HBM, 64 cores)
    Context: 1M tokens
    Batch size: 64 (per pod) → 512 global
    """
    
    # Model
    model_size_params: int = int(16e12)
    vocab_size: int = 128000
    hidden_dim: int = 25600
    num_layers: int = 150
    num_heads: int = 200
    max_seq_len: int = 1048576  # 1M
    
    # Hardware
    tpu_version: str = "v5e-8"
    num_tpu_pods: int = 8
    batch_size: int = 64  # Per pod
    gradient_accumulation_steps: int = 1
    
    # Training
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 80000
    total_steps: int = 800000
    
    # ZENYX Features
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = True
    enable_phase10_sparse_attention: bool = True
    
    # Distributed Training
    use_ring_allreduce: bool = True
    distributed_data_loading: bool = True
    use_tensor_parallel: bool = True
    use_pipeline_parallel: bool = True  # Pipeline parallelism for 16T
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_multipod_16t"
    checkpoint_every_steps: int = 2000


# ============================================================================
# SPECIALIZED CONFIGURATIONS
# ============================================================================

@dataclass
class ConfigFinetune_V5e4_70B:
    """
    Configuration for fine-tuning a 70B model on single TPU v5e-4
    
    Optimized for fine-tuning (faster convergence, lower learning rate).
    Uses LoRA + quantization for memory efficiency.
    
    Model: 70B parameters (base) + LoRA adapters
    Hardware: 1x TPU v5e-4 (8GB HBM, 4 cores)
    Context: 32K tokens
    Batch size: 2
    """
    
    # Model
    model_size_params: int = int(70e9)  # Base model
    vocab_size: int = 128000
    hidden_dim: int = 8192
    num_layers: int = 80
    num_heads: int = 64
    max_seq_len: int = 32768
    
    # LoRA Config
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: float = 128.0
    
    # Hardware
    tpu_version: str = "v5e-4"
    num_tpu_pods: int = 1
    batch_size: int = 2
    
    # Training (fine-tuning parameters)
    learning_rate: float = 5e-5  # Lower LR for fine-tuning
    warmup_steps: int = 500
    total_steps: int = 5000  # Shorter
    
    # ZENYX Features
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = False
    enable_phase10_sparse_attention: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_finetune_70b"
    checkpoint_every_steps: int = 100


@dataclass
class ConfigPretraining_V5e8_500B_SuperLongContext:
    """
    Configuration for pre-training 500B model with 2M token context
    
    Demonstrates Phase 7's extreme context capabilities.
    
    Model: 500B parameters
    Hardware: 1x TPU v5e-8 (16GB HBM, 8 cores)
    Context: 2M tokens (via aggressive NVMe tiering)
    Batch size: 8
    """
    
    # Model
    model_size_params: int = int(500e9)
    vocab_size: int = 128000
    hidden_dim: int = 10240
    num_layers: int = 60
    num_heads: int = 80
    max_seq_len: int = 2097152  # 2M
    
    # Hardware
    tpu_version: str = "v5e-8"
    num_tpu_pods: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    
    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    total_steps: int = 100000
    
    # ZENYX Features (ALL ENABLED, especially Phase 7 & 9)
    enable_phase7_kv_tiering: bool = True
    enable_phase8_fp8_quant: bool = True
    enable_phase9_curriculum: bool = True
    enable_phase10_sparse_attention: bool = True
    
    # Phase 7: Extreme context setup
    use_nvm_tiering: bool = True
    nvm_cache_dir: str = "/ssd/zenyx_2m_cache"  # Use SSD if available
    
    # Phase 9: Aggressive curriculum
    curriculum_start_context: int = 4096
    curriculum_max_context: int = 2097152
    curriculum_phases: int = 6  # More phases for gradual increase
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints_500b_2m_context"
    checkpoint_every_steps: int = 500


# ============================================================================
# CONFIGURATION REGISTRY
# ============================================================================

CONFIGS: Dict[str, dict] = {
    # Single device
    "v5e1-7b": ConfigV5e1_7B().__dict__,
    "v5e4-30b": ConfigV5e4_30B().__dict__,
    "v5e8-1t": ConfigV5e8_1T().__dict__,
    
    # Multi-pod
    "multipod-v5e8x2-4t": ConfigMultiPodV5e8x2_4T().__dict__,
    "multipod-v5e8x4-8t": ConfigMultiPodV5e8x4_8T().__dict__,
    "multipod-v5e8x8-16t": ConfigMultiPodV5e8x8_16T().__dict__,
    
    # Specialized
    "finetune-v5e4-70b": ConfigFinetune_V5e4_70B().__dict__,
    "pretrain-v5e8-500b-2m": ConfigPretraining_V5e8_500B_SuperLongContext().__dict__,
}


def get_config(config_name: str) -> dict:
    """
    Get a configuration by name.
    
    Args:
        config_name: Configuration name (see CONFIGS keys)
    
    Returns:
        Configuration dictionary
    
    Raises:
        KeyError: If configuration not found
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise KeyError(f"Unknown config: {config_name}. Available: {available}")
    
    return CONFIGS[config_name]


def list_configs() -> None:
    """Print all available configurations."""
    print("=" * 80)
    print("ZENYX CONFIGURATION TEMPLATES")
    print("=" * 80)
    
    categories = {
        "Single Device": ["v5e1-7b", "v5e4-30b", "v5e8-1t"],
        "Multi-Pod": ["multipod-v5e8x2-4t", "multipod-v5e8x4-8t", "multipod-v5e8x8-16t"],
        "Specialized": ["finetune-v5e4-70b", "pretrain-v5e8-500b-2m"],
    }
    
    for category, configs in categories.items():
        print(f"\n{category}:")
        for config_name in configs:
            config = get_config(config_name)
            model_size = config.get('model_size_params', 0) / 1e9
            tpu = config.get('tpu_version', 'unknown')
            pods = config.get('num_tpu_pods', 1)
            
            if pods > 1:
                hw_desc = f"{pods}x {tpu}"
            else:
                hw_desc = tpu
            
            print(f"  • {config_name}")
            print(f"    Model: {model_size:.0f}B params")
            print(f"    Hardware: {hw_desc}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Export specific config
        config_name = sys.argv[1]
        config = get_config(config_name)
        print(json.dumps(config, indent=2, default=str))
    else:
        # List all configs
        list_configs()
