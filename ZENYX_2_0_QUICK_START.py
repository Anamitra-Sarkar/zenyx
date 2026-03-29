"""ZENYX 2.0 Phase 1 — Quick Start Examples

Demonstrates the new clean API for distributed training.
"""

# ============================================================================
# Example 1: Basic Single-GPU Training
# ============================================================================

import torch
import torch.nn as nn
from torch.optim import Adam

from zenyx.runtime import Scheduler
from zenyx.memory import ActivationManager

# Create a simple model
model = nn.Sequential(
    nn.Linear(768, 768),
    nn.ReLU(),
    nn.Linear(768, 768),
)

# Enable activation checkpointing for memory efficiency
ActivationManager.hook_into_model(model, use_checkpoint=True)

# Set up training
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = Scheduler(accumulation_steps=1)
criterion = nn.MSELoss()

# Training loop
batch = torch.randn(32, 100, 768)
labels = torch.randn(32, 100, 768)

output = scheduler.forward(model, batch)
loss = criterion(output, labels)
scheduler.backward(loss, optimizer)

print("✓ Single-GPU training works!")


# ============================================================================
# Example 2: Multi-GPU Training with FSDP
# ============================================================================

from zenyx.distributed import FSDPWrapper

# Wrap model for distributed training
fsdp_wrapper = FSDPWrapper(
    model=model,
    world_size=2,           # Assume 2 GPUs available
    rank=0,                 # Current process rank
    mixed_precision="fp16", # Use half precision
)
distributed_model = fsdp_wrapper.wrap()

# Training loop remains the same
output = scheduler.forward(distributed_model, batch)
loss = criterion(output, labels)
scheduler.backward(loss, optimizer)

print("✓ Multi-GPU training works!")


# ============================================================================
# Example 3: Analyzing Execution Graph
# ============================================================================

from zenyx.runtime import ExecutionGraphBuilder

builder = ExecutionGraphBuilder()
sample_input = torch.randn(1, 100, 768)

# Build execution graph by tracing forward pass
graph = builder.build_from_model(model, sample_input)
summary = graph.summarize()

print(f"Execution graph summary:")
print(f"  Forward operations: {summary['num_forward_ops']}")
print(f"  Total parameters: {summary['total_params']:,}")
print(f"  Estimated memory: {summary['total_memory_mb']:.1f} MB")


# ============================================================================
# Example 4: Memory Estimation & Offloading
# ============================================================================

from zenyx.compiler import OffloadManager, make_offload_policy

# Create an offload policy for an 80GB GPU
policy = make_offload_policy(gpu_memory_gb=80.0, batch_size=32)
print(f"Offload policy: {policy}")

# Use offload manager to conditionally move tensors
manager = OffloadManager(policy)
large_tensor = torch.randn(1000000, 1024)

# Check if tensor should be offloaded
tensor_location = manager.maybe_offload(large_tensor)
if tensor_location.device.type == "cpu":
    print("✓ Large tensor moved to CPU")
else:
    print("✓ Tensor remains on GPU")

# Later, move back to GPU when needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_on_device = manager.maybe_load(tensor_location, device)


# ============================================================================
# Example 5: Gradient Accumulation (Multi-GPU)
# ============================================================================

# Train with gradient accumulation (useful for large batches)
scheduler = Scheduler(accumulation_steps=4)

for micro_batch_idx in range(4):
    micro_batch = torch.randn(8, 100, 768)
    micro_labels = torch.randn(8, 100, 768)
    
    output = scheduler.forward(distributed_model, micro_batch)
    loss = criterion(output, micro_labels) / 4  # Scale loss
    scheduler.backward(loss, optimizer)
    
    print(f"Micro-batch {micro_batch_idx + 1}: accumulated")

print("✓ Optimizer step taken after 4 accumulation steps")


# ============================================================================
# Example 6: Custom Activation Checkpointing
# ============================================================================

from zenyx.memory import ActivationManager

manager = ActivationManager(use_checkpoint=True)

# Manually wrap specific layers
attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=12)
ffn_layer = nn.Linear(768, 3072)

# Wrap with checkpointing
wrapped_attn_forward = manager.checkpoint_attention(attention_layer)
wrapped_ffn_forward = manager.checkpoint_ffn(ffn_layer)

# Use in training
x = torch.randn(32, 100, 768)
attn_out = wrapped_attn_forward(x, x, x)  # (query, key, value)
ffn_out = wrapped_ffn_forward(attn_out)

print("✓ Custom checkpointing works!")


# ============================================================================
# Example 7: Memory Savings Estimation
# ============================================================================

savings = ActivationManager.estimate_memory_saving(
    num_layers=12,
    hidden_dim=768,
    seq_len=512,
    batch_size=32,
)

print(f"Memory savings from checkpointing:")
print(f"  Attention activations: {savings['attention_memory_mb']:.1f} MB")
print(f"  FFN activations: {savings['ffn_memory_mb']:.1f} MB")
print(f"  Total savings: {savings['total_memory_gb']:.2f} GB")


# ============================================================================
# Notes
# ============================================================================

"""
ZENYX 2.0 Phase 1 is designed for simplicity and correctness:

1. **Distributed Training**: Use FSDPWrapper for multi-GPU
   - Automatic parameter sharding
   - Mixed precision support (fp16, bf16)
   - Single-GPU compatible (no FSDP overhead)

2. **Memory Efficiency**: Use ActivationManager for checkpointing
   - Reduces peak memory usage
   - Safe (no quantization errors)
   - Configurable per-module

3. **Execution Analysis**: Use ExecutionGraphBuilder for planning
   - Trace forward pass
   - Extract operation metrics
   - Estimate memory and compute

4. **Offloading**: Use OffloadManager for CPU-GPU movement
   - Conservative thresholds
   - Optional (prefer recomputation)
   - Prepared for Phase 2+ optimization

5. **Scheduling**: Use Scheduler for training loops
   - Forward/backward management
   - Gradient accumulation
   - Synchronization points

All components work independently and can be mixed.
"""
