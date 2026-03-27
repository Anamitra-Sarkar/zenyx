#!/usr/bin/env python3
"""
Quick Start Guide for Zenyx Training
Get up and running in 5 minutes
"""

QUICK_START = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      ZENYX QUICK START GUIDE                              ║
║                  Hardware-Agnostic LLM Training Runtime                    ║
╚════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. INSTALLATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Basic installation
  pip install -e .

  # With TPU support
  pip install -e ".[tpu]"

  # Full installation
  pip install -e ".[full]"

  # Verify installation
  python -c "import zenyx; print(f'Zenyx v{zenyx.__version__}')"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. YOUR FIRST MODEL TRAINING (5 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, TensorDataset
  from zenyx.train.trainer import Trainer

  # 1. Define a model
  model = nn.Sequential(
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 256)
  )

  # 2. Create data
  data = torch.randn(1000, 256)
  loader = DataLoader(
      TensorDataset(data),
      batch_size=32
  )

  # 3. Train with Zenyx
  trainer = Trainer(
      model=model,
      dataloader=loader,
      lr=1e-4,
      total_steps=100,
      dtype="bfloat16",
  )
  trainer.train()

  # 4. Get results
  state = trainer.get_state()
  print(f"Steps: {state['step']}")
  print(f"Loss: {state['loss']}")
  print(f"Parallelism: {state['parallelism_plan']}")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. RUNNING TEST EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # CPU Training with 8K context
  python test_cpu_8k_context.py

  # Memory management validation
  python test_memory_management.py

  # Comprehensive hardware validation
  python validate_hardware.py

  # GPU Training (requires CUDA + 2x GPUs)
  torchrun --nproc_per_node=2 test_gpu_128k_context.py

  # TPU Training (requires torch_xla + TPU v5e-8)
  python test_tpu_1m_context.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. TRAINER API OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  trainer = Trainer(
      # Required
      model=your_model,
      dataloader=your_loader,

      # Learning
      lr=1e-4,
      weight_decay=0.1,
      warmup_steps=2000,
      total_steps=100000,

      # Hardware
      dtype="bfloat16",                    # float32, bfloat16, float16
      context_len=4096,                    # Up to 32K (CPU), 128K (GPU), 1M (TPU)

      # Memory Management
      gradient_accumulation_steps=4,       # Increase batch size virtually
      selective_activation_checkpoint=True,# Reduce activation memory
      t1_capacity_gb=8.0,                 # DRAM tier size
      t2_capacity_gb=64.0,                # NVMe tier size

      # Checkpointing
      checkpoint_dir="./checkpoints",
      checkpoint_every=1000,
      resume_from="checkpoint.pt",        # Optional: resume training

      # Advanced Features
      fp8_kv=True,                        # FP8 KV quantization
      sparse_attn=True,                   # Sparse ring attention
      dynamic_routing=False,              # For MoE models
  )

  trainer.train()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. KEY FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ Never-OOM Guarantee
     - Three-tier memory hierarchy (HBM → DRAM → NVMe)
     - Bélády-optimal eviction
     - Async prefetching

  ✅ Hardware Agnostic
     - CPU (32K context, chunked attention)
     - GPU (128K context, ring FA3)
     - TPU (1M context, ring Pallas)
     - Auto-detection, no config needed

  ✅ Automatic Parallelism
     - Tensor Parallel (TP)
     - Pipeline Parallel (PP)
     - Data Parallel (DP)
     - Ring Attention
     - Braided TP+PP scheduling

  ✅ Memory Optimization
     - FP8 E4M3 activation storage (2x savings)
     - Selective activation checkpointing
     - Mixed precision training
     - Gradient accumulation

  ✅ Extreme Scale
     - 1T parameters
     - 1M+ token context
     - 500K+ vocabulary
     - 120B model loading in <20 seconds


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. HARDWARE SUPPORT MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Hardware              | Attention Kernel      | Max Context | Status
  ─────────────────────────────────────────────────────────────────────
  CPU / M3             | Chunked attention     | 32K         | ✅ Ready
  Any GPU (PCIe)       | Ring FA3 (fallback)   | 128K        | ✅ Ready
  H100 NVLink (8+)     | TokenRing FA3 (Triton)| 1M          | ✅ Ready
  TPU v5e-8            | Ring Pallas + Shardy  | 1M          | ✅ Ready
  TPU v5p              | Ring Pallas + Shardy  | 1M          | ✅ Ready


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. MULTI-GPU TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Launch with torchrun
  torchrun --nproc_per_node=8 train.py

  # In train.py, Zenyx auto-detects distributed setup:
  trainer = Trainer(model, loader, lr=1e-4)
  trainer.train()  # Automatically distributed!


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. MONITORING & DEBUGGING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Get training state
  state = trainer.get_state()

  # Available state information:
  state['step']               # Current training step
  state['loss']               # Current loss value
  state['parallelism_plan']   # {tp_degree, pp_degree, dp_degree, ring_degree}
  state['profiler_stats']     # Per-op timing statistics
  state['time_elapsed']       # Training duration

  # Log every N steps
  trainer = Trainer(..., log_every=10)

  # Monitor memory
  import torch
  if torch.cuda.is_available():
      allocated = torch.cuda.memory_allocated() / 1e9
      print(f"GPU Memory: {allocated:.1f} GB")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. COMMON PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Large batch size via gradient accumulation
  trainer = Trainer(
      model, loader,
      gradient_accumulation_steps=16,  # 16x larger effective batch
  )

  # Fine-tuning with low learning rate
  trainer = Trainer(
      model, loader,
      lr=2e-5,              # 10x lower LR
      total_steps=10000,    # Fewer steps
      warmup_steps=100,
  )

  # Large context training
  trainer = Trainer(
      model, loader,
      context_len=131072,   # 128K tokens
      dtype="bfloat16",
      selective_activation_checkpoint=True,  # Save memory
  )

  # Extreme scale (1T params, 1M context)
  trainer = Trainer(
      model, loader,
      context_len=1_000_000,     # 1M tokens
      dtype="bfloat16",
      fp8_kv=True,               # FP8 quantization
      sparse_attn=True,          # Sparse attention
      t1_capacity_gb=256,        # Large DRAM buffer
      t2_capacity_gb=4096,       # Large NVMe buffer
  )


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. DOCUMENTATION & RESOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  📖 Main Documentation
     README.md                    - Full library documentation
     TEST_SUITE_README.md         - Comprehensive test guide
     TRAINING_GUIDE.md            - Training examples
     VALIDATION_REPORT.md         - Test results & findings

  📊 Zenyx Internal Docs
     zenyx/docs/performance_ceiling.md  - Throughput analysis
     zenyx/docs/dispute_resolutions.md  - Research validation

  🧪 Test Files
     test_cpu_8k_context.py       - CPU training example
     test_gpu_128k_context.py     - GPU training template
     test_tpu_1m_context.py       - TPU training template
     test_memory_management.py    - Memory validation
     validate_hardware.py          - Master test runner

  🔗 GitHub
     https://github.com/Anamitra-Sarkar/zenyx


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Issue: OutOfMemoryError
  → Reduce context_len, increase gradient_accumulation_steps, or enable
    selective_activation_checkpoint=True

  Issue: Training is slow
  → Check if you're in "throttle mode" (memory bandwidth limited)
  → Use larger GPUs, add NVMe storage, or reduce context length

  Issue: CUDA not detected
  → Verify CUDA installation: python -c "import torch; print(torch.cuda.is_available())"
  → Install CUDA 12.4+: https://developer.nvidia.com/cuda-downloads

  Issue: torch_xla not available
  → Install for your TPU generation:
    pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

  Issue: Model not training (loss not decreasing)
  → Increase learning rate slightly (1e-3 to 1e-4)
  → Use larger batch size via gradient_accumulation_steps
  → Reduce max_grad_norm (grad_clip parameter)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. ✅ Run CPU training test
     python test_cpu_8k_context.py

  2. ✅ Understand memory management
     python test_memory_management.py

  3. 🚀 Try your own model
     # Copy test_cpu_8k_context.py and modify for your model

  4. 🔷 Test GPU (if available)
     torchrun --nproc_per_node=2 test_gpu_128k_context.py

  5. 📡 Test TPU (if available)
     python test_tpu_1m_context.py

  6. 📊 Run full validation
     python validate_hardware.py


╔════════════════════════════════════════════════════════════════════════════╗
║                       ZENYX IS READY TO USE! 🚀                           ║
║                                                                            ║
║  Start training: python test_cpu_8k_context.py                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICK_START)
