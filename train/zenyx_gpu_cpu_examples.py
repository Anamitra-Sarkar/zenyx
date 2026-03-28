"""
ZENYX GPU & CPU Training Quick Examples
========================================

Copy-paste ready examples for different training scenarios on GPU and CPU.
"""

import sys
from pathlib import Path

# Add train module to path
train_dir = Path(__file__).parent
sys.path.insert(0, str(train_dir))

try:
    from zenyx_gpu_config_templates import GPUConfigTemplates
    from zenyx_cpu_config_templates import CPUConfigTemplates
except ImportError:
    print("Error: Could not import configuration templates")
    print(f"sys.path: {sys.path}")
    print(f"train_dir: {train_dir}")
    sys.exit(1)


def print_gpu_examples():
    """Print GPU training examples"""
    print("\n" + "=" * 100)
    print("ZENYX GPU TRAINING EXAMPLES")
    print("=" * 100)

    examples = [
        {
            "title": "Example 1: Single H100 GPU with 7B Model",
            "command": "python train/zenyx_distributed_gpu_training.py --gpu-type H100 --num-gpus 1 --model-size 7e9 --batch-size 32",
            "description": "Basic single GPU training on H100 with 7 billion parameters"
        },
        {
            "title": "Example 2: Multi-GPU Training (8x H100 for 70B Model)",
            "command": "torchrun --nproc_per_node=8 train/zenyx_distributed_gpu_training.py --gpu-type H100 --num-gpus 8 --model-size 70e9 --batch-size 256 --enable-fp8 --enable-cache-tiering",
            "description": "Distributed training across 8 GPUs with FP8 quantization and cache tiering"
        },
        {
            "title": "Example 3: Multi-Node Training (4 nodes, 8 GPUs each)",
            "command": """torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 \\
    train/zenyx_distributed_gpu_training.py --gpu-type H100 --num-gpus 32 --model-size 140e9 \\
    --batch-size 512 --enable-curriculum --enable-sparse-attention""",
            "description": "Large-scale distributed training across 4 nodes (32 GPUs total)"
        },
        {
            "title": "Example 4: Fine-tuning on A100 (2x A100)",
            "command": "torchrun --nproc_per_node=2 train/zenyx_distributed_gpu_training.py --gpu-type A100 --num-gpus 2 --model-size 30e9 --batch-size 32 --learning-rate 3e-5 --max-steps 5000",
            "description": "Fine-tuning 30B model on 2 A100 GPUs with reduced learning rate"
        },
        {
            "title": "Example 5: RTX 4090 Consumer GPU with 7B Model",
            "command": "python train/zenyx_distributed_gpu_training.py --gpu-type RTX4090 --num-gpus 1 --model-size 7e9 --seq-length 2048 --batch-size 8 --gradient-accumulation-steps 16",
            "description": "Training on consumer-grade GPU with sequence length adjusted for memory"
        },
        {
            "title": "Example 6: All ZENYX Optimizations Enabled",
            "command": "torchrun --nproc_per_node=8 train/zenyx_distributed_gpu_training.py --gpu-type H100 --num-gpus 8 --model-size 70e9 --enable-cache-tiering --enable-fp8 --enable-curriculum --enable-sparse-attention --batch-size 512",
            "description": "Full optimization suite: KV cache tiering, FP8 quantization, curriculum learning, sparse attention"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{example['title']}")
        print("-" * 100)
        print(f"Description: {example['description']}")
        print(f"\nCommand:\n{example['command']}")
        if i < len(examples):
            print()

    print("\n" + "=" * 100)
    print("GPU Configuration Templates Available:")
    print("=" * 100)
    GPUConfigTemplates.print_all_templates()


def print_cpu_examples():
    """Print CPU training examples"""
    print("\n" + "=" * 100)
    print("ZENYX CPU TRAINING EXAMPLES")
    print("=" * 100)

    examples = [
        {
            "title": "Example 1: Single Core Training (200M Model)",
            "command": "python train/zenyx_cpu_training.py --num-workers 1 --model-size 200e6 --batch-size 8",
            "description": "Minimal CPU training on single core with 200 million parameters"
        },
        {
            "title": "Example 2: Multi-Core Training (8 cores, 500M Model)",
            "command": "python train/zenyx_cpu_training.py --num-workers 8 --num-threads-per-worker 2 --model-size 500e6 --batch-size 64 --gradient-accumulation-steps 2",
            "description": "Efficient multi-core training leveraging 8 CPU cores with 500 million parameters"
        },
        {
            "title": "Example 3: High Core Count System (16 cores, 500M Model)",
            "command": "python train/zenyx_cpu_training.py --num-workers 16 --model-size 500e6 --batch-size 256 --enable-curriculum",
            "description": "Training on high core count CPU (Xeon, EPYC) with curriculum learning"
        },
        {
            "title": "Example 4: 1B Model on 8 Cores (with optimizations)",
            "command": "python train/zenyx_cpu_training.py --num-workers 8 --model-size 1e9 --seq-length 2048 --batch-size 64 --enable-gradient-checkpointing --enable-curriculum",
            "description": "Training 1 billion parameter model with gradient checkpointing and curriculum"
        },
        {
            "title": "Example 5: Fine-tuning on Single Core",
            "command": "python train/zenyx_cpu_training.py --num-workers 1 --model-size 200e6 --batch-size 16 --learning-rate 5e-5 --max-steps 5000",
            "description": "Fine-tuning smaller model on single CPU core for quick iteration"
        },
        {
            "title": "Example 6: Long-Context Training (32K tokens, 8 cores)",
            "command": "python train/zenyx_cpu_training.py --num-workers 8 --model-size 200e6 --seq-length 32768 --batch-size 2 --gradient-accumulation-steps 8 --enable-cache-tiering --enable-sparse-attention",
            "description": "Long-context training with 32K tokens using cache tiering and sparse attention"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{example['title']}")
        print("-" * 100)
        print(f"Description: {example['description']}")
        print(f"\nCommand:\n{example['command']}")
        if i < len(examples):
            print()

    print("\n" + "=" * 100)
    print("CPU Configuration Templates Available:")
    print("=" * 100)
    CPUConfigTemplates.print_all_templates()


def print_combined_guide():
    """Print guide for choosing between GPU, CPU, and TPU"""
    print("\n" + "=" * 100)
    print("CHOOSING YOUR TRAINING PLATFORM")
    print("=" * 100)

    print("""
╔════════════════╦════════════════════════════╦════════════════════╦═══════════════════╗
║   Platform     ║      Best For               ║  Speed             ║  Cost             ║
╠════════════════╬════════════════════════════╬════════════════════╬═══════════════════╣
║ GPU (H100)     ║ • Large models (70B+)      ║ • 100-500x CPU    ║ • $1.50-2/hour   ║
║                ║ • Multi-node training      ║ • 10-50x faster   ║ • High upfront   ║
║                ║ • Production workloads     ║   vs A100         ║ • Most efficient ║
╠════════════════╬════════════════════════════╬════════════════════╬═══════════════════╣
║ GPU (A100)     ║ • Medium-large (7B-70B)    ║ • 50-200x CPU     ║ • $1-2/hour      ║
║                ║ • Cost-effective training  ║ • Industry std    ║ • Widely available║
║                ║ • Fine-tuning              ║                   ║ • Good value     ║
╠════════════════╬════════════════════════════╬════════════════════╬═══════════════════╣
║ GPU (RTX4090)  ║ • Research & development   ║ • 50-100x CPU     ║ • One-time cost  ║
║                ║ • Small-medium (1B-13B)    ║ • Consumer grade  ║ • $1.5k-2.5k     ║
║                ║ • Hobby/personal projects  ║                   ║ • Amortized cost ║
╠════════════════╬════════════════════════════╬════════════════════╬═══════════════════╣
║ TPU (v5e)      ║ • Extreme scale (1T+)      ║ • 500x-2000x CPU ║ • $2-4/hour      ║
║                ║ • Long context (1M tokens) ║ • Best attention ║ • Google Cloud   ║
║                ║ • Massive parallelism      ║   throughput     ║ • Scale unlimited║
╠════════════════╬════════════════════════════╬════════════════════╬═══════════════════╣
║ CPU            ║ • Learning & development   ║ • 1x baseline     ║ • Free (use own) ║
║                ║ • Small models (7B-200M)   ║ • Slow            ║ • No GPU needed  ║
║                ║ • Prototyping              ║ • High latency    ║ • High patience  ║
╚════════════════╩════════════════════════════╩════════════════════╩═══════════════════╝

QUICK DECISION GUIDE:
  • Have H100 GPU?         → GPU training (fastest, best for production)
  • Have A100 GPU?         → GPU training (good for large models)
  • Have RTX4090?          → GPU training (good for research)
  • Have TPU Pod?          → TPU training (best for scale)
  • Only have CPU?         → CPU training (learn fundamentals)
  • Need 1M token context? → TPU v5e-8 (only option that scales)
  • Training 1T+ params?   → Multi-pod TPU (distribute across pods)
""")

    print("=" * 100)


def print_hardware_comparison():
    """Print hardware specifications comparison"""
    print("\n" + "=" * 100)
    print("HARDWARE SPECIFICATIONS & RECOMMENDATIONS")
    print("=" * 100)

    specs = """
GPU SPECIFICATIONS:
┌─────────────────┬──────────┬──────────┬──────────────┬──────────────┐
│ GPU Type        │ Memory   │ FP32     │ BF16         │ Best For     │
├─────────────────┼──────────┼──────────┼──────────────┼──────────────┤
│ H100 80GB       │ 80GB     │ 1456 TF  │ 1456 TF      │ 140B params  │
│ H100 40GB       │ 40GB     │ 728 TF   │ 728 TF       │ 70B params   │
│ A100 80GB       │ 80GB     │ 312 TF   │ 312 TF       │ 70B params   │
│ A100 40GB       │ 40GB     │ 156 TF   │ 156 TF       │ 30B params   │
│ L40/L40S        │ 48GB     │ 362 TF   │ 362 TF       │ 30B params   │
│ RTX 4090        │ 24GB     │ 83 TF    │ 83 TF        │ 13B params   │
│ RTX 4080        │ 24GB     │ 49 TF    │ 49 TF        │ 7B params    │
└─────────────────┴──────────┴──────────┴──────────────┴──────────────┘

TPU SPECIFICATIONS:
┌──────────────────┬──────────┬──────────────┬──────────────────────┐
│ TPU Type         │ Memory   │ Peak FLOPS   │ Best For             │
├──────────────────┼──────────┼──────────────┼──────────────────────┤
│ TPU v5e-8        │ 16GB×8   │ 615 TFLOPS   │ 30B-70B (single pod) │
│ TPU v5e-4        │ 16GB×4   │ 307 TFLOPS   │ 13B (single device)  │
│ TPU v5p-8        │ 32GB×8   │ 1968 TFLOPS  │ 280B-1T (multi-pod)  │
│ TPU v4-8         │ 32GB×8   │ 1100 TFLOPS  │ 70B-140B (legacy)    │
└──────────────────┴──────────┴──────────────┴──────────────────────┘

CPU SPECIFICATIONS:
┌──────────────────┬──────────┬──────────┬────────────────┐
│ CPU Type         │ Cores    │ Memory   │ Best For       │
├──────────────────┼──────────┼──────────┼────────────────┤
│ Single Core      │ 1        │ Unlimited│ Learning       │
│ 4-8 Cores        │ 4-8      │ Unlimited│ Development    │
│ 16+ Cores (Xeon) │ 16-64    │ Unlimited│ 200M-1B models │
│ EPYC (96+ cores) │ 64-128   │ Unlimited│ 1B+ models     │
└──────────────────┴──────────┴──────────┴────────────────┘

RECOMMENDED CONFIGURATIONS:

For 7B Model:
  • Single H100:           32 batch, 4K context
  • Single A100:           16 batch, 4K context
  • Single RTX4090:        8 batch, 2K context
  • 8 CPU Cores:           32 batch, 2K context (slow)

For 30B Model:
  • 2x H100:               64 batch, 4K context
  • 2x A100:               32 batch, 4K context
  • 4x RTX4090:            32 batch, 2K context
  • TPU v5e-8:             128 batch, 8K context

For 70B Model:
  • 4x H100:               256 batch, 4K context
  • 8x A100:               256 batch, 4K context
  • TPU v5e-8:             256 batch, 8K context (multi-pod)

For 1T+ Model:
  • TPU v5p-8 (multi-pod): 512 batch, 16K context
  • 64x H100:              2048 batch, 8K context
"""
    print(specs)
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZENYX GPU/CPU Training Examples")
    parser.add_argument("--gpu", action="store_true", help="Show GPU examples")
    parser.add_argument("--cpu", action="store_true", help="Show CPU examples")
    parser.add_argument("--all", action="store_true", help="Show all examples")
    parser.add_argument("--choose-platform", action="store_true", help="Show platform comparison")
    parser.add_argument("--hardware", action="store_true", help="Show hardware specs")

    args = parser.parse_args()

    if args.gpu or args.all:
        print_gpu_examples()

    if args.cpu or args.all:
        print_cpu_examples()

    if args.choose_platform or args.all:
        print_combined_guide()

    if args.hardware or args.all:
        print_hardware_comparison()

    if not any([args.gpu, args.cpu, args.all, args.choose_platform, args.hardware]):
        print_gpu_examples()
        print_cpu_examples()
        print_combined_guide()
        print_hardware_comparison()
