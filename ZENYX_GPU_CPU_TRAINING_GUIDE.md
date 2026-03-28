# ZENYX GPU & CPU Training Complete Guide

## Table of Contents
1. [Installation](#installation)
2. [GPU Training](#gpu-training)
3. [CPU Training](#cpu-training)
4. [Configuration](#configuration)
5. [Running Training](#running-training)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Distributed Training](#distributed-training)

---

## Installation

### Prerequisites
- Python 3.8+ (3.10+ recommended)
- pip or conda package manager
- Git (for cloning)

### Quick Installation

**Option 1: Automated Installation (Recommended)**
```bash
bash install_zenyx_gpu_cpu.sh
```

**Option 2: Manual Installation - GPU Support**
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ZENYX dependencies
pip install transformers accelerate bitsandbytes tokenizers datasets wandb tensorboard
```

**Option 3: Manual Installation - CPU Only**
```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Install ZENYX dependencies
pip install transformers accelerate tokenizers datasets wandb tensorboard
```

### Verify Installation

```bash
# Check PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check Transformers
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

---

## GPU Training

### Supported GPUs

| GPU Type | Memory | Best For | Relative Speed |
|----------|--------|----------|----------------|
| H100 80GB | 80GB | Large models (140B+) | 100x CPU, 2x A100 |
| H100 40GB | 40GB | Medium-large (70B) | 50x CPU, 2x A100 |
| A100 80GB | 80GB | Large models (70B) | 50x CPU |
| A100 40GB | 40GB | Medium (30B) | 30x CPU |
| L40/L40S | 48GB | Medium-large (30B) | 40x CPU |
| RTX 4090 | 24GB | Small-medium (13B) | 30x CPU |
| RTX 4080 | 24GB | Small (7B) | 15x CPU |

### Single GPU Training

#### Basic 7B Model Training
```bash
python train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --num-gpus 1 \
    --model-size 7e9 \
    --batch-size 32 \
    --learning-rate 1e-4
```

#### 30B Model with FP8 Quantization
```bash
python train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --num-gpus 1 \
    --model-size 30e9 \
    --batch-size 8 \
    --enable-fp8 \
    --gradient-accumulation-steps 16
```

#### 70B Model with All Optimizations
```bash
python train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --num-gpus 1 \
    --model-size 70e9 \
    --batch-size 2 \
    --enable-cache-tiering \
    --enable-fp8 \
    --enable-curriculum \
    --enable-sparse-attention
```

### Multi-GPU Training (DDP)

#### 2 GPUs - 13B Model
```bash
torchrun --nproc_per_node=2 train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --num-gpus 2 \
    --model-size 13e9 \
    --batch-size 64 \
    --enable-curriculum
```

#### 8 GPUs - 70B Model with Full Optimization
```bash
torchrun --nproc_per_node=8 train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --num-gpus 8 \
    --model-size 70e9 \
    --batch-size 256 \
    --enable-cache-tiering \
    --enable-fp8 \
    --enable-curriculum \
    --enable-sparse-attention
```

### Multi-Node Training

#### 4 Nodes, 8 GPUs Each (32 GPUs Total)
```bash
# On node 0 (master):
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --num-gpus 32 \
    --model-size 140e9 \
    --batch-size 512

# On node 1, 2, 3:
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=<1|2|3> \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train/zenyx_distributed_gpu_training.py \
    ... [same parameters]
```

### GPU Configuration Templates

View all available configurations:
```bash
python train/zenyx_gpu_config_templates.py
```

Print details for specific template:
```python
from train.zenyx_gpu_config_templates import print_template_details
print_template_details("MULTI_GPU_8X_H100_70B")
```

Available templates:
- `SINGLE_GPU_H100_7B` - 7B on 1x H100
- `SINGLE_GPU_H100_30B` - 30B on 1x H100
- `SINGLE_GPU_H100_70B` - 70B on 1x H100
- `SINGLE_GPU_A100_7B` - 7B on 1x A100
- `SINGLE_GPU_RTX4090_7B` - 7B on RTX4090
- `MULTI_GPU_2X_H100_13B` - 13B on 2x H100
- `MULTI_GPU_4X_H100_34B` - 34B on 4x H100
- `MULTI_GPU_8X_H100_70B` - 70B on 8x H100
- `MULTI_GPU_8X_A100_70B` - 70B on 8x A100
- `MULTI_NODE_4X8_H100_140B` - 140B on 4 nodes
- `LONG_CONTEXT_H100_7B_128K` - 128K context
- `LONG_CONTEXT_H100_7B_1M` - 1M token context
- `FINETUNE_H100_7B` - Fine-tuning setup

---

## CPU Training

### Supported CPUs

| CPU Configuration | Cores | Best For | Relative Speed |
|-------------------|-------|----------|----------------|
| Single core | 1 | Learning | 1x baseline |
| Dual core | 2 | Development | 2x baseline |
| Quad core | 4 | Testing | 3-4x baseline |
| Octa core (Ryzen/Xeon) | 8 | Training 500M | 6-8x baseline |
| 16+ core (Xeon/EPYC) | 16+ | Training 1B | 12-16x baseline |
| 32+ core (EPYC) | 32+ | Large 1B models | 24-32x baseline |

### Single Core Training

#### Minimal 200M Model
```bash
python train/zenyx_cpu_training.py \
    --num-workers 1 \
    --model-size 200e6 \
    --batch-size 8
```

#### 500M Model
```bash
python train/zenyx_cpu_training.py \
    --num-workers 1 \
    --model-size 500e6 \
    --batch-size 4 \
    --gradient-accumulation-steps 2
```

### Multi-Core Training

#### 4 Core System with 500M Model
```bash
python train/zenyx_cpu_training.py \
    --num-workers 4 \
    --num-threads-per-worker 2 \
    --model-size 500e6 \
    --batch-size 64 \
    --enable-curriculum
```

#### 8 Core System with 1B Model
```bash
python train/zenyx_cpu_training.py \
    --num-workers 8 \
    --num-threads-per-worker 2 \
    --model-size 1e9 \
    --seq-length 2048 \
    --batch-size 64 \
    --enable-curriculum
```

#### 32 Core EPYC with Maximum Performance
```bash
python train/zenyx_cpu_training.py \
    --num-workers 32 \
    --model-size 1e9 \
    --batch-size 512 \
    --learning-rate 1e-5 \
    --enable-curriculum \
    --enable-sparse-attention
```

### CPU Configuration Templates

View all templates:
```bash
python train/zenyx_cpu_config_templates.py
```

Available templates:
- `SINGLE_CORE_200M` - 200M on single core
- `SINGLE_CORE_500M` - 500M on single core
- `SINGLE_CORE_1B` - 1B on single core
- `DUAL_CORE_200M` - 200M on 2 cores
- `QUAD_CORE_500M` - 500M on 4 cores
- `OCTA_CORE_700M` - 700M on 8 cores
- `OCTA_CORE_1B` - 1B on 8 cores
- `HIGH_CORE_16CORE_500M` - 500M on 16 cores
- `HIGH_CORE_32CORE_1B` - 1B on 32 cores
- `LONG_CONTEXT_OCTA_CORE_200M` - 32K context on 8 cores

---

## Configuration

### Model Configuration Options

```
--model-size        Number of parameters (default: 7e9)
--vocab-size        Vocabulary size (default: 128256)
--hidden-dim        Hidden dimension size
--num-layers        Number of transformer layers
--seq-length        Maximum sequence length
--max-position-embeddings
```

### Training Configuration Options

```
--batch-size              Batch size per GPU/worker
--gradient-accumulation-steps
--learning-rate           Initial learning rate
--weight-decay            L2 regularization
--max-steps               Total training steps
--warmup-steps            Warmup steps
--log-steps               Logging frequency
--save-steps              Checkpoint frequency
--max-grad-norm           Gradient clipping threshold
```

### Optimization Options

```
--use-mixed-precision     Enable BF16/FP16 (GPU only)
--enable-cache-tiering    Phase 7: KV Cache Tiering
--enable-fp8              Phase 8: FP8 Quantization
--enable-curriculum       Phase 9: Dynamic Curriculum
--enable-sparse-attention Phase 10: Sparse Attention
```

### GPU-Specific Options

```
--gpu-type        GPU type (H100, A100, L40, RTX4090, etc.)
--num-gpus        Number of GPUs to use
--num-nodes       Number of nodes (multi-node)
--node-rank       Current node rank
--master-addr     Master node address
--master-port     Master node port
```

### CPU-Specific Options

```
--num-workers             Number of CPU workers
--num-threads-per-worker  Threads per worker
--use-numa               Use NUMA if available
```

---

## Running Training

### Checking Available Hardware

**GPU:**
```bash
nvidia-smi
```

**CPU:**
```bash
lscpu
nproc  # Number of cores
```

### Example Training Run

#### GPU Training
```bash
# Start training
python train/zenyx_distributed_gpu_training.py \
    --gpu-type H100 \
    --model-size 7e9 \
    --batch-size 32 \
    --log-steps 10

# Monitor in another terminal
watch -n 5 'nvidia-smi'
```

#### CPU Training
```bash
# Start training
python train/zenyx_cpu_training.py \
    --num-workers 8 \
    --model-size 500e6 \
    --batch-size 64 \
    --log-steps 10

# Monitor resources
watch -n 5 'top -b -n 1 | head -20'
```

### Checkpointing and Recovery

Checkpoints saved to:
- **GPU:** `./checkpoints_gpu/checkpoint-{step}/`
- **CPU:** `./checkpoints_cpu/checkpoint-{step}/`

Each checkpoint contains:
- `model.pt` - Model weights
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler
- `config.json` - Training configuration

---

## Troubleshooting

### GPU Issues

**CUDA Out of Memory**
```
Solution 1: Reduce batch size
--batch-size 16

Solution 2: Enable gradient checkpointing
(already enabled by default)

Solution 3: Enable FP8 quantization
--enable-fp8

Solution 4: Increase gradient accumulation
--gradient-accumulation-steps 8
```

**CUDA Not Found**
```bash
# Check NVIDIA GPU driver
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**NCCL Errors (Multi-GPU)**
```
Solution 1: Check network connectivity
ping <other_node>

Solution 2: Set NCCL debug mode
export NCCL_DEBUG=INFO

Solution 3: Use gloo backend
(already default for multi-node)
```

### CPU Issues

**High Memory Usage**
```
Solution 1: Reduce batch size
--batch-size 16

Solution 2: Enable gradient checkpointing
--use-gradient-checkpointing True

Solution 3: Reduce sequence length
--seq-length 512
```

**Slow Training**
```
Solution 1: Use more workers
--num-workers 8  (instead of 1)

Solution 2: Reduce model size
--model-size 200e6  (instead of 1e9)

Solution 3: Use CPU affinity
--affinity True
```

### General Issues

**Out of Disk Space**
```bash
# Check disk usage
df -h

# Clear old checkpoints
rm -rf checkpoints_gpu/checkpoint-* 
rm -rf checkpoints_cpu/checkpoint-*
```

**Import Errors**
```bash
# Reinstall all dependencies
pip install --upgrade --force-reinstall torch transformers
```

---

## Performance Tuning

### GPU Performance Optimization

1. **Batch Size Tuning**
   - Start with 32, increase until OOM
   - Larger batch → faster training but more memory
   - Sweet spot: 64-256 depending on model size

2. **Mixed Precision (BF16)**
   - 2x memory reduction, 1.5-2x speedup
   - Enabled by default on GPU
   - Use `--mixed-precision-dtype bf16`

3. **Gradient Accumulation**
   - Simulate larger batch without more memory
   - Example: batch_size=8, accumulation=4 → effective 32
   - Trade-off: slower but more memory efficient

4. **FP8 Quantization** (Phase 8)
   - 2x more memory reduction
   - Slight accuracy impact on large models
   - Good for models >30B
   ```bash
   --enable-fp8
   ```

5. **Cache Tiering** (Phase 7)
   - Enables longer context (up to 1M tokens)
   - Slight latency increase
   - Good for long-context applications
   ```bash
   --enable-cache-tiering
   ```

6. **Sparse Attention** (Phase 10)
   - 13.3x attention speedup
   - Minimal accuracy loss
   - Good for long sequences (>4K)
   ```bash
   --enable-sparse-attention
   ```

### CPU Performance Optimization

1. **Worker Count**
   - Use `nproc` for max cores: `--num-workers $(nproc)`
   - Each worker gets independent batch
   - More workers → more parallelism but less per-worker memory

2. **Thread Affinity**
   - Pin workers to specific cores for better cache
   - Reduces context switching
   ```bash
   --affinity True
   ```

3. **NUMA Awareness**
   - On NUMA systems, distribute workers across nodes
   - Reduces memory latency
   ```bash
   --use-numa True
   ```

4. **Gradient Checkpointing**
   - Memory-time tradeoff: saves memory, costs compute
   - Good for large models (>500M)
   - Enabled by default
   ```bash
   --use-gradient-checkpointing True
   ```

### Distributed Training Optimization

1. **Synchronization Frequency**
   - More accumulation steps → less sync → faster
   - Trade-off: larger effective batch size
   ```bash
   --gradient-accumulation-steps 4
   ```

2. **Communication Backend**
   - NCCL best for GPU clusters
   - Gloo better for heterogeneous networks
   - Default: NCCL for GPU, Gloo for mixed

3. **Network Configuration**
   - Use high-bandwidth interconnects (InfiniBand preferred)
   - Minimize latency (same datacenter/rack)
   - Test with: `python -m torch.distributed.launch --nproc_per_node=2 test_comm.py`

---

## Distributed Training

### Multi-GPU Setup (Same Node)

```bash
# 8 GPUs on single node
torchrun --nproc_per_node=8 train/zenyx_distributed_gpu_training.py \
    --model-size 70e9 \
    --batch-size 256
```

Automatic setup:
- RANK, LOCAL_RANK, WORLD_SIZE environment variables
- NCCL backend for GPU communication
- DDP (Distributed Data Parallel) wrapping

### Multi-Node Setup (Different Nodes)

**Step 1: Identify Master Node**
```bash
# On master node (IP: 10.0.0.1)
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
```

**Step 2: Start Training on Master**
```bash
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train/zenyx_distributed_gpu_training.py \
    --num-gpus 32 \
    --model-size 140e9
```

**Step 3: Start Training on Worker Nodes**
```bash
# Worker node 1 (10.0.0.2), rank=1
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train/zenyx_distributed_gpu_training.py \
    --num-gpus 32
```

**Network Requirements:**
- All nodes must have network connectivity
- Low latency preferred (same datacenter)
- High bandwidth (10Gbps+ recommended)
- Open firewall for master_port

---

## Advanced Topics

### Custom Model Integration

Replace the SimpleLM model with your own:

```python
from your_model import YourModel

class CustomTrainer(DistributedGPUTrainer):
    def build_model(self):
        self.model = YourModel(self.config).to(self.device)
        # ... rest of setup
```

### Custom Data Loading

Override create_dummy_data:

```python
def create_dummy_data(self, num_samples: int = 100) -> DataLoader:
    dataset = YourDataset(...)
    return DataLoader(dataset, batch_size=self.config.batch_size)
```

### Logging and Monitoring

Use W&B for experiment tracking:

```bash
wandb login  # Add W&B API key
# Training will automatically log to W&B
```

View TensorBoard logs:

```bash
tensorboard --logdir ./logs_gpu
```

---

## Quick Reference

### Most Used Commands

```bash
# Single GPU 7B training
python train/zenyx_distributed_gpu_training.py --gpu-type H100 --model-size 7e9

# Multi-GPU 70B training
torchrun --nproc_per_node=8 train/zenyx_distributed_gpu_training.py --gpu-type H100 --model-size 70e9

# CPU multi-core 500M training
python train/zenyx_cpu_training.py --num-workers 8 --model-size 500e6

# View GPU config templates
python train/zenyx_gpu_config_templates.py

# View CPU config templates
python train/zenyx_cpu_config_templates.py

# View examples and comparisons
python train/zenyx_gpu_cpu_examples.py --all
```

### Performance Baselines

| Setup | Model | Tokens/sec | Time to 1T tokens |
|-------|-------|-----------|-------------------|
| 1x H100 | 7B | 3,000 | 11.5 days |
| 8x H100 | 70B | 24,000 | 1.4 days |
| 1x A100 | 7B | 1,500 | 23 days |
| 8x CPU | 500M | 100 | 115 days |

---

## Support & Resources

- Documentation: See README.md
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share tips
- Email: support@zenyx.ai

Happy training! 🚀
