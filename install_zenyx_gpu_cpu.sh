#!/bin/bash
#
# ZENYX Complete Installation Script for GPU & CPU Training
# =========================================================
#
# Installs all dependencies for GPU (NVIDIA CUDA) and CPU training
# Detects hardware and provides platform-specific installation options
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INSTALL_TYPE="auto"
GPU_SUPPORT=true
CPU_SUPPORT=true
CUDA_VERSION="118"  # CUDA 11.8
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)

# Functions
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_python() {
    print_header "Python Verification"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    print_success "Python found: $(python3 --version)"
    
    # Check Python version
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8+ required (found $PYTHON_MAJOR.$PYTHON_MINOR)"
        exit 1
    fi
    echo ""
}

detect_hardware() {
    print_header "Hardware Detection"
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "NVIDIA GPU detected: $GPU_COUNT x $GPU_INFO"
        GPU_SUPPORT=true
    else
        print_warning "No NVIDIA GPU detected"
        GPU_SUPPORT=false
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc 2>/dev/null || echo "unknown")
    print_success "CPU cores available: $CPU_CORES"
    
    # Check RAM
    if command -v free &> /dev/null; then
        RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
        print_success "System RAM: ${RAM_GB}GB"
    fi
    
    echo ""
}

install_pytorch_gpu() {
    print_header "Installing PyTorch with GPU Support (CUDA $CUDA_VERSION)"
    
    print_info "Installing torch with CUDA $CUDA_VERSION support..."
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
    
    print_success "PyTorch GPU installation complete"
    
    # Verify CUDA
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" && print_success "CUDA verification passed" || print_warning "CUDA verification failed"
    echo ""
}

install_pytorch_cpu() {
    print_header "Installing PyTorch (CPU Only)"
    
    print_info "Installing torch for CPU..."
    pip install --upgrade torch torchvision torchaudio
    
    print_success "PyTorch CPU installation complete"
    echo ""
}

install_transformers() {
    print_header "Installing Hugging Face Transformers"
    
    pip install --upgrade transformers
    print_success "Transformers installed"
    echo ""
}

install_zenyx_dependencies() {
    print_header "Installing ZENYX Core Dependencies"
    
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Core dependencies
    PACKAGES=(
        "numpy>=1.24.0"
        "tqdm>=4.65.0"
        "pydantic>=2.0.0"
        "pyyaml>=6.0"
        "matplotlib>=3.7.0"
        "tensorboard>=2.12.0"
        "wandb>=0.14.0"
        "accelerate>=0.20.0"
        "bitsandbytes>=0.39.0"
        "tokenizers>=0.13.0"
        "datasets>=2.12.0"
    )
    
    for package in "${PACKAGES[@]}"; do
        pip install "$package" && print_success "Installed $package" || print_warning "Failed to install $package"
    done
    
    echo ""
}

install_optional_dependencies() {
    print_header "Installing Optional Dependencies"
    
    # Development tools
    print_info "Installing development tools..."
    pip install --upgrade \
        black \
        flake8 \
        isort \
        pytest \
        pytest-cov \
        pre-commit
    
    print_success "Development tools installed"
    echo ""
}

verify_installation() {
    print_header "Verifying Installation"
    
    python3 << 'EOF'
import sys
import importlib

packages = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "numpy": "NumPy",
    "tqdm": "tqdm",
    "pydantic": "Pydantic",
    "accelerate": "Hugging Face Accelerate",
}

failed = []
for package, name in packages.items():
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
    except ImportError:
        failed.append(name)
        print(f"✗ {name}: NOT INSTALLED")

if failed:
    print(f"\nFailed to import: {', '.join(failed)}")
    sys.exit(1)

# Additional checks
import torch
print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    print(f"✓ CUDA version: {torch.version.cuda}")

EOF
    
    echo ""
}

show_usage() {
    print_header "Installation Complete!"
    
    echo -e "\n${GREEN}GPU Training:${NC}"
    echo "  Single GPU:  python train/zenyx_distributed_gpu_training.py --gpu-type H100 --num-gpus 1"
    echo "  Multi-GPU:   torchrun --nproc_per_node=8 train/zenyx_distributed_gpu_training.py --gpu-type H100 --num-gpus 8"
    echo "  View config: python train/zenyx_gpu_config_templates.py"
    
    echo -e "\n${GREEN}CPU Training:${NC}"
    echo "  Single core: python train/zenyx_cpu_training.py --num-workers 1"
    echo "  Multi-core:  python train/zenyx_cpu_training.py --num-workers 8"
    echo "  View config: python train/zenyx_cpu_config_templates.py"
    
    echo -e "\n${GREEN}TPU Training:${NC}"
    echo "  Single TPU:  python train/zenyx_unified_tpu_training.py --tpu-version v5e-8"
    echo "  Multi-pod:   python train/zenyx_unified_tpu_training.py --num-tpu-pods 4"
    
    echo -e "\n${GREEN}Documentation:${NC}"
    echo "  GPU Guide:   ZENYX_GPU_TRAINING_GUIDE.md"
    echo "  CPU Guide:   ZENYX_CPU_TRAINING_GUIDE.md"
    echo "  TPU Guide:   ZENYX_UNIFIED_TRAINING_GUIDE.md"
    
    echo ""
}

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-only)
            CPU_SUPPORT=false
            shift
            ;;
        --cpu-only)
            GPU_SUPPORT=false
            shift
            ;;
        --cuda-version)
            CUDA_VERSION=$2
            shift 2
            ;;
        --skip-optional)
            SKIP_OPTIONAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu-only           Install GPU support only"
            echo "  --cpu-only           Install CPU support only"
            echo "  --cuda-version       CUDA version (118 for CUDA 11.8, 121 for 12.1, etc.)"
            echo "  --skip-optional      Skip optional dependencies"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main installation flow
print_header "ZENYX Complete Installation for GPU & CPU Training"

check_python
detect_hardware

# Determine what to install
if [ "$GPU_SUPPORT" = true ] && [ "$CPU_SUPPORT" = true ]; then
    print_info "Installing for both GPU and CPU training..."
    install_pytorch_gpu
    install_pytorch_cpu
elif [ "$GPU_SUPPORT" = true ]; then
    print_info "Installing for GPU training only..."
    install_pytorch_gpu
else
    print_info "Installing for CPU training only..."
    install_pytorch_cpu
fi

install_transformers
install_zenyx_dependencies

if [ "$SKIP_OPTIONAL" != true ]; then
    install_optional_dependencies
fi

verify_installation
show_usage

print_success "Installation complete! You're ready to train models with ZENYX."
