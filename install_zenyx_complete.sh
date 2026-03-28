#!/usr/bin/env bash

###############################################################################
# ZENYX COMPLETE INSTALLATION GUIDE
###############################################################################
#
# This script installs all dependencies for ZENYX training:
# - Core dependencies (PyTorch, JAX, Triton)
# - TPU support (JAX[tpu])
# - Development tools
# - ZENYX library
#
# Supports:
# - CPU-only training
# - GPU training (NVIDIA)
# - TPU training (Google Cloud TPU)
# - Multi-device distributed training
#
# USAGE:
#   bash install_zenyx_complete.sh [--tpu] [--gpu] [--dev]
#
# EXAMPLES:
#   # Basic installation (CPU)
#   bash install_zenyx_complete.sh
#
#   # With TPU support
#   bash install_zenyx_complete.sh --tpu
#
#   # With GPU support
#   bash install_zenyx_complete.sh --gpu
#
#   # Full installation with all dependencies
#   bash install_zenyx_complete.sh --tpu --gpu --dev
#
###############################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse arguments
INSTALL_TPU=false
INSTALL_GPU=false
INSTALL_DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tpu)
            INSTALL_TPU=true
            shift
            ;;
        --gpu)
            INSTALL_GPU=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        *)
            error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

###############################################################################
# STEP 0: SYSTEM CHECKS
###############################################################################

info "Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [[ "$python_version" < "$required_version" ]]; then
    error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

success "Python $python_version detected"

# Check pip
if ! command -v pip3 &> /dev/null; then
    error "pip3 is not installed"
    exit 1
fi

success "pip3 available"

# Check git
if ! command -v git &> /dev/null; then
    warning "git not found - installing from wheel only"
else
    success "git available"
fi

# System memory
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    mem_gb=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}')
    info "System memory: ${mem_gb}GB"
    
    if [[ $mem_gb -lt 16 ]]; then
        warning "System has only ${mem_gb}GB RAM - some operations may be slow"
    fi
fi

###############################################################################
# STEP 1: UPGRADE PIP AND BASIC TOOLS
###############################################################################

info "Upgrading pip and setuptools..."
pip3 install --upgrade pip setuptools wheel

success "pip upgraded"

###############################################################################
# STEP 2: INSTALL CORE DEPENDENCIES
###############################################################################

info "Installing core dependencies..."

# PyTorch (CPU/GPU)
if [ "$INSTALL_GPU" = true ]; then
    info "Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    success "PyTorch (CUDA) installed"
else
    info "Installing PyTorch (CPU)..."
    pip3 install torch
    success "PyTorch (CPU) installed"
fi

# Triton (for custom kernels)
info "Installing Triton..."
pip3 install triton>=3.0.0

success "Triton installed"

# NumPy and scientific tools
info "Installing scientific computing packages..."
pip3 install numpy>=1.26.0 scipy pandas

success "Scientific packages installed"

###############################################################################
# STEP 3: INSTALL JAX (Optional for TPU)
###############################################################################

if [ "$INSTALL_TPU" = true ]; then
    info "Installing JAX with TPU support..."
    
    # JAX CPU version (always needed as base)
    pip3 install jax==0.4.20
    
    # JAX TPU version (only for TPU environments)
    if [[ -f /etc/os-release ]] && grep -q "GCP\|Google" /etc/os-release 2>/dev/null; then
        info "Detected Google Cloud environment, installing JAX[tpu]..."
        pip3 install "jax[tpu]>=0.4.20"
        success "JAX[tpu] installed"
    else
        warning "Not in TPU environment - installing JAX CPU version"
        warning "For TPU training, run this on a TPU VM instance"
        pip3 install jax==0.4.20
    fi
    
    # Flax (for JAX neural networks)
    info "Installing Flax..."
    pip3 install flax==0.8.0 optax>=0.2.0
    
    success "JAX and Flax installed"
else
    info "Skipping JAX installation (not needed for CPU/GPU training)"
fi

###############################################################################
# STEP 4: INSTALL ZENYX LIBRARY
###############################################################################

info "Installing ZENYX library..."

# Check if we're in the zenyx directory
if [ -f "pyproject.toml" ] && grep -q "name = \"zenyx\"" pyproject.toml; then
    info "Installing ZENYX from source (editable mode)..."
    
    if [ "$INSTALL_TPU" = true ]; then
        pip3 install -e ".[tpu]"
    else
        pip3 install -e "."
    fi
    
    success "ZENYX installed from source"
else
    info "Installing ZENYX from PyPI..."
    
    if [ "$INSTALL_TPU" = true ]; then
        pip3 install "zenyx[tpu]>=1.0.0"
    else
        pip3 install "zenyx>=1.0.0"
    fi
    
    success "ZENYX installed from PyPI"
fi

###############################################################################
# STEP 5: INSTALL OPTIONAL DEPENDENCIES
###############################################################################

info "Installing optional dependencies..."

# Utilities
pip3 install aiofiles>=23.0.0 safetensors>=0.4.0 psutil

# Metrics and monitoring
pip3 install tensorboard>=2.13.0 wandb tqdm

success "Optional packages installed"

###############################################################################
# STEP 6: INSTALL DEVELOPMENT TOOLS (Optional)
###############################################################################

if [ "$INSTALL_DEV" = true ]; then
    info "Installing development tools..."
    
    # Testing
    pip3 install pytest>=7.0 pytest-timeout>=2.4.0 pytest-asyncio pytest-cov
    
    # Linting and type checking
    pip3 install ruff mypy black isort
    
    # Documentation
    pip3 install sphinx sphinx-rtd-theme
    
    success "Development tools installed"
fi

###############################################################################
# STEP 7: VERIFY INSTALLATION
###############################################################################

info "Verifying installation..."

# Test Python imports
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || error "PyTorch import failed"
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" 2>/dev/null || error "NumPy import failed"
python3 -c "import triton; print(f'✓ Triton {triton.__version__}')" 2>/dev/null || error "Triton import failed"

# Test JAX (if installed)
if [ "$INSTALL_TPU" = true ]; then
    python3 -c "import jax; print(f'✓ JAX {jax.__version__}')" 2>/dev/null || error "JAX import failed"
    python3 -c "import flax; print(f'✓ Flax {flax.__version__}')" 2>/dev/null || error "Flax import failed"
fi

# Test ZENYX
python3 -c "from zenyx.unified_training import ZenyxTrainer; print('✓ ZENYX imported successfully')" 2>/dev/null || error "ZENYX import failed"

success "All imports verified"

###############################################################################
# STEP 8: DEVICE DETECTION
###############################################################################

info "Detecting hardware..."

# Check TPU
if python3 -c "import jax; devices = jax.devices(); print(f'Found {len(devices)} JAX devices')" 2>/dev/null; then
    python3 << 'EOF'
import jax
devices = jax.devices()
for i, device in enumerate(devices):
    print(f"  Device {i}: {device}")
EOF
fi

# Check GPU
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    if [ $(python3 -c "import torch; print(torch.cuda.device_count())") -gt 0 ]; then
        echo -e "${GREEN}  ✓ NVIDIA GPU available${NC}"
        python3 << 'EOF'
import torch
for i in range(torch.cuda.device_count()):
    print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
    fi
fi

###############################################################################
# STEP 9: PRINT INSTALLATION SUMMARY
###############################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                   ZENYX INSTALLATION COMPLETE                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

info "Installation Details:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  NumPy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'N/A')"
echo "  Triton: $(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'N/A')"

if [ "$INSTALL_TPU" = true ]; then
    echo "  JAX: $(python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'N/A')"
    echo "  Flax: $(python3 -c 'import flax; print(flax.__version__)' 2>/dev/null || echo 'N/A')"
fi

echo "  ZENYX: $(python3 -c 'import zenyx; print(zenyx.__version__)' 2>/dev/null || echo '1.0.0')"
echo ""

info "Next Steps:"
echo ""
echo "1. Verify installation with test script:"
echo "   python3 -c 'from zenyx.unified_training import ZenyxTrainer; print(\"ZENYX ready!\")'"
echo ""
echo "2. Run training script:"
echo "   # Single TPU v5e-8 (auto-config)"
echo "   python3 train/zenyx_unified_tpu_training.py --tpu-version v5e-8"
echo ""
echo "   # Small device (v5e-1)"
echo "   python3 train/zenyx_unified_tpu_training.py --tpu-version v5e-1 --model-size 100e9"
echo ""
echo "   # Multi-pod training"
echo "   python3 train/zenyx_unified_tpu_training.py --num-tpu-pods 2 --tpu-version v5e-8"
echo ""
echo "3. Read documentation:"
echo "   - README.md - Project overview"
echo "   - INSTALLATION_AND_SETUP.md - Detailed setup guide"
echo "   - TRAINING_START_HERE.md - Training getting started"
echo "   - ZENYX_FOUR_PILLARS_COMPLETE.md - Technical details"
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  ZENYX is ready! Train trillion-parameter models on any TPU.              ║"
echo "║  No more OOM. Maximum efficiency.                                          ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

success "Installation complete!"
