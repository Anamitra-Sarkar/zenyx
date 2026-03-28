#!/usr/bin/env bash

###############################################################################
# ZENYX QUICK START GUIDE
###############################################################################
#
# This script provides a quick start guide for ZENYX training.
# It covers:
# 1. Installation
# 2. Configuration selection
# 3. Running your first training
# 4. Monitoring and checkpoints
#
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BOLD}${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# STEP 1: INSTALLATION
# ============================================================================

print_header "STEP 1: INSTALLATION"

echo "Choose your setup:"
echo ""
echo "1. CPU/GPU only (no TPU)"
echo "2. TPU support included"
echo "3. Full installation (TPU + GPU + dev tools)"
echo ""
read -p "Enter choice (1-3): " install_choice

case $install_choice in
    1)
        print_step "Installing for CPU/GPU..."
        bash install_zenyx_complete.sh --gpu
        ;;
    2)
        print_step "Installing with TPU support..."
        bash install_zenyx_complete.sh --tpu
        ;;
    3)
        print_step "Installing full package..."
        bash install_zenyx_complete.sh --tpu --gpu --dev
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

print_step "Installation complete!"

# ============================================================================
# STEP 2: VERIFY INSTALLATION
# ============================================================================

print_header "STEP 2: VERIFYING INSTALLATION"

python3 << 'EOF'
import sys

print("Checking imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import zenyx
    print("✓ ZENYX library")
except ImportError as e:
    print(f"✗ ZENYX: {e}")
    sys.exit(1)

try:
    import jax
    print(f"✓ JAX {jax.__version__}")
except ImportError:
    print("⚠ JAX not available (OK for CPU/GPU training)")

print("\nAll required packages available!")
EOF

if [ $? -eq 0 ]; then
    print_step "Verification passed!"
else
    print_warning "Verification failed - reinstall or check errors"
    exit 1
fi

# ============================================================================
# STEP 3: SELECT CONFIGURATION
# ============================================================================

print_header "STEP 3: SELECT CONFIGURATION"

echo "Available configurations:"
echo ""
echo "Single Device:"
echo "  1. v5e1-7b      - 7B model on TPU v5e-1 (2GB)"
echo "  2. v5e4-30b     - 30B model on TPU v5e-4 (8GB)"
echo "  3. v5e8-1t      - 1T model on TPU v5e-8 (16GB) [DEFAULT]"
echo ""
echo "Multi-Pod:"
echo "  4. multipod-v5e8x2-4t   - 4T on 2x TPU v5e-8"
echo "  5. multipod-v5e8x4-8t   - 8T on 4x TPU v5e-8"
echo "  6. multipod-v5e8x8-16t  - 16T on 8x TPU v5e-8"
echo ""
echo "Specialized:"
echo "  7. finetune-v5e4-70b           - Fine-tune 70B on TPU v5e-4"
echo "  8. pretrain-v5e8-500b-2m       - 500B with 2M context"
echo ""

read -p "Enter choice (1-8, default 3): " config_choice
config_choice=${config_choice:-3}

case $config_choice in
    1) CONFIG="v5e1-7b" ;;
    2) CONFIG="v5e4-30b" ;;
    3) CONFIG="v5e8-1t" ;;
    4) CONFIG="multipod-v5e8x2-4t" ;;
    5) CONFIG="multipod-v5e8x4-8t" ;;
    6) CONFIG="multipod-v5e8x8-16t" ;;
    7) CONFIG="finetune-v5e4-70b" ;;
    8) CONFIG="pretrain-v5e8-500b-2m" ;;
    *) CONFIG="v5e8-1t" ;;
esac

print_step "Selected configuration: $CONFIG"

# ============================================================================
# STEP 4: SHOW CONFIGURATION DETAILS
# ============================================================================

print_header "STEP 4: CONFIGURATION DETAILS"

python3 << EOF
from train.zenyx_config_templates import get_config
import json

config = get_config("$CONFIG")

print("Configuration:")
print(json.dumps({
    'model_params': f"{config['model_size_params']/1e9:.0f}B",
    'tpu_version': config['tpu_version'],
    'num_pods': config['num_tpu_pods'],
    'batch_size': config['batch_size'],
    'learning_rate': config['learning_rate'],
    'total_steps': config['total_steps'],
    'max_seq_len': f"{config['max_seq_len']:,}",
}, indent=2, default=str))

print("\nZENYX Phases Enabled:")
if config.get('enable_phase7_kv_tiering'):
    print("  ✓ Phase 7: Bélády KV Cache Tiering")
if config.get('enable_phase8_fp8_quant'):
    print("  ✓ Phase 8: FP8 Quantization")
if config.get('enable_phase9_curriculum'):
    print("  ✓ Phase 9: Dynamic Ring Curriculum")
if config.get('enable_phase10_sparse_attention'):
    print("  ✓ Phase 10: Sparse Ring Attention")
EOF

# ============================================================================
# STEP 5: PREPARE FOR TRAINING
# ============================================================================

print_header "STEP 5: PREPARE FOR TRAINING"

print_info "Creating checkpoint directory..."
mkdir -p "checkpoints_$CONFIG"
print_step "Checkpoint directory ready: checkpoints_$CONFIG"

# ============================================================================
# STEP 6: RUN TRAINING
# ============================================================================

print_header "STEP 6: RUN YOUR FIRST TRAINING"

echo "The following command will start training:"
echo ""
echo -e "${BOLD}python3 train/zenyx_unified_tpu_training.py \\${NC}"
echo -e "    --tpu-version \$(python3 -c \"from train.zenyx_config_templates import get_config; print(get_config('$CONFIG')['tpu_version'])\") \\${NC}"
echo -e "    --num-epochs 2${NC}"
echo ""

read -p "Start training now? (y/n): " start_training

if [ "$start_training" = "y" ] || [ "$start_training" = "Y" ]; then
    print_step "Starting training..."
    python3 train/zenyx_unified_tpu_training.py \
        --tpu-version $(python3 -c "from train.zenyx_config_templates import get_config; print(get_config('$CONFIG')['tpu_version'])") \
        --num-epochs 2
else
    echo "You can start training manually later:"
    echo "  python3 train/zenyx_unified_tpu_training.py --num-epochs 2"
fi

# ============================================================================
# STEP 7: MONITOR TRAINING
# ============================================================================

print_header "STEP 7: MONITOR YOUR TRAINING"

echo "Checkpoint directory: checkpoints_$CONFIG"
echo ""
echo "View training metrics:"
echo "  cat checkpoints_$CONFIG/metrics.json"
echo ""
echo "View latest checkpoint:"
echo "  ls -lh checkpoints_$CONFIG/checkpoint_*.pt | tail -1"
echo ""

if command -v tensorboard &> /dev/null; then
    echo "View with TensorBoard:"
    echo "  tensorboard --logdir checkpoints_$CONFIG"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_header "YOU'RE ALL SET!"

echo "Next steps:"
echo ""
echo "1. Customize the training:"
echo "   python3 train/zenyx_training_examples.py --example 1"
echo ""
echo "2. View all examples:"
echo "   python3 train/zenyx_training_examples.py"
echo ""
echo "3. List all configurations:"
echo "   python3 train/zenyx_config_templates.py"
echo ""
echo "4. Advanced usage:"
echo "   python3 train/zenyx_unified_tpu_training.py --help"
echo ""

echo -e "${BOLD}${GREEN}ZENYX is ready! Train trillion-parameter models with zero OOMs.${NC}"
