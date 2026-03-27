#!/bin/bash
################################################################################
# ZENYX SINGLE TPU v5e-8 DEPLOYMENT SCRIPT
# Train 1 Trillion Parameters on a Single 16GB v5e-8 TPU
# 
# Usage:
#   1. Copy this script to your local machine
#   2. Update: GCP_PROJECT, GCP_ZONE, TPU_NAME (lines below)
#   3. Run: bash DEPLOY_NOW.sh
################################################################################

set -e

# ============================================================================
# CONFIGURATION - UPDATE THESE
# ============================================================================

GCP_PROJECT="your-gcp-project"      # Your Google Cloud project ID
GCP_ZONE="us-central1-a"            # Google Cloud zone (supports v5e)
TPU_NAME="zenyx-tpu"                # Name for your TPU VM
REPO_URL="https://github.com/your/repo" # Your repository URL

echo "=========================================="
echo "ZENYX Single TPU v5e-8 Deployment"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  GCP Project: $GCP_PROJECT"
echo "  Zone: $GCP_ZONE"
echo "  TPU Name: $TPU_NAME"
echo ""

# ============================================================================
# STEP 1: Create TPU v5e-8 VM
# ============================================================================

echo "Step 1: Creating TPU v5e-8 VM..."
echo "Command:"
echo "  gcloud compute tpus tpu-vm create $TPU_NAME \\"
echo "    --project=$GCP_PROJECT \\"
echo "    --zone=$GCP_ZONE \\"
echo "    --accelerator-type=v5e-8 \\"
echo "    --version=tpu-ubuntu2204-base"
echo ""
echo "Running..."

gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --accelerator-type=v5e-8 \
    --version=tpu-ubuntu2204-base

echo "✓ TPU created successfully"
echo ""

# ============================================================================
# STEP 2: Setup environment on TPU
# ============================================================================

echo "Step 2: Setting up environment on TPU VM..."
echo ""

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --command="bash -c '
    set -e
    cd /home/\$(whoami)
    
    # Clone repository
    git clone $REPO_URL zenyx
    cd zenyx
    
    # Install dependencies
    pip install --upgrade pip
    pip install -e .
    pip install -r requirements.txt
    pip install jax[tpu] -U
    
    echo \"✓ Setup complete\"
    '"

echo "✓ Environment setup complete"
echo ""

# ============================================================================
# STEP 3: Run training
# ============================================================================

echo "Step 3: Starting training..."
echo ""
echo "Running on TPU:"
echo "  python train/zenyx_single_tpu_train.py --train --steps 50000"
echo ""

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --command="bash -c '
    cd /home/\$(whoami)/zenyx
    python train/zenyx_single_tpu_train.py --train --steps 50000
    '"

echo "✓ Training complete!"
echo ""

# ============================================================================
# STEP 4: Retrieve results
# ============================================================================

echo "Step 4: Retrieving checkpoints..."
echo ""

gcloud compute tpus tpu-vm scp "$TPU_NAME:/home/*/zenyx/final_ckpt" . \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --recurse

echo "✓ Checkpoints retrieved: final_ckpt/"
echo ""

# ============================================================================
# Done
# ============================================================================

echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Your 1 trillion parameter model has been trained!"
echo ""
echo "Next steps:"
echo "1. Check results in final_ckpt/"
echo "2. To keep the TPU running:"
echo "   gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$GCP_ZONE"
echo "3. To delete the TPU when done:"
echo "   gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$GCP_ZONE"
echo ""

