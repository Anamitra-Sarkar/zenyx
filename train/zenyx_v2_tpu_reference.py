# ZENYX-V2 PRETRAINING — TPU v5e-8 | JAX/Flax | Pure BF16
# Architecture: Nano-Titan | ~85M unique params | 8192 context
# Tokenizer: Arko007/zenyx-v2-tokenizer | vocab=32,768
# Data: Math 45% + StarCoderData 35% + FineWeb-Edu 20%
# Depth: 8 unique blocks x 4 recurrences = 32 effective layers
#
# Fix 20 [OOM]: DROPOUT_RATE 0.1 -> 0.0
# Fix 21 [DATA]: Math datasets fresh — skip=0 for all three math streams.
#                Code and English skip normally from resume_step.
# Fix 22 [CRITICAL]: @jax.checkpoint restored on compute_mtp_loss
# Fix 23 [BUG]: Regex double-escape in load_latest_checkpoint fixed
# Fix 24 [MINOR]: ConvSwiGLU dropout deterministic propagation corrected
# Fix 25 [DATA]: Shard-index skip — zero-cost parquet shard-file sli
cing.
# Fix 26 [DATA]: Correct parquet path patterns per dataset (finemath/starcoder/fineweb)

import os, re, gc, sys, json, math, time, logging
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
from pathlib import Path
from functools import partial
from queue import Queue
from threading import Thread

import jax
import jax.numpy as jnp
from jax import random as jrand
import optax
import flax
import flax.linen as nn
from flax.training import train_state
from flax import serialization
import flax.jax_utils

_jax_ver = tuple(int(x) for x in jax.__version__.split(".")[:3])
if _jax_ver < (0, 4, 16):
    raise RuntimeError(f"JAX >= 0.4.16 required. Found: {jax.__version__}")

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, login

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ZenyxV2-Train")

# ════════════════════════════════════════════════════════════════════════════════
# §1  AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════════
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
if HF_TOKEN is None:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass
if HF_TOKEN is None:
    HF_TOKEN = "HF_TOKEN" #put your actual token over here
login(token=HF_TOKEN, add_to_git_credential=False)

# REFERENCE SCRIPT - See zenyx_v2_tpu_production.py for production implementation
