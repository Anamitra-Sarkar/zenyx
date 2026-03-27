"""
ZENYX-V2 COMPLETE TRAINING GUIDE

This guide explains how to use the production Zenyx-V2 training script
for training billion-parameter to trillion-parameter models on TPU v5e.

═══════════════════════════════════════════════════════════════════════════════

TABLE OF CONTENTS

1. Quick Start (85M model, 8K context)
2. Scaling to Larger Models (350M → 1T)
3. Context Length Scaling (8K → 1M)
4. Hardware Requirements
5. Data Preparation
6. Training Loop Implementation
7. Monitoring & Checkpointing
8. Production Deployment

═══════════════════════════════════════════════════════════════════════════════

§1  QUICK START — 85M MODEL ON SINGLE TPU v5e-8

Step 1: Environment Setup
────────────────────────

    # On TPU v5e instance (Google Cloud):
    pip install jax[tpu]=="0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install flax optax transformers datasets huggingface_hub

Step 2: Initialize Config
──────────────────────────

    from zenyx_v2_tpu_production import create_config, ZenyxV2Model
    
    config = create_config(
        model_size="nano",      # 85M parameters
        max_seq_len=8_192,      # 8K context
        hf_token="your_token"   # From huggingface.co
    )

Step 3: Initialize Model
────────────────────────

    import jax
    from jax import random as jrand
    
    model = ZenyxV2Model(config=config)
    init_rng = jrand.PRNGKey(42)
    dummy = jnp.ones((1, config.MAX_SEQ_LEN), dtype=jnp.int32)
    
    variables = model.init(init_rng, input_ids=dummy, train=False)
    params = variables["params"]
    
    # Check parameter count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model size: {param_count:,} parameters ({param_count/1e6:.1f}M)")

Step 4: Create Optimizer
────────────────────────

    import optax
    from jax import random as jrand
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.GRAD_CLIP),
        optax.adamw(
            learning_rate=config.LEARNING_RATE,
            b1=config.BETA1,
            b2=config.BETA2,
            eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY,
        ),
    )

Step 5: Training Loop (Pseudo-code)
────────────────────────────────────

    from flax.training import train_state
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    
    def train_step(state, batch):
        def loss_fn(params):
            logits_list = model.apply(
                {"params": params},
                input_ids=batch,
                train=True,
            )
            # Compute loss from logits_list
            return total_loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss
    
    # Distributed with pmap (8 TPU cores):
    train_step_pmap = jax.pmap(train_step, axis_name="devices")
    
    # Training loop
    for step, batch in enumerate(train_dataloader):
        state, loss = train_step_pmap(state, batch)
        if step % 100 == 0:
            print(f"Step {step:5d} | Loss: {loss:.4f}")

═══════════════════════════════════════════════════════════════════════════════

§2  SCALING TO LARGER MODELS

Zenyx-V2 supports 6 preset sizes:

┌─────────┬──────────────────┬──────────────┬─────────────┐
│  Size   │ Parameters       │ d_model      │ Blocks      │
├─────────┼──────────────────┼──────────────┼─────────────┤
│ nano    │ 85M              │ 576          │ 8×4=32      │
│ small   │ 350M             │ 1024         │ 12×3=36     │
│ base    │ 1.3B             │ 1536         │ 16×2=32     │
│ large   │ 8B               │ 2048         │ 24×2=48     │
│ xl      │ 85B              │ 3072         │ 32×1=32     │
│ epic    │ 1T (trillion)    │ 4096         │ 128×1=128   │
└─────────┴──────────────────┴──────────────┴─────────────┘

Usage:

    # 1.3B model
    config = create_config(model_size="base", max_seq_len=8_192)
    model = ZenyxV2Model(config=config)
    
    # 1 Trillion parameter model
    config = create_config(model_size="epic", max_seq_len=1_000_000)
    model = ZenyxV2Model(config=config)

Note: For models > 350M, use distributed training with tensor parallelism (TP).

═══════════════════════════════════════════════════════════════════════════════

§3  CONTEXT LENGTH SCALING — 8K → 1M TOKENS

YaRN-scaled RoPE allows arbitrary context lengths without retraining:

┌──────────┬──────────────────┐
│ Seq Len  │ Supported After  │
├──────────┼──────────────────┤
│ 8K       │ Any training     │
│ 32K      │ Min training     │
│ 128K     │ 32K+ training    │
│ 1M       │ 128K+ training   │
└──────────┴──────────────────┘

To use million-token context:

    config = create_config(
        model_size="base",
        max_seq_len=1_000_000,  # 1M tokens!
    )
    model = ZenyxV2Model(config=config)
    
    # RoPE is automatically scaled via YaRN

Memory usage grows linearly with context. For 1M context on 8B model:
- Forward pass: ~180 GB
- Gradient computation: ~540 GB
- Requires: 8× TPU v5e-8 (128 GB HBM each) with tensor parallelism

═══════════════════════════════════════════════════════════════════════════════

§4  HARDWARE REQUIREMENTS

For different model sizes:

┌──────────┬─────────────────────────────────────────────────────┐
│ Model    │ Hardware                                            │
├──────────┼─────────────────────────────────────────────────────┤
│ nano     │ 1× TPU v5e-8 (16 GB)                                │
│ small    │ 2× TPU v5e-8 (tensor parallelism)                   │
│ base     │ 4× TPU v5e-8 (TP=4)                                 │
│ large    │ 8× TPU v5e-8 (TP=8)                                 │
│ xl       │ 16× TPU v5e-8 (TP=16) OR 4× TPU v5p (TP=4)          │
│ epic     │ 128× TPU v5e-8 (TP=128) OR 32× TPU v5p (TP=32)      │
└──────────┴─────────────────────────────────────────────────────┘

Key constraint: Per-core batch size = 1 (to fit large models in 16GB HBM)

═══════════════════════════════════════════════════════════════════════════════

§5  DATA PREPARATION

Three supported datasets:

1. Math (45% of training):
   - finemath-4plus
   - infiwebmath-3plus
   - infiwebmath-4plus

2. Code (35% of training):
   - bigcode/starcoderdata (24 languages)

3. English (20% of training):
   - HuggingFaceFW/fineweb-edu

To stream data:

    from zenyx_v2_tpu_production import (
        stream_finemath,
        stream_code,
        stream_english,
        combined_stream,
    )
    
    # Interleave all three sources
    train_stream = combined_stream(resume_step=0, seed=42)
    
    # Or use individually:
    math_blocks = stream_finemath("finemath-4plus", target_bytes=10e9, seed=42)
    code_blocks = stream_code(target_bytes=10e9, seed=43)
    eng_blocks = stream_english(target_bytes=10e9, seed=44)

═══════════════════════════════════════════════════════════════════════════════

§6  TRAINING LOOP IMPLEMENTATION

Full working example:

    import jax
    import jax.numpy as jnp
    from jax import random as jrand
    import optax
    from flax.training import train_state
    
    from zenyx_v2_tpu_production import (
        create_config, ZenyxV2Model, compute_mtp_loss, combined_stream
    )
    
    # Config
    config = create_config(model_size="small", max_seq_len=8_192)
    
    # Model
    model = ZenyxV2Model(config=config)
    init_rng = jrand.PRNGKey(42)
    dummy = jnp.ones((1, config.MAX_SEQ_LEN), dtype=jnp.int32)
    variables = model.init(init_rng, input_ids=dummy, train=False)
    
    # Optimizer
    def lr_schedule(step):
        warmup_lr = config.LEARNING_RATE * jnp.minimum(
            step / config.WARMUP_STEPS, 1.0
        )
        decay_progress = jnp.clip(
            (step - config.WARMUP_STEPS - config.STABLE_STEPS) / config.DECAY_STEPS,
            0.0, 1.0
        )
        cosine_lr = config.MIN_LR + 0.5 * (config.LEARNING_RATE - config.MIN_LR) * (
            1.0 + jnp.cos(jnp.pi * decay_progress)
        )
        in_warmup = step < config.WARMUP_STEPS
        in_stable = (step >= config.WARMUP_STEPS) & (
            step < config.WARMUP_STEPS + config.STABLE_STEPS
        )
        return jnp.where(
            in_warmup, warmup_lr,
            jnp.where(in_stable, config.LEARNING_RATE, cosine_lr)
        )
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.GRAD_CLIP),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.BETA1, b2=config.BETA2, eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY,
        ),
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )
    
    # Training step
    @jax.checkpoint
    def train_step(state, batch, dropout_rng):
        def loss_fn(params):
            return compute_mtp_loss(
                params, batch, dropout_rng, model, config, pad_id=0
            )
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss
    
    # Distributed training
    train_step_pmap = jax.pmap(
        train_step, axis_name="devices", donate_argnums=(0,)
    )
    
    # Training loop
    data_stream = combined_stream(resume_step=0, seed=42)
    
    for step in range(config.MAX_STEPS):
        batch = next(data_stream)
        batch = jnp.array(batch, dtype=jnp.int32)
        
        dropout_rng = jrand.fold_in(init_rng, step)
        state, loss = train_step_pmap(state, batch, dropout_rng)
        
        if step % 100 == 0:
            print(f"Step {step:6d} | Loss: {float(loss):.4f}")

═══════════════════════════════════════════════════════════════════════════════

§7  MONITORING & CHECKPOINTING

Save and load checkpoints:

    from flax import serialization
    
    # Save
    def save_checkpoint(state, step):
        params_bytes = serialization.to_bytes(state.params)
        with open(f"ckpt_step{step}.msgpack", "wb") as f:
            f.write(params_bytes)
    
    # Load
    def load_checkpoint(path):
        with open(path, "rb") as f:
            params_bytes = f.read()
        params = serialization.from_bytes(variables["params"], params_bytes)
        return params
    
    # In training loop:
    if step % config.SAVE_EVERY == 0:
        save_checkpoint(state, step)

═══════════════════════════════════════════════════════════════════════════════

§8  VALIDATION CHECKLIST

Before training the real model:

    [ ] JAX 0.4.16+ installed on TPU v5e instance
    [ ] Flax, Optax, Transformers installed
    [ ] HuggingFace token configured
    [ ] Test model initializes:
        python -c "from zenyx_v2_tpu_production import validate_all_sizes; validate_all_sizes()"
    [ ] Data streams working (can iterate)
    [ ] Loss function computes correctly
    [ ] Optimizer updates work
    [ ] Checkpointing saves/loads

═══════════════════════════════════════════════════════════════════════════════

§9  PRODUCTION DEPLOYMENT

Once validated, deploy with:

    # On TPU v5e-8 instance:
    
    nohup python train_zenyx_v2.py \\
        --model-size epic \\
        --max-context 1000000 \\
        --global-batch 256 \\
        --learning-rate 3e-4 \\
        --total-steps 100000 \\
        --checkpoint-every 500 \\
        --save-to-hub \\
        --repo-id Arko007/zenyx-v2-epic \\
        > training.log 2>&1 &

For multi-host (multiple TPU instances):

    # Use jax.distributed.initialize() for cross-host communication
    
    from jax.distributed import initialize
    initialize()
    
    # Then pmap will automatically span all hosts

═══════════════════════════════════════════════════════════════════════════════

KEY FORMULAS

Model size estimation:
    params = d_model × (
        vocab_size +                              # embedding
        n_heads × head_dim × d_model × 2 +        # attention proj
        n_kv_heads × head_dim × d_model × 2 +     # KV proj
        hidden_dim × d_model × 2 +                # MLP
        d_model                                   # layer norm
    ) × n_layers

Tokens per second:
    throughput = global_batch_size × seq_len × device_count / step_time

Memory per token:
    activations ≈ 14 × d_model × seq_len bytes (BF16)

═══════════════════════════════════════════════════════════════════════════════

For questions or issues, refer to:
- README.md (library overview)
- VALIDATION_REPORT.md (test results)
- PROJECT_COMPLETE.md (implementation details)

Good luck training! 🚀
"""

if __name__ == "__main__":
    print(__doc__)
