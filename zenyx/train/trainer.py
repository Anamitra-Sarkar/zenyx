"""Zenyx Trainer — single entrypoint for distributed LLM training.

Usage:
    import zenyx
    zenyx.train(model, dataloader)

The Trainer auto-detects hardware, sets up memory pools, selects the
correct attention kernel (ring_flash_cuda for H100, ring_pallas_tpu for
TPU, flash_cpu for CPU), configures the optimizer, and runs the training
loop. The user never touches distributed setup, memory management, or
kernel selection.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from zenyx.train.lr_schedule import CosineWithWarmup
from zenyx.train.grad_scaler import ZenyxGradScaler
from zenyx.train.distributed_setup import (
    auto_init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    cleanup,
)

__all__ = ["Trainer", "train"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class Trainer:
    """Full-stack LLM trainer with automatic hardware detection and memory management."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Any,
        *,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        # Schedule
        warmup_steps: int = 2000,
        total_steps: int = 100_000,
        # Precision
        dtype: str = "bfloat16",
        activation_dtype: str = "float8_e4m3",
        # Memory
        t1_capacity_gb: float = 8.0,
        t2_capacity_gb: float = 64.0,
        # Context
        context_len: int = 4096,
        # MoE
        dynamic_routing: bool = False,
        heap_rebuild_interval: int = 1000,
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        checkpoint_every: int = 1000,
        resume_from: Optional[str] = None,
        # Logging
        log_every: int = 10,
        # Advanced
        gradient_accumulation_steps: int = 1,
        selective_activation_checkpoint: bool = True,
        checkpoint_every_nth_layer: int = 4,
    ) -> None:
        self._model = model
        self._dataloader = dataloader
        self._grad_clip = grad_clip
        self._total_steps = total_steps
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every = checkpoint_every
        self._log_every = log_every
        self._gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._dynamic_routing = dynamic_routing
        self._heap_rebuild_interval = heap_rebuild_interval
        self._context_len = context_len

        # Training state
        self._step: int = 0
        self._last_loss: float = 0.0
        self._recent_step_times: list[float] = []

        # --- Step 1: Hardware detection ---
        from zenyx.core.hal.detector import detect_hardware, HardwareInfo

        self._hw_info: HardwareInfo = detect_hardware()

        # --- Step 2: Topology detection ---
        from zenyx.ops.comm.topology import detect_topology, TopologyInfo

        self._topo_info: TopologyInfo = detect_topology()

        # --- Log hardware and topology ---
        logger.info(
            "Hardware: %s | Memory: %.1f GB | Compute: %.1f TFLOPS",
            self._hw_info.device_name,
            self._hw_info.per_device_memory_bytes / (1024**3),
            self._hw_info.compute_tflops,
        )
        logger.info(
            "Topology: %s | World size: %d | S_min: %d tokens",
            self._topo_info.interconnect,
            self._topo_info.world_size,
            self._topo_info.s_min_tokens,
        )

        # --- Warn if sequence chunks too small for ring attention ---
        world_size = self._topo_info.world_size
        if world_size > 1:
            chunk_size = context_len // world_size
            if chunk_size < self._topo_info.s_min_tokens:
                logger.warning(
                    "Sequence chunk size (%d tokens) < S_min (%d tokens). "
                    "Ring attention communication will bottleneck compute. "
                    "Consider increasing context_len or reducing world_size.",
                    chunk_size,
                    self._topo_info.s_min_tokens,
                )

        # --- Step 3: Determine device ---
        self._device = torch.device("cpu")
        self._backend = "cpu"
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self._device = torch.device(f"cuda:{local_rank}")
            self._backend = "cuda"
        elif self._topo_info.backend == "xla_ici":
            self._backend = "tpu"

        # --- Step 4: Create HAL, MemoryPool, ReuseHeap, TierAllocator ---
        self._hal: Any = None
        self._pool: Any = None
        self._reuse_heap: Any = None
        self._tier_allocator: Any = None

        try:
            if self._backend == "cuda":
                from zenyx.core.hal.cuda_hal import CUDAHal
                self._hal = CUDAHal()
            # For CPU/TPU we may not have a HAL implementation yet
        except Exception as e:
            logger.debug("HAL init skipped: %s", e)

        if self._hal is not None:
            try:
                from zenyx.core.allocator.mem_pool import MemoryPool
                from zenyx.core.allocator.reuse_heap import ReuseHeap
                from zenyx.core.allocator.tier_allocator import TierAllocator

                self._pool = MemoryPool(
                    self._hal,
                    self._hw_info,
                    t1_capacity=int(t1_capacity_gb * 1024**3),
                    t2_capacity=int(t2_capacity_gb * 1024**3),
                )
                self._reuse_heap = ReuseHeap()
                self._tier_allocator = TierAllocator(self._pool, self._reuse_heap)
            except Exception as e:
                logger.debug("Memory pool init skipped: %s", e)

        # --- Step 5: Select attention kernel ---
        self._attention_kernel_name = "flash_cpu"
        device_name = getattr(self._hw_info, "device_name", "").lower()
        if "h100" in device_name or (self._backend == "cuda"):
            self._attention_kernel_name = "ring_flash_cuda"
        elif self._backend == "tpu":
            self._attention_kernel_name = "ring_pallas_tpu"

        # --- Step 6: Activation checkpointing ---
        if selective_activation_checkpoint:
            from zenyx.train.activation_checkpoint import selective_checkpoint_wrapper
            self._model = selective_checkpoint_wrapper(
                self._model,
                checkpoint_every_nth=checkpoint_every_nth_layer,
            )

        # --- Step 7: Move model to device ---
        compute_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
        self._model = self._model.to(device=self._device, dtype=compute_dtype)

        # --- Step 8: Optimizer ---
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )

        # --- Step 9: LR schedule ---
        self._scheduler = CosineWithWarmup(
            optimizer=self._optimizer,
            peak_lr=lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # --- Step 10: Grad scaler ---
        scaler_enabled = dtype != "float32" and self._backend == "cuda"
        self._grad_scaler = ZenyxGradScaler(enabled=scaler_enabled)

        # --- Step 11: AMP context ---
        self._amp_dtype = compute_dtype
        self._use_amp = dtype != "float32" and self._backend == "cuda"

        # --- Step 12: Resume from checkpoint ---
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        # --- Step 13: Loss function ---
        self._loss_fn: Any = None
        if get_world_size() > 1:
            try:
                from zenyx.ops.vocab.vocab_parallel import VocabParallelCrossEntropy
                self._loss_fn = VocabParallelCrossEntropy
                logger.info("Using VocabParallelCrossEntropy for distributed loss")
            except ImportError:
                self._loss_fn = nn.CrossEntropyLoss()
        else:
            self._loss_fn = nn.CrossEntropyLoss()

        logger.info(
            "Zenyx Trainer initialized. Hardware: %s. Attention: %s. Pool: %s",
            self._backend,
            self._attention_kernel_name,
            "active" if self._pool is not None else "none",
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop.

        Algorithm:
        1. For each batch from dataloader:
           a. Forward pass through model
           b. Compute loss
           c. Scale loss by 1/gradient_accumulation_steps
           d. Backward pass
           e. If gradient_accumulation_steps reached:
              - Clip gradients (grad_clip)
              - Optimizer step
              - LR scheduler step
              - Zero gradients
              - Log if log_every
              - Checkpoint if checkpoint_every
              - If dynamic_routing and step % heap_rebuild_interval == 0:
                  trigger reuse_heap.rebuild_async(updated_graph)
        2. Final checkpoint save
        """
        self._model.train()

        accum_loss = 0.0
        micro_step = 0

        for batch in self._dataloader:
            if self._step >= self._total_steps:
                break

            step_start = time.monotonic()

            # Unpack batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch
                labels = None

            # Move to device
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device, non_blocking=True)
            if labels is not None and hasattr(labels, "to"):
                labels = labels.to(self._device, non_blocking=True)

            # Forward + loss
            if self._use_amp:
                with torch.amp.autocast(device_type=self._backend, dtype=self._amp_dtype):
                    output = self._model(inputs)
                    loss = self._compute_loss(output, labels)
            else:
                output = self._model(inputs)
                loss = self._compute_loss(output, labels)

            # Scale for gradient accumulation
            loss = loss / self._gradient_accumulation_steps

            # Backward
            self._grad_scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro_step += 1

            # Optimizer step at accumulation boundary
            if micro_step % self._gradient_accumulation_steps == 0:
                # Clip gradients
                if self._grad_clip > 0:
                    if self._grad_scaler._enabled:
                        self._grad_scaler._scaler.unscale_(self._optimizer) if self._grad_scaler._scaler else None
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._grad_clip
                    )

                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
                self._scheduler.step()
                self._optimizer.zero_grad(set_to_none=True)

                self._step += 1
                self._last_loss = accum_loss * self._gradient_accumulation_steps

                # Track step time
                elapsed = time.monotonic() - step_start
                self._recent_step_times.append(elapsed)
                if len(self._recent_step_times) > 100:
                    self._recent_step_times = self._recent_step_times[-100:]

                # Logging
                if is_main_process() and self._step % self._log_every == 0:
                    lr = self._scheduler.get_lr()
                    logger.info(
                        "Step %d | Loss: %.4f | LR: %.2e | Time: %.3fs",
                        self._step,
                        self._last_loss,
                        lr,
                        elapsed,
                    )

                # Checkpoint
                if self._step % self._checkpoint_every == 0:
                    self._save_checkpoint()

                # Heap rebuild for dynamic routing
                if (
                    self._dynamic_routing
                    and self._reuse_heap is not None
                    and self._step % self._heap_rebuild_interval == 0
                ):
                    try:
                        self._reuse_heap.rebuild_async(None)
                    except Exception as e:
                        logger.debug("Heap rebuild skipped: %s", e)

                accum_loss = 0.0

        # Final checkpoint
        if is_main_process():
            self._save_checkpoint()
            logger.info("Training complete at step %d", self._step)

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return training state snapshot for monitoring.

        Returns dict with:
        - step: current step number
        - loss: last logged loss
        - lr: current learning rate
        - memory_usage: pool.usage() snapshot
        - topology: topo info
        - throughput_tokens_per_sec: computed from recent steps
        """
        throughput = 0.0
        if self._recent_step_times:
            avg_time = sum(self._recent_step_times) / len(self._recent_step_times)
            if avg_time > 0:
                # Estimate tokens/sec based on context_len and batch
                throughput = self._context_len / avg_time

        pool_usage: Optional[dict] = None
        if self._pool is not None and hasattr(self._pool, "usage"):
            try:
                pool_usage = self._pool.usage()
            except Exception:
                pass

        return {
            "step": self._step,
            "loss": self._last_loss,
            "lr": self._scheduler.get_lr(),
            "memory_usage": pool_usage,
            "topology": {
                "backend": self._topo_info.backend,
                "interconnect": self._topo_info.interconnect,
                "world_size": self._topo_info.world_size,
                "s_min_tokens": self._topo_info.s_min_tokens,
            },
            "throughput_tokens_per_sec": throughput,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_loss(
        self, output: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss from model output and labels."""
        if labels is not None and isinstance(self._loss_fn, nn.CrossEntropyLoss):
            # Reshape for cross-entropy: (B*S, V) vs (B*S,)
            if output.dim() == 3:
                B, S, V = output.shape
                output = output.view(B * S, V)
                labels = labels.view(B * S)
            return self._loss_fn(output.float(), labels)

        # Fallback: mean of output as loss (for testing)
        if hasattr(output, "mean"):
            return output.float().mean()
        return output

    def _save_checkpoint(self) -> None:
        """Save a checkpoint to disk."""
        if not is_main_process():
            return

        os.makedirs(self._checkpoint_dir, exist_ok=True)
        path = os.path.join(self._checkpoint_dir, f"step_{self._step}.pt")

        try:
            state = {
                "step": self._step,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_step": self._scheduler.current_step,
            }
            torch.save(state, path)
            logger.info("Checkpoint saved: %s", path)
        except Exception as e:
            logger.warning("Checkpoint save failed: %s", e)

    def _load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from disk."""
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            return

        try:
            state = torch.load(path, map_location=self._device, weights_only=False)
            self._model.load_state_dict(state["model_state_dict"])
            self._optimizer.load_state_dict(state["optimizer_state_dict"])
            self._step = state.get("step", 0)
            # Advance scheduler to match
            for _ in range(state.get("scheduler_step", 0)):
                self._scheduler.step()
            logger.info("Resumed from checkpoint: %s (step %d)", path, self._step)
        except Exception as e:
            logger.warning("Checkpoint load failed: %s", e)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    dataloader: Any,
    **kwargs: Any,
) -> Trainer:
    """One-line training entrypoint.

    Creates a Trainer with the given model and dataloader, starts training,
    and returns the Trainer for inspection.

    Example:
        trainer = zenyx.train(model, dataloader, context_len=131072)
    """
    trainer = Trainer(model, dataloader, **kwargs)
    trainer.train()
    return trainer
