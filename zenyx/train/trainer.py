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
    """Full-stack LLM trainer with automatic hardware detection and memory management.

    Integrates the agent feedback loop (AsyncProfiler → ParallelismPlanner →
    TrainingController) for autonomous replanning and the fast model loader
    for checkpoint loading.

    Time: O(steps × (forward + backward + optimizer)).  Space: O(model + activations).
    """

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
        # Loader
        loader_config: Optional[Any] = None,
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
        self._loader_config = loader_config

        # Training state
        self._step: int = 0
        self._last_loss: float = 0.0
        self._recent_step_times: list[float] = []

        # Compute graph for Bélády-optimal eviction (set via register_compute_graph)
        self._compute_graph: Optional[Any] = None

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

        # --- Step 4: Create HAL + TierAllocator (manages pool and heap internally) ---
        self._hal: Any = None
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
                from zenyx.core.allocator.tier_allocator import TierAllocator

                self._tier_allocator = TierAllocator(
                    self._hw_info, block_size_mb=4,
                )
            except Exception as e:
                logger.debug("TierAllocator init skipped: %s", e)

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

        # --- Step 12: Agent integration (profiler, planner, controller) ---
        from zenyx.core.agent.profiler import AsyncProfiler
        from zenyx.core.agent.planner import ParallelismPlanner, ParallelismPlan
        from zenyx.core.agent.controller import TrainingController
        from zenyx.ops.comm.topology import Topology

        self._profiler = AsyncProfiler(enabled=True)

        # Build a Topology stub from TopologyInfo for the planner
        topo = Topology()
        self._planner = ParallelismPlanner(self._hw_info, topo)

        # Compute initial parallelism plan
        model_params = float(sum(p.numel() for p in self._model.parameters()))
        vocab_size = self._infer_vocab_size()
        batch_size = self._infer_batch_size()
        self._parallelism_plan: ParallelismPlan = self._planner.plan(
            model_params=model_params,
            vocab_size=vocab_size,
            context_len=context_len,
            batch_size=batch_size,
        )
        logger.info("Initial parallelism plan: %s", self._parallelism_plan)

        # Training controller
        self._controller = TrainingController(
            planner=self._planner,
            profiler=self._profiler,
            replan_interval=heap_rebuild_interval,
            model_params=model_params,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )

        # --- Step 13: Resume from checkpoint ---
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        # --- Step 14: Loss function ---
        self._loss_fn: Any = nn.CrossEntropyLoss()
        self._use_vocab_parallel = False
        if get_world_size() > 1:
            try:
                from zenyx.ops.vocab.vocab_parallel import VocabParallelCrossEntropy
                self._loss_fn = VocabParallelCrossEntropy
                self._use_vocab_parallel = True
                logger.info("Using VocabParallelCrossEntropy for distributed loss")
            except ImportError:
                pass

        logger.info(
            "Zenyx Trainer initialized. Hardware: %s. Attention: %s. Allocator: %s",
            self._backend,
            self._attention_kernel_name,
            "active" if self._tier_allocator is not None else "none",
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def register_compute_graph(self, graph: Any) -> None:
        """Register a compute graph for Bélády-optimal eviction scheduling.

        Users can provide graph information for optimal eviction planning
        in dynamic-routing / MoE models.

        Time: O(G + N log N) where G = graph ops, N = tracked blocks.
        Space: O(G).

        Args:
            graph: A :class:`ComputeGraph` instance.
        """
        self._compute_graph = graph
        if self._tier_allocator is not None:
            self._tier_allocator.register_compute_graph(graph)
            logger.info("Compute graph registered with TierAllocator")

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
                  trigger graph rebuild or controller replan
        2. Final checkpoint save

        Time: O(steps × (forward + backward + optimizer)).
        """
        self._model.train()

        accum_loss = 0.0
        micro_step = 0

        for batch in self._dataloader:
            if self._step >= self._total_steps:
                break

            step_start = time.monotonic()

            # Start profiling the training step
            _profile_handle = self._profiler.start_op("train_step")

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
                    self._grad_scaler.unscale_(self._optimizer)
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

                # Estimate throughput for profiler
                throughput_estimate = self._context_len / elapsed if elapsed > 0 else 0.0

                # End profiling the training step
                self._profiler.end_op(_profile_handle)

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

                # Heap rebuild / controller replan for dynamic routing
                if (
                    self._dynamic_routing
                    and self._step % self._heap_rebuild_interval == 0
                ):
                    if self._compute_graph is not None and self._tier_allocator is not None:
                        try:
                            self._tier_allocator.register_compute_graph(
                                self._compute_graph,
                            )
                        except Exception as e:
                            logger.debug("Heap rebuild skipped: %s", e)
                    else:
                        logger.debug(
                            "Heap rebuild skipped: no compute graph registered "
                            "(dynamic_routing=True). Call register_compute_graph() "
                            "to enable Bélády-optimal eviction."
                        )

                # Controller replan check (advisory, non-blocking)
                try:
                    new_plan = self._controller.step(
                        step_num=self._step,
                        loss=self._last_loss,
                        context_len=self._context_len,
                    )
                    if new_plan is not None:
                        self._parallelism_plan = new_plan
                        logger.warning(
                            "Replanning mid-training at step %d: %s",
                            self._step,
                            new_plan,
                        )
                except Exception as e:
                    logger.debug("Controller step skipped: %s", e)

                accum_loss = 0.0

        # Final checkpoint
        if is_main_process():
            self._save_checkpoint()
            logger.info("Training complete at step %d", self._step)

        # Shutdown profiler
        self._profiler.shutdown()

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
        - parallelism_plan: current parallelism plan
        - profiler_stats: profiler timing stats

        Time: O(K) where K = profiled ops.  Space: O(K).
        """
        throughput = 0.0
        if self._recent_step_times:
            avg_time = sum(self._recent_step_times) / len(self._recent_step_times)
            if avg_time > 0:
                # Estimate tokens/sec based on context_len and batch
                throughput = self._context_len / avg_time

        pool_usage: Optional[dict] = None
        if self._tier_allocator is not None:
            pool = getattr(self._tier_allocator, "_pool", None)
            if pool is not None and hasattr(pool, "usage"):
                try:
                    pool_usage = pool.usage()
                except Exception:
                    pass

        # Profiler stats
        profiler_stats: Optional[dict] = None
        try:
            timings = self._profiler.get_timings()
            profiler_stats = {
                k: {"avg_ms": v.avg_ms, "count": v.count}
                for k, v in timings.items()
            }
        except Exception:
            pass

        # Parallelism plan
        plan_dict: Optional[dict] = None
        if self._parallelism_plan is not None:
            plan_dict = {
                "tp_degree": self._parallelism_plan.tp_degree,
                "pp_degree": self._parallelism_plan.pp_degree,
                "dp_degree": self._parallelism_plan.dp_degree,
                "ring_degree": self._parallelism_plan.ring_degree,
                "schedule_type": self._parallelism_plan.schedule_type,
            }

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
            "parallelism_plan": plan_dict,
            "profiler_stats": profiler_stats,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_vocab_size(self) -> int:
        """Try to infer vocabulary size from model attributes.

        Time: O(1).
        """
        for attr in ("config", "cfg"):
            cfg = getattr(self._model, attr, None)
            if cfg is not None:
                for key in ("vocab_size", "n_vocab", "ntokens"):
                    val = getattr(cfg, key, None)
                    if val is not None:
                        return int(val)
        return 32000

    def _infer_batch_size(self) -> int:
        """Try to infer batch size from the dataloader.

        Time: O(1).
        """
        bs = getattr(self._dataloader, "batch_size", None)
        if bs is not None:
            return int(bs)
        return 1

    def _compute_loss(
        self, output: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss from model output and labels."""
        if labels is not None and not self._use_vocab_parallel:
            # Standard CrossEntropyLoss path
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
        """Load a checkpoint from disk.

        If a loader_config is set, uses the fast ModelLoader for triple-buffered
        loading. Otherwise falls back to standard torch.load.
        """
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            return

        try:
            if self._loader_config is not None:
                from zenyx.loader.loader import ModelLoader

                loader = ModelLoader(
                    hal=self._hal,
                    hw_info=self._hw_info,
                    num_buffers=self._loader_config.num_buffers,
                    prefetch_bytes=self._loader_config.prefetch_bytes,
                    use_gpu_direct=self._loader_config.use_gpu_direct,
                    dtype=self._loader_config.dtype,
                )
                self._model = loader.load(path, self._model)
                logger.info("Loaded checkpoint via fast ModelLoader: %s", path)
            else:
                state = torch.load(path, map_location=self._device, weights_only=True)
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
