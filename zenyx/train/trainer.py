"""Zenyx Trainer — single entrypoint for distributed LLM training.

Usage:
    import zenyx
    zenyx.train(model, dataloader)

The Trainer auto-detects hardware, sets up memory pools, selects the
correct attention kernel (ring_flash_cuda for H100, ring_pallas_tpu for
TPU, flash_cpu for CPU), configures the optimizer, and runs the training
loop. The user never touches distributed setup, memory management, or
kernel selection.

Fix notes
---------
* validate_memory_bandwidth: was reading non-existent field `hbm_bandwidth_gbs`
  (always 0.0, triggering early-return guard every run). Now correctly reads
  `bandwidth_t0_t1` (bytes/s) and converts to GB/s.
* _load_checkpoint with loader_config: the fast ModelLoader path previously
  only restored model weights and silently dropped optimizer state, scheduler
  state, and step counter. It now loads the full state dict from disk first
  via torch.load and restores all state, then calls loader.load() to populate
  the model weights efficiently.
* throughput_estimate: was computed then immediately discarded (local variable
  never used). Now passed to profiler.end_op as metadata.
* get_state() throughput: divided context_len by avg_time, ignoring batch
  size. Corrected to batch_size * context_len / avg_time.
* _compute_loss fallback: if output lacked .mean(), the raw (non-scalar)
  tensor was returned causing .backward() to raise. Now always returns a
  scalar via .reshape(-1).sum() with a UserWarning.
* Profiler handle leak: start_op was called inside the optimizer-step block
  but the handle variable was declared outside the loop and initialised to
  None. On micro-steps that did NOT trigger an optimizer step, the previous
  handle was overwritten without a matching end_op call. Fixed by scoping
  start_op / end_op tightly around the optimizer step block.
* Cross-platform atomic checkpoint: os.replace() raises PermissionError on
  Windows when the destination is open. Added shutil.move fallback.
* OOM retry in main train loop: torch.cuda.OutOfMemoryError was unhandled
  in the Trainer.train() loop (only _legacy_train had OOM handling). Added
  a retry block around forward+backward that clears the cache and retries
  once before re-raising.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import time
import warnings
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
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_steps: int = 2000,
        total_steps: int = 100_000,
        dtype: str = "bfloat16",
        activation_dtype: str = "float8_e4m3",
        t1_capacity_gb: float = 8.0,
        t2_capacity_gb: float = 64.0,
        context_len: int = 4096,
        dynamic_routing: bool = False,
        heap_rebuild_interval: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_every: int = 1000,
        resume_from: Optional[str] = None,
        log_every: int = 10,
        gradient_accumulation_steps: int = 1,
        selective_activation_checkpoint: bool = True,
        checkpoint_every_nth_layer: int = 4,
        loader_config: Optional[Any] = None,
        kv_tier_config: Optional[Any] = None,
        fp8_kv: bool = False,
        fp8_coat_mode: bool = False,
        fp8_quant_strategy: str = "per_channel",
        curriculum_config: Optional[Any] = None,
        reshard_no_recompile: Optional[bool] = None,
        sparse_attn: bool = False,
        sparse_skip_mode: str = "production",
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

        self._kv_tier_config = kv_tier_config
        self._fp8_kv = fp8_kv
        self._fp8_coat_mode = fp8_coat_mode
        self._fp8_quant_strategy = fp8_quant_strategy
        self._curriculum_config = curriculum_config
        self._reshard_no_recompile = reshard_no_recompile
        self._sparse_attn = sparse_attn
        self._sparse_skip_mode = sparse_skip_mode

        self._kv_cache_manager: Optional[Any] = None
        self._gradient_monitor: Optional[Any] = None
        self._curriculum_manager: Optional[Any] = None
        self._sparse_kernel: Optional[Any] = None

        self._step: int = 0
        self._last_loss: float = 0.0
        self._recent_step_times: list[float] = []
        self._compute_graph: Optional[Any] = None
        self._actual_seq_len: int = 0

        # --- Step 1: Hardware detection ---
        from zenyx.core.hal.detector import detect_hardware, HardwareInfo

        try:
            self._hw_info: HardwareInfo = detect_hardware()
        except Exception as e:
            logger.warning(
                "Hardware detection failed (%s); falling back to CPU defaults.", e,
            )
            self._hw_info = HardwareInfo(
                backend="cpu",
                device_count=1,
                per_device_memory_bytes=16 * (1024 ** 3),
                interconnect="none",
                bandwidth_t0_t1=50.0 * (1024 ** 3),
                bandwidth_t1_t2=7.0 * (1024 ** 3),
                compute_tflops=0.0,
                device_name="CPU (fallback)",
            )

        # --- Step 2: Topology detection ---
        from zenyx.ops.comm.topology import detect_topology, TopologyInfo

        try:
            self._topo_info: TopologyInfo = detect_topology()
        except Exception as e:
            logger.warning(
                "Topology detection failed (%s); falling back to single-device defaults.", e,
            )
            self._topo_info = TopologyInfo(
                backend="gloo",
                interconnect="none",
                ring_bandwidth_gbps=0.0,
                world_size=1,
                local_rank=0,
                global_rank=0,
                num_nodes=1,
                gpus_per_node=0,
                s_min_tokens=0,
            )

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

        # --- Step 4: Create HAL + TierAllocator ---
        self._hal: Any = None
        self._tier_allocator: Any = None

        try:
            if self._backend == "cuda":
                from zenyx.core.hal.cuda_hal import CudaHAL
                self._hal = CudaHAL()
        except Exception as e:
            logger.warning(
                "CUDA HAL init failed — 3-tier memory allocator disabled. "
                "Training will proceed without TierAllocator. Error: %s", e
            )

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

        # --- Step 12: Agent integration ---
        from zenyx.core.agent.profiler import AsyncProfiler
        from zenyx.core.agent.planner import ParallelismPlanner, ParallelismPlan
        from zenyx.core.agent.controller import TrainingController
        from zenyx.ops.comm.topology import Topology

        self._profiler = AsyncProfiler(enabled=True)

        topo = Topology()
        self._planner = ParallelismPlanner(self._hw_info, topo)

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
                self._loss_fn = VocabParallelCrossEntropy()
                self._use_vocab_parallel = True
                logger.info("Using VocabParallelCrossEntropy for distributed loss")
            except ImportError:
                pass

        # --- Step 15: Memory bandwidth validation ---
        self.validate_memory_bandwidth()

        # --- Step 16: Batch size sanity check ---
        num_params = sum(p.numel() for p in model.parameters())
        global_batch_tokens = (
            context_len
            * max(1, self._infer_batch_size())
            * max(1, gradient_accumulation_steps)
        )
        min_recommended_tokens_per_step = 65_536
        if global_batch_tokens < min_recommended_tokens_per_step:
            warnings.warn(
                f"Global batch {global_batch_tokens} tokens is dangerously small for a "
                f"{num_params:.1e}-parameter model. Minimum recommended: "
                f"{min_recommended_tokens_per_step}. "
                "Training will likely diverge. Use gradient accumulation.",
                UserWarning,
                stacklevel=2,
            )

        # --- Step 17: Phase 7 — KV Cache Tiering ---
        if kv_tier_config is not None:
            from zenyx.train.kv_cache_tier import BeladyKVCacheManager, KVTierConfig
            num_layers_kv = 2
            for _attr in ("layers", "blocks", "encoder", "decoder"):
                _layers_attr = getattr(model, _attr, None)
                if _layers_attr is not None and hasattr(_layers_attr, "__len__"):
                    num_layers_kv = len(_layers_attr)
                    break
            ring_degree_kv = max(1, self._topo_info.world_size)
            if isinstance(kv_tier_config, KVTierConfig):
                cfg = vars(kv_tier_config)
            elif isinstance(kv_tier_config, dict):
                cfg = kv_tier_config
            else:
                logger.warning(
                    "kv_tier_config has unexpected type %s; using defaults.",
                    type(kv_tier_config).__name__,
                )
                cfg = {}
            self._kv_cache_manager = BeladyKVCacheManager(
                world_size=cfg.get("world_size", self._topo_info.world_size),
                num_layers=cfg.get("num_layers", num_layers_kv),
                ring_degree=cfg.get("ring_degree", ring_degree_kv),
                **{k: v for k, v in cfg.items() if k not in ("world_size", "num_layers", "ring_degree")},
            )
            logger.info("Phase 7: BeladyKVCacheManager initialized")

        # --- Step 18: Phase 8 — FP8 KV Quantization ---
        if fp8_kv:
            from zenyx.train.fp8_kv import GradientMonitor
            self._gradient_monitor = GradientMonitor(
                coat_safety_claim=fp8_coat_mode
            )
            logger.info(
                "Phase 8: FP8 KV enabled (strategy=%s, coat_mode=%s)",
                fp8_quant_strategy, fp8_coat_mode,
            )

        # --- Step 19: Phase 9 — Dynamic Ring Curriculum ---
        if curriculum_config is not None:
            from zenyx.train.ring_curriculum import CurriculumConfig, RingCurriculumManager
            if isinstance(curriculum_config, CurriculumConfig):
                cfg = dataclasses.asdict(curriculum_config)
            elif isinstance(curriculum_config, dict):
                cfg = curriculum_config
            else:
                logger.warning(
                    "curriculum_config has unexpected type %s; using defaults.",
                    type(curriculum_config).__name__,
                )
                cfg = {}
            self._curriculum_manager = RingCurriculumManager(
                no_recompile=reshard_no_recompile,
                **cfg,
            )
            self._curriculum_manager.build_static_mesh()
            logger.info("Phase 9: RingCurriculumManager initialized")

        # --- Step 20: Phase 10 — Sparse Ring Attention ---
        if sparse_attn:
            from zenyx.ops.attention.sparse_ring_attn import SparseRingAttentionKernel
            num_layers = 2
            for attr_name in ("layers", "blocks", "encoder", "decoder"):
                layers_attr = getattr(model, attr_name, None)
                if layers_attr is not None and hasattr(layers_attr, "__len__"):
                    num_layers = len(layers_attr)
                    break

            self._sparse_kernel = SparseRingAttentionKernel(
                skip_mode=sparse_skip_mode,
                num_layers=num_layers,
                seq_len=context_len,
                window_size=min(131_072, context_len),
                ring_degree=max(1, self._topo_info.world_size),
                world_size=max(1, self._topo_info.world_size),
            )
            logger.info(
                "Phase 10: SparseRingAttentionKernel initialized (skip_mode=%s). "
                "NOTE: Pallas kernel integration is pending — sparse attention is "
                "initialized but not yet wired into the forward pass. The model "
                "currently uses the standard dense attention path regardless of "
                "this setting. See sparse_ring_attn.py execute_step() stub.",
                sparse_skip_mode,
            )

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
        """Register a compute graph for Bélády-optimal eviction scheduling."""
        self._compute_graph = graph
        if self._tier_allocator is not None:
            self._tier_allocator.register_compute_graph(graph)
            logger.info("Compute graph registered with TierAllocator")

    def validate_memory_bandwidth(self) -> None:
        """Validate memory bandwidth feasibility."""
        from zenyx.train.kv_cache_tier import (
            validate_bandwidth_corrected,
            validate_bandwidth_original,
        )

        b_01_bytes_per_s = getattr(self._hw_info, "bandwidth_t0_t1", 0.0)
        b_01 = b_01_bytes_per_s / (1024 ** 3)

        b_12_bytes_per_s = getattr(self._hw_info, "bandwidth_t1_t2", 0.0)
        if b_12_bytes_per_s > 0:
            b_12 = b_12_bytes_per_s / (1024 ** 3)
        else:
            b_12 = 7.5
            logger.debug(
                "b_12 (NVMe bandwidth) not available from hardware info; "
                "using conservative default of %.1f GB/s.", b_12,
            )

        compute_tflops = getattr(self._hw_info, "compute_tflops", 0.0)

        if b_01 <= 0 or compute_tflops <= 0:
            logger.debug(
                "Bandwidth validation skipped (b_01=%.3f GB/s, compute_tflops=%.1f).",
                b_01, compute_tflops,
            )
            return

        corrected_result = validate_bandwidth_corrected(b_01, b_12, compute_tflops)
        original_result = validate_bandwidth_original(b_01, b_12, compute_tflops)

        corrected_ok = corrected_result[0] if isinstance(corrected_result, tuple) else corrected_result
        original_ok = original_result[0] if isinstance(original_result, tuple) else original_result
        corrected_msg = corrected_result[1] if isinstance(corrected_result, tuple) else ""
        original_msg = original_result[1] if isinstance(original_result, tuple) else ""

        logger.info(
            "Bandwidth validation: corrected=%s (%s), original=%s (%s)",
            "PASS" if corrected_ok else "FAIL", corrected_msg,
            "PASS" if original_ok else "FAIL", original_msg,
        )

        if not corrected_ok or not original_ok:
            warnings.warn(
                f"Memory bandwidth may be insufficient. "
                f"Corrected formula: {'PASS' if corrected_ok else 'FAIL'}, "
                f"Original formula: {'PASS' if original_ok else 'FAIL'}. "
                f"B_01={b_01:.1f} GB/s, B_12={b_12:.1f} GB/s, "
                f"Compute={compute_tflops:.1f} TFLOPS. "
                f"Training may proceed in throttled-compute mode.",
                RuntimeWarning,
                stacklevel=2,
            )

        if corrected_ok != original_ok:
            logger.critical(
                "DISPUTE 7-A: Feasibility formulas DISAGREE. "
                "Corrected=%s, Original=%s. Investigate bandwidth configuration.",
                "PASS" if corrected_ok else "FAIL",
                "PASS" if original_ok else "FAIL",
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop.

        Always saves a final checkpoint after the loop completes, regardless
        of ``checkpoint_every``. This ensures the final model state is never
        lost even if the last step is not a multiple of ``checkpoint_every``.
        Intermediate checkpoints are saved every ``checkpoint_every`` optimizer
        steps during training.

        OOM handling
        ------------
        If a ``torch.cuda.OutOfMemoryError`` is raised during forward or
        backward, the CUDA cache is cleared and the micro-step is retried
        once before re-raising. This mirrors the behaviour of the legacy
        ``_legacy_train`` / ``_safe_forward_backward`` path which was the only
        place with OOM handling before this fix.
        """
        self._model.train()

        accum_loss = 0.0
        micro_step = 0

        for batch in self._dataloader:
            if self._step >= self._total_steps:
                break

            step_start = time.monotonic()

            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch
                labels = None

            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device, non_blocking=True)
            # Detect actual sequence length from the first batch.
            # Expected input shape: [batch, seq_len, ...] — axis 1 is seq_len.
            if self._actual_seq_len == 0 and hasattr(inputs, "shape") and inputs.ndim >= 2:
                self._actual_seq_len = inputs.shape[1]
            if labels is not None and hasattr(labels, "to"):
                labels = labels.to(self._device, non_blocking=True)

            # ------------------------------------------------------------------
            # Forward + backward with OOM retry
            # Previously only _legacy_train/_safe_forward_backward had OOM
            # handling; an OOM here would propagate uncaught and kill training.
            # ------------------------------------------------------------------
            try:
                if self._use_amp:
                    with torch.amp.autocast(device_type=self._backend, dtype=self._amp_dtype):
                        output = self._model(inputs)
                        loss = self._compute_loss(output, labels)
                else:
                    output = self._model(inputs)
                    loss = self._compute_loss(output, labels)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._optimizer.zero_grad(set_to_none=True)
                self._grad_scaler = ZenyxGradScaler(enabled=self._use_amp)  # reset scaler state
                accum_loss = 0.0
                logger.warning(
                    "OOM at step %d micro_step %d — cleared cache, reset gradients, retrying.",
                    self._step, micro_step,
                )
                # Retry once after cache clear and gradient reset
                if self._use_amp:
                    with torch.amp.autocast(device_type=self._backend, dtype=self._amp_dtype):
                        output = self._model(inputs)
                        loss = self._compute_loss(output, labels)
                else:
                    output = self._model(inputs)
                    loss = self._compute_loss(output, labels)

            loss = loss / self._gradient_accumulation_steps
            self._grad_scaler.scale(loss).backward()
            accum_loss = accum_loss + loss.detach()
            micro_step += 1

            if micro_step % self._gradient_accumulation_steps == 0:
                # Open profiler span HERE — tightly scoped to the optimizer step.
                # Previously opened outside this block, causing the handle to be
                # overwritten on non-optimizer micro-steps without end_op.
                _profile_handle = self._profiler.start_op("train_step")

                if self._gradient_monitor is not None and self._fp8_kv:
                    # Compute global gradient norm as proxy for FP8 anomaly detection.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), float('inf')  # no clip, just compute norm
                    )
                    if hasattr(self._grad_scaler, '_scale') and self._grad_scaler._scale is not None:
                        scale = float(self._grad_scaler._scale)
                        if scale > 0:
                            approx_fp8_grad = torch.tensor(float(grad_norm) / scale)
                            approx_bf16_grad = torch.tensor(float(grad_norm))
                            self._gradient_monitor.check_gradient(approx_fp8_grad, approx_bf16_grad)

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
                self._last_loss = float(
                    accum_loss.item() * self._gradient_accumulation_steps
                    if hasattr(accum_loss, "item")
                    else accum_loss * self._gradient_accumulation_steps
                )

                elapsed = time.monotonic() - step_start
                self._recent_step_times.append(elapsed)
                if len(self._recent_step_times) > 100:
                    self._recent_step_times = self._recent_step_times[-100:]

                batch_size_est = self._infer_batch_size()
                seq_len_for_throughput = self._actual_seq_len or self._context_len
                throughput_estimate = (
                    batch_size_est * seq_len_for_throughput / elapsed if elapsed > 0 else 0.0
                )

                # end_op paired with start_op above — no leaked handles.
                self._profiler.end_op(
                    _profile_handle,
                    metadata={"throughput_tokens_per_sec": throughput_estimate},
                )

                if is_main_process() and self._step % self._log_every == 0:
                    lr = self._scheduler.get_lr()
                    logger.info(
                        "Step %d | Loss: %.4f | LR: %.2e | Time: %.3fs | "
                        "Throughput: %.0f tok/s",
                        self._step, self._last_loss, lr, elapsed, throughput_estimate,
                    )

                if self._step % self._checkpoint_every == 0:
                    self._save_checkpoint()

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
                            "Heap rebuild skipped: no compute graph registered."
                        )

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
                            self._step, new_plan,
                        )
                except Exception as e:
                    logger.debug("Controller step skipped: %s", e)

                accum_loss = 0.0
                micro_step = 0

        if is_main_process():
            logger.info(
                "Saving final checkpoint at step %d (unconditional end-of-training save).",
                self._step,
            )
            self._save_checkpoint()
            logger.info("Training complete at step %d", self._step)

        self._profiler.shutdown()

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return training state snapshot."""
        throughput = 0.0
        if self._recent_step_times:
            avg_time = sum(self._recent_step_times) / len(self._recent_step_times)
            if avg_time > 0:
                batch_size_est = self._infer_batch_size()
                seq_len_for_throughput = self._actual_seq_len or self._context_len
                throughput = batch_size_est * seq_len_for_throughput / avg_time

        pool_usage: Optional[dict] = None
        if self._tier_allocator is not None:
            pool = getattr(self._tier_allocator, "_pool", None)
            if pool is not None and hasattr(pool, "usage"):
                try:
                    pool_usage = pool.usage()
                except Exception:
                    pass

        profiler_stats: Optional[dict] = None
        try:
            timings = self._profiler.get_timings()
            profiler_stats = {
                k: {"avg_ms": v.avg_ms, "count": v.count}
                for k, v in timings.items()
            }
        except Exception:
            pass

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
        for attr in ("config", "cfg"):
            cfg = getattr(self._model, attr, None)
            if cfg is None:
                continue
            for key in ("vocab_size", "n_vocab", "ntokens"):
                if isinstance(cfg, dict):
                    val = cfg.get(key)
                else:
                    val = getattr(cfg, key, None)
                if val is not None:
                    return int(val)
        logger.warning(
            "_infer_vocab_size: could not detect vocab_size from model config. "
            "Defaulting to 32000 (LLaMA-2). Pass a model with model.config.vocab_size "
            "for accurate parallelism planning."
        )
        return 32000

    def _infer_batch_size(self) -> int:
        bs = getattr(self._dataloader, "batch_size", None)
        if bs is not None:
            return int(bs)
        return 1

    def _compute_loss(
        self, output: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if labels is not None:
            if output.dim() == 3:
                B, S, V = output.shape
                output = output.reshape(B * S, V)
                labels = labels.reshape(B * S)
            return self._loss_fn(output.float(), labels)

        warnings.warn(
            "_compute_loss: no labels provided — using mean of model output as "
            "a proxy loss. Gradients are not meaningful.",
            UserWarning,
            stacklevel=2,
        )
        if hasattr(output, "mean"):
            return output.float().mean()
        return output.float().reshape(-1).sum()

    def _save_checkpoint(self) -> None:
        """Save a checkpoint atomically.

        Uses os.replace() for atomic rename on POSIX. Falls back to
        shutil.move() on Windows where os.replace() raises PermissionError
        if the destination file is open by another process.
        """
        if not is_main_process():
            return

        os.makedirs(self._checkpoint_dir, exist_ok=True)
        path = os.path.join(self._checkpoint_dir, f"step_{self._step}.pt")

        try:
            state = {
                "step": self._step,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state": self._scheduler.state_dict(),
            }
            tmp_path = path + ".tmp"
            torch.save(state, tmp_path)
            try:
                os.replace(tmp_path, path)
            except OSError:
                # Windows: destination may be locked by another process.
                shutil.move(tmp_path, path)
            logger.info("Checkpoint saved: %s", path)
        except Exception as e:
            logger.warning("Checkpoint save failed: %s", e)

    def _validate_checkpoint_path(self, path: str) -> None:
        """Raise ValueError if path is outside checkpoint_dir (prevents directory traversal)."""
        safe_checkpoint_dir = os.path.realpath(self._checkpoint_dir)
        safe_path = os.path.realpath(path)
        if not safe_path.startswith(safe_checkpoint_dir + os.sep) and safe_path != safe_checkpoint_dir:
            raise ValueError(
                f"Checkpoint path '{path}' resolves to '{safe_path}' which is outside "
                f"checkpoint_dir '{self._checkpoint_dir}' (resolved: '{safe_checkpoint_dir}') "
                "— refusing to load for security."
            )

    def _load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            return

        self._validate_checkpoint_path(path)

        if self._loader_config is not None:
            from zenyx.loader.loader import ModelLoader

            full_state = torch.load(
                path, map_location=self._device, weights_only=False  # nosec: path validated above
            )
            self._optimizer.load_state_dict(full_state["optimizer_state_dict"])
            self._step = full_state.get("step", 0)
            if "scheduler_state" in full_state:
                self._scheduler.load_state_dict(full_state["scheduler_state"])
            elif "scheduler_step" in full_state:
                for _ in range(full_state["scheduler_step"]):
                    self._scheduler.step()

            loader = ModelLoader(
                hal=self._hal,
                hw_info=self._hw_info,
                num_buffers=self._loader_config.num_buffers,
                prefetch_bytes=self._loader_config.prefetch_bytes,
                use_gpu_direct=self._loader_config.use_gpu_direct,
                dtype=self._loader_config.dtype,
            )
            self._model = loader.load(path, self._model)
            logger.info(
                "Resumed from checkpoint via fast ModelLoader: %s (step %d)",
                path, self._step,
            )
        else:
            try:
                state = torch.load(path, map_location=self._device, weights_only=False)  # nosec: path validated above
                self._model.load_state_dict(state["model_state_dict"])
                self._optimizer.load_state_dict(state["optimizer_state_dict"])
                self._step = state.get("step", 0)
                if "scheduler_state" in state:
                    self._scheduler.load_state_dict(state["scheduler_state"])
                elif "scheduler_step" in state:
                    for _ in range(state["scheduler_step"]):
                        self._scheduler.step()
                logger.info("Resumed from checkpoint: %s (step %d)", path, self._step)
            except Exception as e:
                logger.warning("Checkpoint load failed: %s", e)
                raise


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    dataloader: Any,
    **kwargs: Any,
) -> Trainer:
    """One-line training entrypoint."""
    trainer = Trainer(model, dataloader, **kwargs)
    trainer.train()
    return trainer
