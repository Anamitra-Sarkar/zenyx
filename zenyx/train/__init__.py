"""Zenyx training subsystem — mixed precision, pipeline, checkpointing, loop."""

from __future__ import annotations

from zenyx.train.checkpoint import AsyncCheckpointer
from zenyx.train.mixed_prec import (
    FP8ActivationStorage,
    FP8CheckpointFunction,
    fp8_checkpoint,
)
from zenyx.train.pipeline import BraidedPipeline, ScheduleStep, StepAction
from zenyx.train.trainer import Trainer, train
from zenyx.train.lr_schedule import CosineWithWarmup
from zenyx.train.grad_scaler import ZenyxGradScaler
from zenyx.train.distributed_setup import (
    auto_init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    cleanup,
)
from zenyx.train.activation_checkpoint import (
    CheckpointedBlock,
    selective_checkpoint_wrapper,
)

# Phase 7: KV Cache Tiering
from zenyx.train.kv_cache_tier import (
    BeladyKVCacheManager,
    T0_KV_BUDGET_BYTES,
    validate_bandwidth_corrected,
    validate_bandwidth_original,
)

# Phase 8: FP8 KV Quantization
from zenyx.train.fp8_kv import (
    quantize_kv_fp8,
    dequantize_kv,
    smooth_swiglu_scale,
    GradientMonitor,
)

# Phase 9: Dynamic Ring Curriculum
from zenyx.train.ring_curriculum import (
    RingCurriculumManager,
    CurriculumConfig,
    compute_reshard_cost_optimistic,
    compute_reshard_cost_pessimistic,
)

__all__ = [
    # mixed_prec
    "FP8ActivationStorage",
    "FP8CheckpointFunction",
    "fp8_checkpoint",
    # pipeline
    "BraidedPipeline",
    "ScheduleStep",
    "StepAction",
    # checkpoint
    "AsyncCheckpointer",
    # trainer (Phase 4)
    "Trainer",
    "train",
    # lr_schedule
    "CosineWithWarmup",
    # grad_scaler
    "ZenyxGradScaler",
    # distributed_setup
    "auto_init_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "cleanup",
    # activation_checkpoint
    "CheckpointedBlock",
    "selective_checkpoint_wrapper",
    # Phase 7: KV Cache Tiering
    "BeladyKVCacheManager",
    "T0_KV_BUDGET_BYTES",
    "validate_bandwidth_corrected",
    "validate_bandwidth_original",
    # Phase 8: FP8 KV Quantization
    "quantize_kv_fp8",
    "dequantize_kv",
    "smooth_swiglu_scale",
    "GradientMonitor",
    # Phase 9: Dynamic Ring Curriculum
    "RingCurriculumManager",
    "CurriculumConfig",
    "compute_reshard_cost_optimistic",
    "compute_reshard_cost_pessimistic",
]


# Legacy submodule — wrap() from loop.py moved here for backward compat
class legacy:  # noqa: N801
    """Legacy training functions from loop.py (deprecated)."""

    @staticmethod
    def wrap(*args, **kwargs):  # type: ignore[no-untyped-def]
        """Deprecated. Use zenyx.train.trainer.Trainer instead."""
        from zenyx.train.loop import wrap as _legacy_wrap
        return _legacy_wrap(*args, **kwargs)
