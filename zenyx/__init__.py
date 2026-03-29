"""ZENYX 2.0 — Phase 1 Foundation

A production-grade distributed LLM training runtime with:
- Clean separation of concerns (runtime, distributed, memory, compiler)
- FSDP-based distributed training
- Selective activation checkpointing
- Graph capture for future torch.compile integration
- Offload policy framework

Phase 1 is the foundation. Future phases will add:
- Pipeline parallelism
- Async execution overlap
- Advanced memory tiering (NVMe, KV cache)
- Advanced quantization (FP8, INT4)

Usage::

    import torch
    import torch.nn as nn
    from zenyx.distributed import FSDPWrapper
    from zenyx.memory import ActivationManager
    from zenyx.runtime import Scheduler

    model = nn.TransformerEncoderLayer(d_model=768, nhead=12)
    batch = torch.randn(32, 100, 768)

    # Wrap for distributed training
    fsdp = FSDPWrapper(model, world_size=2, mixed_precision="fp16")
    model = fsdp.wrap()

    # Add activation checkpointing
    ActivationManager.hook_into_model(model, use_checkpoint=True)

    # Set up training
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = Scheduler(accumulation_steps=1)

    # Training loop
    output = scheduler.forward(model, batch)
    loss = output.mean()
    scheduler.backward(loss, optimizer)
"""

from __future__ import annotations

__version__ = "2.0.0-phase1"

__all__ = [
    "__version__",
    # Runtime
    "Scheduler",
    "ExecutionPlan",
    "ExecutionGraph",
    "ExecutionGraphBuilder",
    "OpNode",
    "OpType",
    # Distributed
    "FSDPWrapper",
    # Memory
    "ActivationManager",
    # Compiler
    "OffloadManager",
    "OffloadPolicy",
    "make_offload_policy",
    # Utils
    "setup_logging",
    "get_logger",
]

import logging

logger = logging.getLogger("zenyx")

# Import all public components
from zenyx.compiler import (
    ExecutionGraph,
    OffloadManager,
    OffloadPolicy,
    make_offload_policy,
)
from zenyx.distributed import FSDPWrapper
from zenyx.memory import ActivationManager
from zenyx.runtime import ExecutionGraphBuilder, ExecutionPlan, OpNode, Scheduler
from zenyx.utils import get_logger, setup_logging

# Set up default logging
_logger = setup_logging(level=logging.INFO)
logger = _logger
