"""Shared allocator constants that must stay in sync across subsystems."""
from __future__ import annotations

# FIX: Centralize the pipeline prefetch horizon used by TierAllocator and feasibility checks.
PIPELINE_DEPTH_STEPS: int = 100
