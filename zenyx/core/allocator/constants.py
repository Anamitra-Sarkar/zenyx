"""Shared allocator constants that must stay in sync across subsystems."""
from __future__ import annotations

# Analytical OOM-proof feasibility bound used by check_feasibility():
#   F_compute ≤ FEASIBILITY_PIPELINE_DEPTH × max(B_01, B_12)
# This is a mathematical proof parameter, NOT a runtime prefetch window size.
# It represents the number of pipeline steps over which bandwidth must keep up
# with compute throughput to guarantee the allocator never runs out of memory.
FEASIBILITY_PIPELINE_DEPTH: int = 100

# Runtime prefetch lookahead window for TierAllocator.
# At each step(), the allocator promotes blocks needed within the next
# PREFETCH_WINDOW_OPS operations from T1/T2 → T0 asynchronously.
# Keep small: a large window causes speculative over-prefetching and
# can exhaust T0 GPU memory before the ops that need those blocks run.
PREFETCH_WINDOW_OPS: int = 3
