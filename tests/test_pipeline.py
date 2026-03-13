"""Tests for pipeline.py — BraidedPipeline schedule generation and execution."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from zenyx.train.pipeline import BraidedPipeline, ScheduleStep, StepAction


# ---------------------------------------------------------------------------
# Schedule generation tests
# ---------------------------------------------------------------------------


class TestBraidedScheduleGeneration:
    """Verify schedule structure and correctness invariants."""

    def test_smoke_instantiation(self) -> None:
        bp = BraidedPipeline(num_stages=2, num_microbatches=4, num_devices=2)
        assert bp is not None

    def test_minimal_case(self) -> None:
        """num_stages=1, num_microbatches=1 — smallest valid schedule."""
        bp = BraidedPipeline(num_stages=1, num_microbatches=1, num_devices=1)
        schedule = bp.generate_schedule()
        assert len(schedule) > 0
        # All stages covered: exactly stage 0
        stages = {s.stage_id for s in schedule}
        assert stages == {0}
        # Exactly 1 microbatch
        mbs = {s.microbatch_id for s in schedule}
        assert mbs == {0}

    def test_typical_case_no_duplicate_stage_in_microbatch_step(self) -> None:
        """4 stages, 8 microbatches — no duplicate stage in a single
        microbatch forward or backward."""
        bp = BraidedPipeline(num_stages=4, num_microbatches=8, num_devices=4)
        schedule = bp.generate_schedule()
        assert len(schedule) > 0

        # For each (microbatch, action) pair where action is FORWARD or BACKWARD,
        # each stage appears at most once.
        seen: dict[tuple[int, StepAction], set[int]] = defaultdict(set)
        for step in schedule:
            if step.action is StepAction.COMM:
                continue  # COMM steps appear in both forward and backward phases
            key = (step.microbatch_id, step.action)
            assert step.stage_id not in seen[key], (
                f"Duplicate stage {step.stage_id} for mb={step.microbatch_id}, "
                f"action={step.action}"
            )
            seen[key].add(step.stage_id)

        # All stages covered in forward and backward for every microbatch.
        for mb in range(8):
            fwd_stages = seen[(mb, StepAction.FORWARD)]
            bwd_stages = seen[(mb, StepAction.BACKWARD)]
            assert fwd_stages == set(range(4))
            assert bwd_stages == set(range(4))

    def test_all_stages_covered(self) -> None:
        """Every stage appears in at least one forward step."""
        bp = BraidedPipeline(num_stages=4, num_microbatches=8, num_devices=4)
        schedule = bp.generate_schedule()
        fwd_stages = {
            s.stage_id for s in schedule if s.action is StepAction.FORWARD
        }
        assert fwd_stages == set(range(4))

    def test_unique_step_ids(self) -> None:
        """All step_ids must be unique."""
        bp = BraidedPipeline(num_stages=3, num_microbatches=5, num_devices=3)
        schedule = bp.generate_schedule()
        ids = [s.step_id for s in schedule]
        assert len(ids) == len(set(ids))
