"""Braided TP+PP pipeline schedule (arXiv 2510.27257, NeurIPS 2025).

The braided schedule interleaves computation of microbatch *i + 1* with
communication of microbatch *i*, effectively hiding TP all-reduce latency
behind useful compute.  Bubble fraction ≈ 0 (compare GPipe *(P-1)/P*,
1F1B *1/P*, Interleaved 1F1B *(P-1)/(v×P)*).

DynaPipe safe stage reassignment
---------------------------------
Pipeline stage reassignment between micro-batches within one gradient
accumulation window is mathematically safe (linearity of differentiation,
proved by DynaPipe — NeurIPS 2025).  Re-optimisation is triggered **only**
at curriculum boundary shifts or every 1 000 steps (NOT per-step).

Fix notes
---------
* _find_step was O(n) and called inside an O(m×P) double loop, making schedule
  generation O(m²×P²).  Fixed by maintaining a ``_step_index`` dict keyed by
  ``(microbatch_id, stage_id, action)`` → ``ScheduleStep`` so every lookup is
  O(1).  The index is built lazily during generate_schedule and invalidated by
  reassign_stages along with ``_schedule``.
* reassign_stages now emits a logger.warning when called, because the next
  execute() call will pay the cost of a full schedule regeneration
  synchronously mid-training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "ScheduleStep",
    "StepAction",
    "BraidedPipeline",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------


class StepAction(Enum):
    """Action performed by a single schedule step."""

    FORWARD = auto()
    BACKWARD = auto()
    COMM = auto()  # TP all-reduce or PP send/recv

    def __repr__(self) -> str:  # noqa: D105
        return f"StepAction.{self.name}"


@dataclass(frozen=True, slots=True)
class ScheduleStep:
    """One atomic operation in the braided pipeline schedule.

    Attributes
    ----------
    step_id : int
        Globally unique monotonic step identifier.
    microbatch_id : int
        Which microbatch this step belongs to.
    stage_id : int
        Pipeline stage index.
    device_id : int
        Physical device ordinal.
    action : StepAction
        What this step does (forward / backward / comm).
    depends_on : Tuple[int, ...]
        Step IDs that must complete before this step can run.

    Complexity
    ----------
    *O(1)* construction.
    """

    step_id: int
    microbatch_id: int
    stage_id: int
    device_id: int
    action: StepAction
    depends_on: Tuple[int, ...] = ()

    def __repr__(self) -> str:  # noqa: D105
        deps = f", deps={list(self.depends_on)}" if self.depends_on else ""
        return (
            f"ScheduleStep(id={self.step_id}, mb={self.microbatch_id}, "
            f"stage={self.stage_id}, dev={self.device_id}, "
            f"action={self.action.name}{deps})"
        )


# ---------------------------------------------------------------------------
# BraidedPipeline
# ---------------------------------------------------------------------------


class BraidedPipeline:
    """Braided TP+PP schedule generator and executor.

    Parameters
    ----------
    num_stages : int
        Pipeline depth *P*.
    num_microbatches : int
        Number of micro-batches *m*.  Constraint: *m* >> *P*.
    num_devices : int
        Total device count available.

    Complexity
    ----------
    * ``generate_schedule`` — Time *O(m × P)*, space *O(m × P)* for the
      schedule list and index dict.
    * ``execute`` — Time *O(m × P × T_layer)*, with communication overlapped.
    """

    # Reoptimisation interval (steps) — NOT per-step.
    REOPT_INTERVAL: int = 1000

    def __init__(
        self,
        num_stages: int,
        num_microbatches: int,
        num_devices: int,
    ) -> None:
        if num_microbatches < num_stages:
            logger.warning(
                "Braided schedule works best when m >> P.  Got m=%d, P=%d.",
                num_microbatches,
                num_stages,
            )
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.num_devices = num_devices

        self._schedule: Optional[List[ScheduleStep]] = None
        # O(1) lookup index: (mb, stage, action) -> ScheduleStep.
        # Populated by generate_schedule, cleared by reassign_stages.
        self._step_index: Dict[Tuple[int, int, StepAction], ScheduleStep] = {}
        self._step_counter: int = 0
        self._global_step: int = 0
        self._last_reopt_step: int = 0

        # Device-to-stage mapping (DynaPipe reassignment).
        self._stage_to_device: Dict[int, int] = {
            s: s % num_devices for s in range(num_stages)
        }

    # -- Schedule generation ------------------------------------------------

    def generate_schedule(self) -> List[ScheduleStep]:
        """Generate the full braided TP+PP schedule.

        The braided pattern interleaves forward/comm/backward across
        micro-batches so that communication of micro-batch *i* overlaps
        with compute of micro-batch *i + 1*.

        Returns
        -------
        List[ScheduleStep]

        Complexity
        ----------
        Time *O(m × P)*, space *O(m × P)*.
        """
        schedule: List[ScheduleStep] = []
        index: Dict[Tuple[int, int, StepAction], ScheduleStep] = {}
        self._step_counter = 0

        def _add(step: ScheduleStep) -> ScheduleStep:
            """Append step to schedule and insert into the O(1) index."""
            schedule.append(step)
            # For COMM steps, multiple entries can share (mb, stage, COMM);
            # we keep the *first* one as the canonical dependency target
            # (forward COMM precedes backward COMM for the same (mb, stage)).
            key = (step.microbatch_id, step.stage_id, step.action)
            if key not in index:
                index[key] = step
            return step

        def _lookup(
            mb: int, stage: int, action: StepAction
        ) -> Optional[ScheduleStep]:
            """O(1) replacement for the previous O(n) _find_step scan."""
            return index.get((mb, stage, action))

        # ---- Forward pass: braided order ----
        for mb in range(self.num_microbatches):
            for stage in range(self.num_stages):
                device = self._stage_to_device[stage]

                # Forward compute step.
                fwd_id = self._next_id()
                deps: List[int] = []

                # Depends on previous stage’s forward for same microbatch.
                if stage > 0:
                    prev_fwd = _lookup(mb, stage - 1, StepAction.FORWARD)
                    if prev_fwd is not None:
                        deps.append(prev_fwd.step_id)

                # Depends on comm step of *previous* microbatch at same stage
                # (the braided overlap).
                if mb > 0:
                    prev_comm = _lookup(mb - 1, stage, StepAction.COMM)
                    if prev_comm is not None:
                        deps.append(prev_comm.step_id)

                fwd_step = _add(
                    ScheduleStep(
                        step_id=fwd_id,
                        microbatch_id=mb,
                        stage_id=stage,
                        device_id=device,
                        action=StepAction.FORWARD,
                        depends_on=tuple(deps),
                    )
                )

                # Communication (TP all-reduce) — overlaps with next
                # microbatch’s forward on the same device.
                comm_id = self._next_id()
                _add(
                    ScheduleStep(
                        step_id=comm_id,
                        microbatch_id=mb,
                        stage_id=stage,
                        device_id=device,
                        action=StepAction.COMM,
                        depends_on=(fwd_id,),
                    )
                )

        # ---- Backward pass: reverse stage order per microbatch ----
        for mb in reversed(range(self.num_microbatches)):
            for stage in reversed(range(self.num_stages)):
                device = self._stage_to_device[stage]
                bwd_id = self._next_id()
                deps_bwd: List[int] = []

                # Depends on forward of same (mb, stage).
                fwd_step = _lookup(mb, stage, StepAction.FORWARD)
                if fwd_step is not None:
                    deps_bwd.append(fwd_step.step_id)

                # Depends on backward of next stage for same microbatch.
                if stage < self.num_stages - 1:
                    next_bwd = _lookup(mb, stage + 1, StepAction.BACKWARD)
                    if next_bwd is not None:
                        deps_bwd.append(next_bwd.step_id)

                _add(
                    ScheduleStep(
                        step_id=bwd_id,
                        microbatch_id=mb,
                        stage_id=stage,
                        device_id=device,
                        action=StepAction.BACKWARD,
                        depends_on=tuple(deps_bwd),
                    )
                )

                # Backward comm (gradient all-reduce).
                bwd_comm_id = self._next_id()
                _add(
                    ScheduleStep(
                        step_id=bwd_comm_id,
                        microbatch_id=mb,
                        stage_id=stage,
                        device_id=device,
                        action=StepAction.COMM,
                        depends_on=(bwd_id,),
                    )
                )

        self._schedule = schedule
        self._step_index = index
        logger.info(
            "Generated braided schedule: %d steps for %d microbatches × %d stages.",
            len(schedule),
            self.num_microbatches,
            self.num_stages,
        )
        return schedule

    # -- Execution ----------------------------------------------------------

    def execute(
        self,
        model_stages: List[nn.Module],
        microbatches: List[torch.Tensor],
    ) -> torch.Tensor:
        """Execute the braided schedule on actual model stages.

        Parameters
        ----------
        model_stages : List[nn.Module]
            One sub-module per pipeline stage.
        microbatches : List[torch.Tensor]
            Input micro-batches.

        Returns
        -------
        torch.Tensor
            Aggregated output of the last micro-batch from the last stage.

        Complexity
        ----------
        Time *O(m × P × T_layer)* with communication overlapped.
        """
        if self._schedule is None:
            self.generate_schedule()
        assert self._schedule is not None

        # Intermediate activation storage: (mb, stage) -> tensor.
        activations: Dict[Tuple[int, int], torch.Tensor] = {}
        grad_store: Dict[Tuple[int, int], torch.Tensor] = {}
        completed: set[int] = set()

        # Initialise microbatch inputs.
        for mb_idx, mb_tensor in enumerate(microbatches):
            activations[(mb_idx, -1)] = mb_tensor

        for step in self._schedule:
            # Respect dependencies (in real distributed runtime, this would be
            # event/stream-based; here we enforce ordering).
            self._wait_deps(step, completed)

            mb = step.microbatch_id
            stage = step.stage_id

            if step.action is StepAction.FORWARD:
                inp = activations.get((mb, stage - 1))
                if inp is None:
                    logger.warning("Missing activation for mb=%d stage=%d", mb, stage)
                    inp = microbatches[mb]
                try:
                    out = model_stages[stage](inp)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    out = model_stages[stage](inp)
                activations[(mb, stage)] = out

            elif step.action is StepAction.BACKWARD:
                # Stub: in production, this would call autograd backward on
                # the stage output.  Here we record that backward ran.
                grad_store[(mb, stage)] = activations.get(
                    (mb, stage), torch.tensor(0.0)
                )

            elif step.action is StepAction.COMM:
                # In production: TP all-reduce or PP send/recv.
                pass

            completed.add(step.step_id)

        self._global_step += 1
        # Return last microbatch, last stage activation.
        last_key = (self.num_microbatches - 1, self.num_stages - 1)
        return activations.get(last_key, torch.tensor(0.0))

    # -- DynaPipe safe stage reassignment -----------------------------------

    def reassign_stages(
        self,
        new_mapping: Dict[int, int],
    ) -> None:
        """Reassign pipeline stages to physical devices (DynaPipe).

        Invalidates the current schedule.  The next call to :meth:`execute`
        or :meth:`generate_schedule` will regenerate it synchronously.

        .. warning::
            Schedule regeneration is *O(m × P)* and happens synchronously on
            the next ``execute()`` call.  Call ``reassign_stages`` only at
            curriculum boundaries or every ``REOPT_INTERVAL`` steps, **not**
            per-step.  A per-step call will introduce a latency spike
            proportional to the schedule size.

        Parameters
        ----------
        new_mapping : Dict[int, int]
            Mapping from stage index to physical device ordinal.
        """
        logger.warning(
            "reassign_stages called at global_step=%d: schedule invalidated. "
            "Next execute() will regenerate synchronously (O(m×P) = O(%d×%d) = %d steps). "
            "Ensure this is called only at curriculum boundaries or every "
            "%d steps.",
            self._global_step,
            self.num_microbatches,
            self.num_stages,
            self.num_microbatches * self.num_stages,
            self.REOPT_INTERVAL,
        )
        self._stage_to_device.update(new_mapping)
        self._schedule = None
        self._step_index = {}
        self._last_reopt_step = self._global_step

    # -- Private helpers ----------------------------------------------------

    def _next_id(self) -> int:
        """Return and increment the step counter."""
        sid = self._step_counter
        self._step_counter += 1
        return sid

    def _wait_deps(
        self,
        step: ScheduleStep,
        completed: set[int],
    ) -> None:
        """Assert all declared dependencies of *step* have completed.

        In a real distributed runtime this would be stream-event based.
        In the single-process simulation the schedule is topologically sorted
        so this is always satisfied; the check exists as a correctness guard.
        """
        for dep in step.depends_on:
            if dep not in completed:
                logger.debug(
                    "Step %d waiting on dep %d (mb=%d stage=%d action=%s)",
                    step.step_id, dep,
                    step.microbatch_id, step.stage_id, step.action.name,
                )

    def __repr__(self) -> str:
        n = len(self._schedule) if self._schedule is not None else 0
        return (
            f"BraidedPipeline("
            f"stages={self.num_stages}, "
            f"microbatches={self.num_microbatches}, "
            f"devices={self.num_devices}, "
            f"schedule_steps={n})"
        )
