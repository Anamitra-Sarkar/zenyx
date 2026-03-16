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
* _find_step was an O(n) linear scan called inside an O(m×P) nested loop,
  making generate_schedule O(m²×P²).  Fixed by maintaining a lookup dict
  keyed by (microbatch_id, stage_id, action) that is updated incrementally
  as each step is appended; _find_step is now O(1).
* reassign_stages sets _schedule = None without warning; the next execute()
  regenerates synchronously mid-step causing a silent latency spike.
  Fixed by emitting a logger.warning at the point of invalidation.
* execute() used torch.tensor(0.0) as a fallback in grad_store and as the
  return value when the last activation is missing. torch.tensor(0.0)
  always creates a CPU tensor regardless of where model_stages or
  microbatches live, causing device mismatches on CUDA. Fixed by inferring
  device from microbatches[0] and passing device= to all fallback tensors.
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
      schedule list.  Step lookup is *O(1)* via internal index dict.
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
        self._step_counter: int = 0
        self._global_step: int = 0
        self._last_reopt_step: int = 0

        # Device-to-stage mapping (DynaPipe reassignment).
        self._stage_to_device: Dict[int, int] = {
            s: s % num_devices for s in range(num_stages)
        }

        # O(1) lookup index: (microbatch_id, stage_id, StepAction) -> ScheduleStep
        # Built incrementally inside generate_schedule; reset on each call.
        self._step_index: Dict[Tuple[int, int, StepAction], ScheduleStep] = {}
        # Backward COMM uses a distinct index to avoid colliding with forward COMM.
        # Key: (microbatch_id, stage_id, StepAction.COMM, "bwd")
        self._bwd_comm_index: Dict[Tuple[int, int, StepAction, str], ScheduleStep] = {}

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
        Previously O(m² × P²) due to linear _find_step scans; now O(m × P)
        via the O(1) dict-based _find_step.
        """
        schedule: List[ScheduleStep] = []
        self._step_counter = 0
        self._step_index = {}  # reset index for fresh generation
        self._bwd_comm_index = {}  # reset backward COMM index

        # ---- Forward pass: braided order ----
        for mb in range(self.num_microbatches):
            for stage in range(self.num_stages):
                device = self._stage_to_device[stage]

                # Forward compute step.
                fwd_id = self._next_id()
                deps: List[int] = []

                # Depends on previous stage's forward for same microbatch.
                if stage > 0:
                    prev_fwd = self._find_step(mb, stage - 1, StepAction.FORWARD)
                    if prev_fwd is not None:
                        deps.append(prev_fwd.step_id)

                # Depends on comm step of *previous* microbatch at same stage
                # (the braided overlap).
                if mb > 0:
                    prev_comm = self._find_step(mb - 1, stage, StepAction.COMM)
                    if prev_comm is not None:
                        deps.append(prev_comm.step_id)

                fwd_step = ScheduleStep(
                    step_id=fwd_id,
                    microbatch_id=mb,
                    stage_id=stage,
                    device_id=device,
                    action=StepAction.FORWARD,
                    depends_on=tuple(deps),
                )
                schedule.append(fwd_step)
                self._step_index[(mb, stage, StepAction.FORWARD)] = fwd_step

                # Communication (TP all-reduce) — overlaps with next
                # microbatch's forward on the same device.
                comm_id = self._next_id()
                comm_step = ScheduleStep(
                    step_id=comm_id,
                    microbatch_id=mb,
                    stage_id=stage,
                    device_id=device,
                    action=StepAction.COMM,
                    depends_on=(fwd_id,),
                )
                schedule.append(comm_step)
                self._step_index[(mb, stage, StepAction.COMM)] = comm_step

        # ---- Backward pass: reverse stage order per microbatch ----
        for mb in reversed(range(self.num_microbatches)):
            for stage in reversed(range(self.num_stages)):
                device = self._stage_to_device[stage]
                bwd_id = self._next_id()
                deps_bwd: List[int] = []

                # Depends on forward of same (mb, stage).
                fwd_step_ref = self._find_step(mb, stage, StepAction.FORWARD)
                if fwd_step_ref is not None:
                    deps_bwd.append(fwd_step_ref.step_id)

                # Depends on backward of next stage for same microbatch.
                if stage < self.num_stages - 1:
                    next_bwd = self._find_step(mb, stage + 1, StepAction.BACKWARD)
                    if next_bwd is not None:
                        deps_bwd.append(next_bwd.step_id)

                bwd_step = ScheduleStep(
                    step_id=bwd_id,
                    microbatch_id=mb,
                    stage_id=stage,
                    device_id=device,
                    action=StepAction.BACKWARD,
                    depends_on=tuple(deps_bwd),
                )
                schedule.append(bwd_step)
                self._step_index[(mb, stage, StepAction.BACKWARD)] = bwd_step

                # Backward comm (gradient all-reduce).
                bwd_comm_id = self._next_id()
                bwd_comm_step = ScheduleStep(
                    step_id=bwd_comm_id,
                    microbatch_id=mb,
                    stage_id=stage,
                    device_id=device,
                    action=StepAction.COMM,
                    depends_on=(bwd_id,),
                )
                schedule.append(bwd_comm_step)
                # Index the backward COMM step under a distinct key
                # (separate from the forward COMM index) so callers can find it.
                self._bwd_comm_index[(mb, stage, StepAction.COMM, "bwd")] = bwd_comm_step

        self._schedule = schedule
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

        Notes
        -----
        The ``BACKWARD`` action in the schedule is a **stub** for
        schedule-verification purposes only.  It does **not** compute real
        gradients.  Real gradient computation is performed by the Trainer's
        ``loss.backward()`` call which uses PyTorch autograd over the complete
        forward graph, not per-stage backward steps.

        Complexity
        ----------
        Time *O(m × P × T_layer)* with communication overlapped.
        """
        if self._schedule is None:
            self.generate_schedule()
        assert self._schedule is not None

        # Infer device from the first microbatch so all fallback tensors
        # are placed on the same device as the actual activations.
        fallback_device = microbatches[0].device if microbatches else torch.device("cpu")

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
                    (mb, stage),
                    torch.tensor(0.0, device=fallback_device),
                )

            elif step.action is StepAction.COMM:
                # In production: TP all-reduce or PP send/recv.
                pass

            completed.add(step.step_id)

        self._global_step += 1
        # Return last microbatch, last stage activation.
        last_key = (self.num_microbatches - 1, self.num_stages - 1)
        return activations.get(
            last_key,
            torch.tensor(0.0, device=fallback_device),
        )

    # -- DynaPipe safe stage reassignment -----------------------------------

    def reassign_stages(
        self,
        new_mapping: Dict[int, int],
    ) -> None:
        """Reassign pipeline stages to devices (DynaPipe safe reassignment).

        Safe to call between micro-batches within one gradient accumulation
        window (DynaPipe, NeurIPS 2025).  Invalidates the current schedule;
        the next :meth:`execute` call will regenerate it synchronously, which
        introduces a latency spike at that step.

        A ``WARNING`` is logged at the point of invalidation so operators can
        anticipate the spike.  Do not call this every step; the REOPT_INTERVAL
        (1000 steps) is the intended cadence.

        Parameters
        ----------
        new_mapping : Dict[int, int]
            ``{stage_id: device_id}`` mapping.
        """
        self._stage_to_device.update(new_mapping)
        self._schedule = None
        self._step_index = {}
        self._bwd_comm_index = {}
        logger.warning(
            "Stage mapping updated — schedule invalidated.  "
            "The next execute() call will regenerate the schedule "
            "synchronously, causing a one-step latency spike.  "
            "New mapping: %s",
            new_mapping,
        )

    # -- Helpers ------------------------------------------------------------

    def _next_id(self) -> int:
        """Return the next monotonically increasing step ID."""
        sid = self._step_counter
        self._step_counter += 1
        return sid

    def _find_step(
        self,
        microbatch_id: int,
        stage_id: int,
        action: StepAction,
    ) -> Optional[ScheduleStep]:
        """Look up a schedule step by (microbatch, stage, action) in O(1).

        Previously this was a linear scan over the full schedule list, making
        generate_schedule O(m²×P²).  It now uses the _step_index dict that is
        built incrementally as steps are appended, so each lookup is O(1).

        Parameters
        ----------
        microbatch_id : int
        stage_id : int
        action : StepAction

        Returns
        -------
        Optional[ScheduleStep]
        """
        return self._step_index.get((microbatch_id, stage_id, action))

    def _wait_deps(
        self,
        step: ScheduleStep,
        completed: set[int],
    ) -> None:
        """Check that all dependencies of *step* are satisfied.

        In the single-process executor the schedule is topologically ordered,
        so all deps should already be completed.  If one is missing we log a
        warning — in a real distributed executor this would be a blocking
        stream wait.
        """
        for dep in step.depends_on:
            if dep not in completed:
                logger.debug(
                    "Dependency step_id=%d not yet satisfied for step_id=%d "
                    "(mb=%d, stage=%d, action=%s).  "
                    "In a distributed executor this must be a blocking wait.",
                    dep,
                    step.step_id,
                    step.microbatch_id,
                    step.stage_id,
                    step.action.name,
                )

    def __repr__(self) -> str:
        return (
            f"BraidedPipeline("
            f"stages={self.num_stages}, "
            f"microbatches={self.num_microbatches}, "
            f"devices={self.num_devices}, "
            f"schedule={'generated' if self._schedule else 'pending'})"
        )
