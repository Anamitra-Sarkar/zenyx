"""Zenyx agent — Autonomous training controller.

Detects curriculum shifts (context-length changes), triggers replanning at
shift boundaries or every N steps, and tracks comprehensive training
statistics.

Complexity
----------
- ``step()`` is O(1) amortised (O(K) when replanning, K = profiled ops).
- ``get_training_stats()`` is O(1).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from zenyx.core.agent.planner import ParallelismPlan, ParallelismPlanner
from zenyx.core.agent.profiler import AsyncProfiler

logger = logging.getLogger("zenyx.agent.controller")


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class TrainingStats:
    """Comprehensive training statistics.

    Attributes
    ----------
    steps_completed : int
        Total training steps completed.
    avg_loss : float
        Running average loss.
    avg_throughput_tokens_s : float
        Average throughput in tokens/second.
    peak_memory_gb : float
        Peak GPU memory usage observed (GB).
    replan_count : int
        Number of replanning events triggered.
    total_training_time_s : float
        Wall-clock training time in seconds.
    """

    steps_completed: int = 0
    avg_loss: float = 0.0
    avg_throughput_tokens_s: float = 0.0
    peak_memory_gb: float = 0.0
    replan_count: int = 0
    total_training_time_s: float = 0.0

    def __repr__(self) -> str:
        return (
            f"TrainingStats(steps={self.steps_completed}, "
            f"avg_loss={self.avg_loss:.4f}, "
            f"throughput={self.avg_throughput_tokens_s:,.0f} tok/s, "
            f"peak_mem={self.peak_memory_gb:.2f}GB, "
            f"replans={self.replan_count}, "
            f"time={self.total_training_time_s:.1f}s)"
        )


# ── Controller ───────────────────────────────────────────────────────────


class TrainingController:
    """Autonomous training controller that triggers replanning when needed.

    Tracks loss, throughput, and memory history.  Detects curriculum shifts
    (changes in ``context_len``) and triggers the :class:`ParallelismPlanner`
    to replan at shift boundaries or every *replan_interval* steps.

    Parameters
    ----------
    planner : ParallelismPlanner
        The parallelism planner used to generate new plans.
    profiler : AsyncProfiler
        The profiler that supplies timing and memory data.
    replan_interval : int
        Number of steps between periodic replanning (default 1000).
    model_params : float
        Number of model parameters for replanning.
    vocab_size : int
        Vocabulary size for replanning.
    batch_size : int
        Global batch size for replanning.

    Time: O(1) construction.  Space: O(H) where H = history window size.
    """

    _HISTORY_WINDOW = 100  # Rolling window for averages

    def __init__(
        self,
        planner: ParallelismPlanner,
        profiler: AsyncProfiler,
        replan_interval: int = 1000,
        model_params: float = 0.0,
        vocab_size: int = 0,
        batch_size: int = 0,
    ) -> None:
        self._planner = planner
        self._profiler = profiler
        self._replan_interval = replan_interval
        self._model_params = model_params
        self._vocab_size = vocab_size
        self._batch_size = batch_size

        # Curriculum tracking
        self._last_context_len: Optional[int] = None

        # Statistics accumulators
        self._loss_history: Deque[float] = deque(maxlen=self._HISTORY_WINDOW)
        self._throughput_history: Deque[float] = deque(maxlen=self._HISTORY_WINDOW)
        self._peak_memory_gb: float = 0.0
        self._replan_count: int = 0
        self._steps_completed: int = 0
        self._start_time: float = time.monotonic()
        self._last_step_time: float = self._start_time

        # Current plan
        self._current_plan: Optional[ParallelismPlan] = None

    # ── Public API ───────────────────────────────────────────────────────

    def step(
        self,
        step_num: int,
        loss: float,
        context_len: int,
    ) -> Optional[ParallelismPlan]:
        """Process one training step and possibly trigger replanning.

        Time: O(1) amortised; O(K) when replanning occurs.  Space: O(1).

        Parameters
        ----------
        step_num : int
            Current training step number.
        loss : float
            Loss value for this step.
        context_len : int
            Context length used in this step.

        Returns
        -------
        Optional[ParallelismPlan]
            A new plan if replanning was triggered, ``None`` otherwise.
        """
        now = time.monotonic()
        step_duration = now - self._last_step_time
        self._last_step_time = now

        # Update statistics
        self._steps_completed = step_num
        self._loss_history.append(loss)

        # Estimate throughput from step duration
        if step_duration > 0:
            tokens_this_step = context_len * self._batch_size if self._batch_size > 0 else context_len
            throughput = tokens_this_step / step_duration
            self._throughput_history.append(throughput)

        # Track peak memory
        mem_usage = self._profiler.get_memory_usage()
        if mem_usage.t0_used_gb > self._peak_memory_gb:
            self._peak_memory_gb = mem_usage.t0_used_gb

        # Check if replanning is needed
        new_plan: Optional[ParallelismPlan] = None

        curriculum_shift = self.detect_curriculum_shift(context_len)
        periodic_replan = self.should_replan(step_num)

        if curriculum_shift or periodic_replan:
            reason = "curriculum shift" if curriculum_shift else f"periodic (step {step_num})"
            logger.info("Replanning triggered: %s", reason)
            new_plan = self._trigger_replan(context_len)
            self._replan_count += 1

        # Update last context length
        self._last_context_len = context_len

        return new_plan

    def detect_curriculum_shift(self, current_context_len: int) -> bool:
        """Detect whether the context length changed from the last step.

        Time: O(1).  Space: O(1).

        Parameters
        ----------
        current_context_len : int
            Context length of the current step.

        Returns
        -------
        bool
            ``True`` if context length changed after the first recorded step.
        """
        # FIX: Treat the first step as the baseline (no shift) to avoid spurious replans.
        if self._last_context_len is None:
            return False
        return current_context_len != self._last_context_len

    def should_replan(self, step_num: int) -> bool:
        """Check whether periodic replanning is due.

        Time: O(1).  Space: O(1).

        Parameters
        ----------
        step_num : int
            Current training step number.

        Returns
        -------
        bool
            ``True`` if ``step_num % replan_interval == 0`` (and step > 0).
        """
        if step_num == 0:
            return False
        return step_num % self._replan_interval == 0

    def get_training_stats(self) -> TrainingStats:
        """Return comprehensive training statistics.

        Time: O(1).  Space: O(1).

        Returns
        -------
        TrainingStats
            Snapshot of current training statistics.
        """
        now = time.monotonic()
        avg_loss = (
            sum(self._loss_history) / len(self._loss_history)
            if self._loss_history
            else 0.0
        )
        avg_throughput = (
            sum(self._throughput_history) / len(self._throughput_history)
            if self._throughput_history
            else 0.0
        )
        return TrainingStats(
            steps_completed=self._steps_completed,
            avg_loss=avg_loss,
            avg_throughput_tokens_s=avg_throughput,
            peak_memory_gb=self._peak_memory_gb,
            replan_count=self._replan_count,
            total_training_time_s=now - self._start_time,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _trigger_replan(self, context_len: int) -> ParallelismPlan:
        """Execute a replan using current profiling data.

        Time: O(K) where K = number of profiled ops.
        """
        profiling_data = self._profiler.get_timings()
        new_plan = self._planner.replan(
            profiling_data=profiling_data,
            model_params=self._model_params,
            vocab_size=self._vocab_size,
            context_len=context_len,
            batch_size=self._batch_size,
        )
        self._current_plan = new_plan
        logger.info("New plan after replan: %s", new_plan)
        return new_plan

    @property
    def current_plan(self) -> Optional[ParallelismPlan]:
        """The currently active parallelism plan, if any."""
        return self._current_plan

    def __repr__(self) -> str:
        stats = self.get_training_stats()
        return (
            f"TrainingController(steps={stats.steps_completed}, "
            f"avg_loss={stats.avg_loss:.4f}, "
            f"replans={stats.replan_count}, "
            f"plan={self._current_plan!r})"
        )
