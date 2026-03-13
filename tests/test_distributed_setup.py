"""Tests for distributed_setup.py — single-process mode."""

from __future__ import annotations

import os

from zenyx.train.distributed_setup import (
    auto_init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestDistributedSetupSmoke:

    def test_module_imports(self) -> None:
        """Module imports without error."""
        import zenyx.train.distributed_setup  # noqa: F401


# ---------------------------------------------------------------------------
# Single-process behaviour
# ---------------------------------------------------------------------------


class TestSingleProcess:
    """In CI (no env vars set) all helpers should return safe defaults."""

    def test_auto_init_returns_false(self) -> None:
        """auto_init_distributed() should return False (no init)."""
        # Ensure distributed env vars are NOT set
        for v in ("TORCHELASTIC_RESTART_COUNT", "SLURM_PROCID",
                   "TPU_WORKER_ID", "RANK", "WORLD_SIZE"):
            os.environ.pop(v, None)
        result = auto_init_distributed()
        assert result is False

    def test_get_rank_returns_0(self) -> None:
        assert get_rank() == 0

    def test_get_world_size_returns_1(self) -> None:
        assert get_world_size() == 1

    def test_is_main_process_returns_true(self) -> None:
        assert is_main_process() is True
