"""Tests for Audit Fix 2 — loop.py deprecation warnings."""

from __future__ import annotations

import warnings

import pytest
import torch.nn as nn


@pytest.fixture
def tiny_model() -> nn.Module:
    """A minimal nn.Module for wrap() tests."""
    return nn.Linear(10, 10)


class TestLoopDeprecation:
    """Verify loop.py public functions emit DeprecationWarning."""

    def test_wrap_emits_deprecation(self) -> None:
        """wrap() should emit a DeprecationWarning."""
        model = nn.Linear(10, 10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from zenyx.train.loop import wrap
            try:
                wrap(model)
            except Exception:
                pass  # May fail for other reasons; we just check the warning
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1, "wrap() should emit DeprecationWarning"
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_wrap_not_in_train_all_toplevel(self) -> None:
        """wrap should NOT be a top-level export of zenyx.train.__init__.py."""
        import importlib
        train_module = importlib.import_module("zenyx.train")
        all_exports = getattr(train_module, "__all__", [])
        assert "wrap" not in all_exports, (
            "wrap should not be in zenyx.train.__all__ (moved to legacy submodule)"
        )

    def test_legacy_submodule_exists(self) -> None:
        """zenyx.train.legacy should exist and provide wrap()."""
        from zenyx.train import legacy
        assert hasattr(legacy, "wrap")


# ---------------------------------------------------------------------------
# Issue 12 — legacy.wrap() stacklevel correctness
# ---------------------------------------------------------------------------


def test_legacy_wrap_emits_exactly_one_warning(tiny_model: nn.Module) -> None:
    """legacy.wrap() must emit exactly one DeprecationWarning.

    The inner loop.wrap() DeprecationWarning is suppressed by legacy.wrap(),
    so the caller receives exactly one warning, not two.
    """
    from zenyx.train import legacy

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            legacy.wrap(tiny_model)
        except Exception:
            pass  # wrap() may fail for unrelated reasons; we only test warnings

    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, (
        f"Expected exactly 1 DeprecationWarning, got {len(dep_warnings)}"
    )


def test_legacy_wrap_warning_points_to_caller(tiny_model: nn.Module) -> None:
    """legacy.wrap() warning must point to this test file (the caller).

    With stacklevel=2 inside legacy.wrap(), the warning skips legacy.wrap()
    itself (frame 1) and lands on the caller (frame 2 = this test function).
    It must NOT point to loop.py (the internal implementation module).
    """
    from zenyx.train import legacy

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            legacy.wrap(tiny_model)
        except Exception:
            pass

    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    # The warning must NOT point to loop.py (internal module).
    # It must point to this test file (the caller).
    assert "loop.py" not in dep_warnings[0].filename, (
        f"Warning incorrectly points to internal module: "
        f"{dep_warnings[0].filename}:{dep_warnings[0].lineno}"
    )
    assert "test_loop_deprecation" in dep_warnings[0].filename, (
        f"Warning should point to caller (test file), "
        f"got: {dep_warnings[0].filename}"
    )
