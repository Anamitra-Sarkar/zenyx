"""Tests for Audit Fix 2 — loop.py deprecation warnings."""

from __future__ import annotations

import warnings

import pytest


class TestLoopDeprecation:
    """Verify loop.py public functions emit DeprecationWarning."""

    def test_wrap_emits_deprecation(self) -> None:
        """wrap() should emit a DeprecationWarning."""
        import torch.nn as nn

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
