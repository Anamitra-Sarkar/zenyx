"""Tests for Audit Fix 5 — checkpoint.py weights_only=True verification.

Ensures no torch.load call uses weights_only=False.
"""

from __future__ import annotations

import os


def test_no_weights_only_false_in_checkpoint() -> None:
    """Assert weights_only=False does not appear in checkpoint.py."""
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "zenyx", "train", "checkpoint.py"
    )
    with open(checkpoint_path, "r") as f:
        content = f.read()
    assert "weights_only=False" not in content, (
        "checkpoint.py contains weights_only=False — this is a security risk "
        "for PyTorch >= 2.5. Use weights_only=True."
    )


def test_no_weights_only_false_in_trainer() -> None:
    """Assert weights_only=False does not appear in trainer.py."""
    trainer_path = os.path.join(
        os.path.dirname(__file__), "..", "zenyx", "train", "trainer.py"
    )
    with open(trainer_path, "r") as f:
        content = f.read()
    assert "weights_only=False" not in content, (
        "trainer.py contains weights_only=False — this is a security risk "
        "for PyTorch >= 2.5. Use weights_only=True."
    )
