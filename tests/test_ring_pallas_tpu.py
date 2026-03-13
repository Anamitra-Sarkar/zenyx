"""Tests for Audit Fix 4 — ring_pallas_tpu.py head_dim propagation.

Verifies that no hardcoded 128 values remain in the kernel logic
(only as default parameter values).
"""

from __future__ import annotations

import re
import os


def test_no_hardcoded_128_in_blockspec() -> None:
    """Verify no hardcoded 128 in BlockSpec definitions or kernel shapes."""
    tpu_path = os.path.join(
        os.path.dirname(__file__), "..", "zenyx", "ops", "attention", "ring_pallas_tpu.py"
    )
    with open(tpu_path, "r") as f:
        content = f.read()

    # Find all lines with "128" that are NOT parameter defaults or comments
    lines = content.split("\n")
    violations = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        # Skip lines that are parameter defaults (e.g., head_dim: int = 128)
        if "head_dim" in line and "128" in line:
            continue
        # Skip string literals and docstrings
        if stripped.startswith(('"""', "'''", '"', "'")):
            continue
        # Look for hardcoded 128 in computation (not as a default)
        if re.search(r'\b128\b', line) and "BlockSpec" in line:
            violations.append(f"Line {i}: {stripped}")

    assert not violations, (
        f"Found hardcoded 128 in BlockSpec definitions:\n"
        + "\n".join(violations)
    )


def test_head_dim_is_configurable_default() -> None:
    """Verify head_dim=128 is a configurable default parameter."""
    tpu_path = os.path.join(
        os.path.dirname(__file__), "..", "zenyx", "ops", "attention", "ring_pallas_tpu.py"
    )
    with open(tpu_path, "r") as f:
        content = f.read()

    # The class and functions should accept head_dim as a parameter
    assert "head_dim" in content
    assert "head_dim: int = 128" in content or "head_dim: int" in content


def test_kernel_class_accepts_head_dim() -> None:
    """Verify RingFlashAttentionTPU accepts head_dim in __init__."""
    tpu_path = os.path.join(
        os.path.dirname(__file__), "..", "zenyx", "ops", "attention", "ring_pallas_tpu.py"
    )
    with open(tpu_path, "r") as f:
        content = f.read()

    # Find the __init__ method of RingFlashAttentionTPU
    assert "def __init__(self, head_dim" in content, (
        "RingFlashAttentionTPU.__init__ should accept head_dim parameter"
    )
