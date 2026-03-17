"""Pytest configuration for the repository test suite."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root is on sys.path so zenyx is importable without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
