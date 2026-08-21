"""Ensure repo root is on sys.path for local packages (app)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: hits external network / live APIs"
    )
