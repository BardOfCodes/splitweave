"""
Pytest configuration for splitweave.

Ensures the repository root is on sys.path so `import splitweave` works
when running pytest from the repo root without an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root = parent of tests/
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
