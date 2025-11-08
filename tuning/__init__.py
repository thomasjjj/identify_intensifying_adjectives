"""Utilities for dataset preparation, evaluation, and threshold tuning."""

from pathlib import Path
import sys

# Ensure repository root is importable when running modules as scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
