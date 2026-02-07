#!/usr/bin/env python3
"""Compatibility wrapper for Experiment 1 allocation sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp1_allocation.run_allocation_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
