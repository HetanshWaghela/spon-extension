#!/usr/bin/env python3
"""Compatibility wrapper for Experiment 2 interpretability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp2_interpretability.run_interpretability import main


if __name__ == "__main__":
    main()
