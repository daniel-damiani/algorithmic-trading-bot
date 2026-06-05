#!/usr/bin/env python3
"""Entry point wrapper for production model training."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "training" / "train_production_model.py"

if __name__ == "__main__":
    raise SystemExit(
        subprocess.run([sys.executable, str(SCRIPT), *sys.argv[1:]]).returncode
    )
