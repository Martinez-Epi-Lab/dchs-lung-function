#!/usr/bin/env python3
"""Validate configuration and execute the implemented manuscript modules."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from dchs_lung_function.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(root=ROOT))
