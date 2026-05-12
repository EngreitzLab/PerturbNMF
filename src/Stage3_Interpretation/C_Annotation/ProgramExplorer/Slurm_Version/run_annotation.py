#!/usr/bin/env python3
"""Standalone entry point for the Annotation pipeline.

Can be run directly or submitted via SLURM.
Wraps the run_pipeline.py orchestrator with sys.path setup.
"""
import sys
import warnings
from pathlib import Path

# Add the src directory to sys.path so that all modules are importable
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from run_pipeline import main

if __name__ == "__main__":
    warnings.warn(
        "C_Annotation / ProgramExplorer is a BETA feature still being developed. "
        "Outputs may change and edge cases are not fully tested.",
        FutureWarning,
        stacklevel=2,
    )
    sys.exit(main())
