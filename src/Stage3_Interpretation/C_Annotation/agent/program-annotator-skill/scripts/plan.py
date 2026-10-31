#!/usr/bin/env python3
"""Show or tick a program's plan: a thin entry point to the pipeline's 0.Prepare/program_dir.py.

    python scripts/plan.py show --program_dir Programs/P10
    python scripts/plan.py done --program_dir Programs/P10 --step gene_info --by agent
    python scripts/plan.py skip --program_dir Programs/P10 --step driver_genes --reason "not requested"

This orchestrator skill belongs to the PerturbNMF annotation pipeline, so unlike the data skills it calls the
pipeline: the skill folder's real location is C_Annotation/agent/program-annotator-skill (it is symlinked into
<run>/.claude/skills), and program_dir.py runs the step's checks before ticking it.
"""

import runpy
import sys
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parents[3] / "0.Prepare" / "program_dir.py"
if not PROGRAM_DIR.exists():
    sys.exit(
        f"cannot find {PROGRAM_DIR}: this skill must stay inside C_Annotation/agent/"
    )
sys.argv[0] = str(PROGRAM_DIR)
runpy.run_path(str(PROGRAM_DIR), run_name="__main__")
