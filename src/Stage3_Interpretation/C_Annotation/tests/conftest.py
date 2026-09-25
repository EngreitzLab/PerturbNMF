"""Make annotator_core and the annotator script directories importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "annotator_core", ROOT / "RegulatorGroupAnnotator" / "scripts"):
    sys.path.insert(0, str(path))
