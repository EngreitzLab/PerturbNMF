"""Copy agent/_shared/skill_lib.py into every skill's scripts/_lib/ (or, with --check, report stale copies).

    python agent/_shared/sync_skill_lib.py [--check]
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent
LIB = SHARED / "skill_lib.py"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="Only report skills whose copy differs (exit 1).")
    args = p.parse_args()
    stale = []
    for skill in sorted(SHARED.parent.glob("*-skill")):
        lib_dir = skill / "scripts" / "_lib"
        if not lib_dir.is_dir():
            continue
        target = lib_dir / "skill_lib.py"
        if target.exists() and filecmp.cmp(LIB, target, shallow=False):
            continue
        stale.append(skill.name)
        if not args.check:
            shutil.copy2(LIB, target)
            (lib_dir / "__init__.py").touch()
    print(("stale: " if args.check else "updated: ") + (", ".join(stale) or "none"))
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
