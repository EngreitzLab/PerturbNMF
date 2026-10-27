"""Where every program stands: one table over a run's Programs/P<k>/plan.md, and a non-zero exit while a
required step is not done.

    python status.py --run_dir <out_dir>/<run_name>/Annotation [--recheck]

Writes <run_dir>/Annotation_status.tsv (program x step: done / todo / failed / skipped, plus each step's note)
and prints a count per step. --recheck re-runs the checks of every step marked done
(1.Annotate/validate_program.py) and reports a done step whose outputs no longer pass as "stale".
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "0.Prepare"), str(HERE / "1.Annotate")]
from program_dir import STEPS, list_programs, read_plan  # noqa: E402

STEP_IDS = [s[0] for s in STEPS]
OPTIONAL = {s[0] for s in STEPS if s[3]}


def main():
    p = argparse.ArgumentParser(description="Status of every step of every program in an annotation run.")
    p.add_argument("--run_dir", required=True, help="Annotation run folder (holds Programs/).")
    p.add_argument("--out", default=None, help="Status table. Default: <run_dir>/Annotation_status.tsv.")
    p.add_argument("--recheck", action="store_true", help="Re-run the checks of every done step; failures show as stale.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    folders = list_programs(run_dir)
    if not folders:
        sys.exit(f"no Programs/P<k>/ in {run_dir}: run 0.Prepare/prepare_programs.py first")
    if args.recheck:
        from validate_program import validate_step

    table = {}
    for d in folders:
        row = {}
        for s in read_plan(d).steps:
            status, note = s.status, s.note
            if args.recheck and status == "done":
                errors = validate_step(d, s.id)
                if errors:
                    status, note = "stale", "; ".join(errors)[:200]
            row[s.id] = (status, note)
        table[d.name] = row

    out = Path(args.out or run_dir / "Annotation_status.tsv")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["program", *STEP_IDS, "notes"])
        for prog, row in table.items():
            notes = "; ".join(f"{s}: {row[s][1]}" for s in STEP_IDS
                              if s in row and row[s][0] in ("failed", "stale", "skipped"))
            w.writerow([prog, *(row.get(s, ("missing", ""))[0] for s in STEP_IDS), notes])

    print(f"{len(table)} programs in {run_dir}")
    blocking = 0
    for s in STEP_IDS:
        counts = {}
        for row in table.values():
            st = row.get(s, ("missing", ""))[0]
            counts[st] = counts.get(st, 0) + 1
        not_done = [prog for prog, row in table.items() if row.get(s, ("missing",))[0] not in ("done", "skipped")]
        flag = "" if not not_done else (" (optional)" if s in OPTIONAL
                                        else f"  <- {', '.join(not_done[:8])}" + (" ..." if len(not_done) > 8 else ""))
        print(f"  {s:<19} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + flag)
        if s not in OPTIONAL:
            blocking += len(not_done)
    print(f"[status] {out}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
