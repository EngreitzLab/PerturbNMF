"""One folder per program, and its plan: the only code that reads or writes <run>/Programs/P<k>/plan.md.

    <run>/Programs/meta.json        run-level inputs, keys and params (prepare_programs.py)
    <run>/Programs/P<k>/plan.md     the program's steps; a step is ticked only through mark_done()

plan.md is a checklist, one line per step, written back whole after every change:

    - [x] prepare — genes_top300.tsv, regulators_sig.tsv, ... — done 2026-10-25 by prepare_programs.py
    - [ ] gene_info — gene_info.jsonl — skills: mygene-skill, ncbi-gene-skill, uniprot-skill
    - [!] label — label.json — failed 2026-10-25: label.json does not match the schema

[ ] to do, [x] done, [!] failed (the note says why), [-] skipped (optional steps only).

CLI, used by the per-program agent and by hand:

    python program_dir.py done --program_dir <run>/Programs/P10 --step gene_info [--by agent]
    python program_dir.py show --program_dir <run>/Programs/P10

`done` runs the step's checks (1.Annotate/validate_program.py) first: the step is ticked only when they pass,
otherwise it is marked [!] with the reasons and the command exits 1. Every outcome is also written to
<program_dir>/_runs/<step>.json (api/manifest.py).
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# step id, outputs, skills the agent may use for it, optional
STEPS = (
    ("prepare", ("program.json", "genes_top300.tsv", "regulators_sig.tsv", "go_sig.tsv", "string_sig.tsv",
                 "literature.jsonl"), (), False),
    ("confounder_screens", ("confounder_screens.json",), (), False),
    ("gene_info", ("gene_info.jsonl",), ("mygene-skill", "ncbi-gene-skill", "uniprot-skill"), False),
    ("interactions", ("interactions.jsonl",), ("omnipath-skill",), False),
    ("literature", ("literature.jsonl",), ("gene-function-papers-skill", "generif-papers-skill", "pubmed-skill",
                                           "openalex-skill", "crossref-skill", "paper-fetch-skill"), False),
    ("regulator_context", ("regulator_context.md",), (), False),
    ("label", ("label.json",), (), False),
    ("driver_genes", ("driver_genes.json",), ("gene-function-papers-skill", "paper-fetch-skill"), True),
    ("interpretation", ("interpretation.json",), (), False),
)
STEP_IDS = tuple(s[0] for s in STEPS)
MARKS = {"todo": " ", "done": "x", "failed": "!", "skipped": "-"}
LINE = re.compile(r"^- \[(?P<mark>[ x!\-])\] (?P<id>[a-z_]+)(?P<opt> \(optional\))? — (?P<rest>.*)$")


@dataclass
class Step:
    id: str
    status: str = "todo"
    optional: bool = False
    outputs: tuple = ()
    skills: tuple = ()
    note: str = ""

    def line(self):
        parts = [", ".join(self.outputs) or "-"]
        if self.skills:
            parts.append("skills: " + ", ".join(self.skills))
        if self.note:
            parts.append(self.note)
        return (f"- [{MARKS[self.status]}] {self.id}{' (optional)' if self.optional else ''} — "
                + " — ".join(parts))


@dataclass
class Plan:
    program: str
    steps: list = field(default_factory=list)

    def step(self, step_id):
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"no step {step_id!r} in the plan of {self.program} (steps: {[s.id for s in self.steps]})")

    def next_todo(self):
        """First required step not done, else None (optional steps never block)."""
        return next((s for s in self.steps if s.status != "done" and not s.optional), None)

    def render(self):
        return "\n".join([f"# {self.program} — plan", "",
                          "Tick a step only with `python 0.Prepare/program_dir.py done --program_dir <this folder> "
                          "--step <id>` (it runs the step's checks first).", "",
                          *(s.line() for s in self.steps)]) + "\n"


def new_plan(program):
    return Plan(program, [Step(sid, "todo", opt, outs, skills) for sid, outs, skills, opt in STEPS])


def read_plan(program_dir):
    """Plan of a program folder; lines that do not parse are an error, not silently dropped."""
    program_dir = Path(program_dir)
    text = (program_dir / "plan.md").read_text()
    plan = Plan(program_dir.name)
    spec = {sid: (outs, skills) for sid, outs, skills, _ in STEPS}
    for raw in text.splitlines():
        if not raw.startswith("- ["):
            continue
        m = LINE.match(raw)
        if not m:
            raise ValueError(f"{program_dir / 'plan.md'}: cannot parse {raw!r}")
        status = {v: k for k, v in MARKS.items()}[m["mark"]]
        parts = m["rest"].split(" — ")
        note = " — ".join(p for p in parts[1:] if not p.startswith("skills: "))
        outs, skills = spec.get(m["id"], (tuple(x.strip() for x in parts[0].split(",") if x.strip() != "-"), ()))
        plan.steps.append(Step(m["id"], status, bool(m["opt"]), tuple(outs), tuple(skills), note))
    return plan


def write_plan(program_dir, plan):
    path = Path(program_dir) / "plan.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(plan.render())
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mark(program_dir, step_id, status, note="", by="", run_detail=None, errors=()):
    """Set one step's status in plan.md and write its run record."""
    sys.path.insert(0, str(ROOT / "api"))
    from manifest import StepRun
    program_dir = Path(program_dir)
    plan = read_plan(program_dir)
    step = plan.step(step_id)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    step.status = status
    step.note = {"done": f"done {today}" + (f" by {by}" if by else ""),
                 "failed": f"failed {today}: " + "; ".join(errors)[:300],
                 "skipped": f"skipped {today}" + (f": {note}" if note else ""),
                 "todo": note}[status]
    write_plan(program_dir, plan)
    outputs = {o: sha256(program_dir / o) for o in step.outputs if (program_dir / o).is_file()}
    run = StepRun(program_dir, program_dir.name, step_id, by or "unknown", record_name=step_id)
    run_status = {"done": "completed", "failed": "failed_validation", "skipped": "cached", "todo": "cached"}[status]
    run.finish(run_status, {"outputs": outputs, **(run_detail or {})}, errors=list(errors))
    return step


def mark_done(program_dir, step_id, by="", check=True):
    """Run the step's checks; tick it when they pass, else mark it failed. Returns the list of errors."""
    errors = []
    if check:
        sys.path.insert(0, str(ROOT / "1.Annotate"))
        from validate_program import validate_step
        errors = validate_step(Path(program_dir), step_id)
    mark(program_dir, step_id, "failed" if errors else "done", by=by, errors=errors)
    return errors


# readers shared by every stage
def list_programs(run_dir):
    """Program folders of a run, in numeric order."""
    root = Path(run_dir) / "Programs"
    dirs = [d for d in root.glob("P*") if d.is_dir() and re.fullmatch(r"P\d+", d.name)]
    return sorted(dirs, key=lambda d: int(d.name[1:]))


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, columns):
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description="Show a program's plan, or tick one of its steps after its checks pass.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("done", help="Run the step's checks and tick it (or mark it failed).")
    d.add_argument("--program_dir", required=True)
    d.add_argument("--step", required=True, choices=STEP_IDS)
    d.add_argument("--by", default="", help="Who did the step (e.g. agent, a script name).")
    s = sub.add_parser("skip", help="Skip an optional step.")
    s.add_argument("--program_dir", required=True)
    s.add_argument("--step", required=True, choices=STEP_IDS)
    s.add_argument("--reason", default="")
    sh = sub.add_parser("show", help="Print the plan and the next step to do.")
    sh.add_argument("--program_dir", required=True)
    args = p.parse_args()

    if args.cmd == "show":
        plan = read_plan(args.program_dir)
        print(plan.render())
        nxt = plan.next_todo()
        print(f"next: {nxt.id}" if nxt else "all required steps done")
        return 0
    if args.cmd == "skip":
        if not read_plan(args.program_dir).step(args.step).optional:
            sys.exit(f"{args.step} is not optional; only optional steps can be skipped")
        mark(args.program_dir, args.step, "skipped", note=args.reason, by="agent")
        print(f"{args.step}: skipped")
        return 0
    errors = mark_done(args.program_dir, args.step, by=args.by)
    if errors:
        print(f"{args.step}: FAILED its checks; not ticked:\n" + "\n".join(f"  - {e}" for e in errors))
        return 1
    print(f"{args.step}: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
