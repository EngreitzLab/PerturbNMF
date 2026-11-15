"""Make the program labels of a run distinguishable from each other: the set-level half of the label step.

    python resolve_label_collisions.py --run_dir <out_dir>/<run_name>/Annotation

Each program agent labels its program alone (1.Annotate, references/label.md), so nothing stops two agents from
choosing near-duplicate names for sibling programs. This step reads every Programs/P<k>/label.json together:

  detect  (no model call) groups programs whose labels collide: the same label_family (normalised), or a word-token
          Jaccard >= --min_jaccard between their names. Groups are joined by union-find.
  resolve one call per group. The prompt holds only each member's label fields, function sentence, top-loading and
          distinctive genes; no regulators, interpretation or other programs. The answer standardises the shared
          family and rewrites only the colliding labels (prompts/label_collisions/).
  apply   writes the new name into label.json (program_name, label_family, label_distinguisher), keeping the
          agent's name as program_name_before_disambiguation (and all its label fields in before_disambiguation), and rewrites the label string of interpretation.json,
          evidence.json and driver_genes.json where it held the old name, so validate_program.py and the viewers
          keep matching them.

Detection always reads the agent's own names (program_name_before_disambiguation when present), so a rerun finds
the same groups and reuses each cached answer while its input is unchanged. Answers and run records go to
<run_dir>/Synthesis/Label_collisions/{group<NN>.json, _runs/}. Claude answers through a Claude Code session by
default (--llm_backend session, two passes; no API cost); --llm_backend api needs ANTHROPIC_API_KEY.
--detect_only prints the groups and stops. While any group's answer is pending the step exits 2, so
run_synthesis.sh (set -e) stops before the steps that read the labels.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parent / "api"),
    str(HERE.parent / "0.Prepare"),
    str(HERE.parent / "1.Annotate"),
]
from budget import add_budget_args, attach, resolve_budget_dir  # noqa: E402
from manifest import StepRun  # noqa: E402
from program_dir import list_programs, read_json, read_tsv  # noqa: E402
from prompts import load as load_prompt, record as prompt_record  # noqa: E402
from validate_program import (  # noqa: E402
    LABEL_BANNED_WORDS,
    LABEL_MAX_WORDS,
    capitalize_distinguisher,
)

SYSTEM = load_prompt("label_collisions/system.md")
TASK = load_prompt("label_collisions/task.md")
STOPWORDS = {"and", "or", "of", "the", "in", "a", "an", "to", "via", "with"}


class ResolvedLabel(BaseModel):
    program_id: str
    label: str
    distinguisher: str
    evidence: str
    belongs_to_family: bool
    kept_verbatim: bool
    used_bare_number: bool


class GroupResolution(BaseModel):
    family: Optional[str]
    programs: List[ResolvedLabel]


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).strip()


def tokens(text):
    return {t for t in normalize(text).split() if t not in STOPWORDS}


def jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def agent_field(label, key):
    """The agent's own value of a label field, before this step rewrote it."""
    before = label.get("before_disambiguation") or {}
    return before[key] if key in before else label.get(key)


def agent_name(label):
    return agent_field(label, "program_name")


def find_collisions(labels, min_jaccard):
    """[[pid, ...], ...]: programs joined by an equal normalised label_family or a near-duplicate agent name."""
    parent = {pid: pid for pid in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_family = defaultdict(list)
    for pid, label in labels.items():
        family = normalize(agent_field(label, "label_family") or "")
        if family:
            by_family[family].append(pid)
    for members in by_family.values():
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    ids = list(labels)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if jaccard(agent_name(labels[a]), agent_name(labels[b])) >= min_jaccard:
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for pid in ids:
        groups[find(pid)].append(pid)
    return [members for members in groups.values() if len(members) > 1]


def program_block(program_dir, label, n_genes):
    genes = read_tsv(program_dir / "genes_top300.tsv")
    top = [g["gene"] for g in genes[:n_genes]]
    by_distinctive = sorted(genes, key=lambda g: -float(g["distinctive_score"] or 0))
    distinctive = [g["gene"] for g in by_distinctive if g["gene"] not in top][:n_genes]
    return (
        f"### {program_dir.name}\n"
        f"- current label: {agent_name(label)}\n"
        f"- claimed family: {agent_field(label, 'label_family') or '(none)'}\n"
        f"- claimed distinguisher: {agent_field(label, 'label_distinguisher') or '(none)'}\n"
        f"- distinguisher evidence: {agent_field(label, 'distinguisher_evidence') or '(none)'}\n"
        f"- function: {label.get('function_sentence', '')}\n"
        f"- top-loading genes: {', '.join(top)}\n"
        f"- distinctive genes (high here, low in other programs): {', '.join(distinctive) or '(none)'}\n"
    )


def check_resolution(result, members):
    """Errors of an answer: every member exactly once, and each label within the label rules."""
    errors = []
    got = [p.program_id for p in result.programs]
    if sorted(got) != sorted(members):
        errors.append(f"answer covers {sorted(got)}, the group is {sorted(members)}")
    for p in result.programs:
        if len(p.label.replace(" - ", " ").split()) > LABEL_MAX_WORDS:
            errors.append(
                f"{p.program_id}: label has more than {LABEL_MAX_WORDS} words"
            )
        banned = [
            w for w in LABEL_BANNED_WORDS if re.search(rf"\b{w}\b", p.label, re.I)
        ]
        if banned:
            errors.append(f"{p.program_id}: label uses {', '.join(banned)}")
        if re.search(r"\([^)]*\)", p.label):
            errors.append(f"{p.program_id}: label carries a parenthesized tag")
    return errors


def split_label(name):
    """(label_family, label_distinguisher) of a final name: 'F - D', a comma list (no family) or a bare family."""
    if " - " in name:
        family, distinguisher = name.split(" - ", 1)
        return family.strip(), distinguisher.strip()
    return ("", "") if "," in name else (name.strip(), "")


def apply_label(program_dir, label, new_name, resolved):
    """Write new_name into label.json and the label strings of the files made for the old name."""
    old = label["program_name"]
    label.setdefault("program_name_before_disambiguation", old)
    label.setdefault(
        "before_disambiguation",
        {
            k: label.get(k)
            for k in (
                "program_name",
                "label_family",
                "label_distinguisher",
                "distinguisher_evidence",
            )
        },
    )
    label["program_name"] = new_name
    label["label_family"], label["label_distinguisher"] = split_label(new_name)
    label["distinguisher_evidence"] = (
        resolved.evidence
        if resolved.evidence and not resolved.kept_verbatim
        else label["before_disambiguation"]["distinguisher_evidence"]
    )
    label["disambiguation"] = {
        "kept_verbatim": resolved.kept_verbatim,
        "belongs_to_family": resolved.belongs_to_family,
        "used_bare_number": resolved.used_bare_number,
    }
    (program_dir / "label.json").write_text(
        json.dumps(label, indent=2, ensure_ascii=False)
    )
    for name, key in (
        ("interpretation.json", "label"),
        ("evidence.json", "label"),
        ("driver_genes.json", "annotation"),
    ):
        block = read_json(program_dir / name)
        if block is not None and block.get(key) == old and old != new_name:
            block[key] = new_name
            (program_dir / name).write_text(
                json.dumps(block, indent=2, ensure_ascii=False)
            )


def build_parser():
    p = argparse.ArgumentParser(
        description="Make a run's program labels distinguishable from each other."
    )
    p.add_argument(
        "--run_dir", required=True, help="Folder holding Programs/P<k>/label.json."
    )
    p.add_argument(
        "--out_dir", default=None, help="Default: <run_dir>/Synthesis/Label_collisions."
    )
    p.add_argument(
        "--min_jaccard",
        type=float,
        default=0.6,
        help="Two names collide at this word-token Jaccard (stop words dropped) or above.",
    )
    p.add_argument(
        "--n_genes",
        type=int,
        default=12,
        help="Top-loading and distinctive genes shown per program.",
    )
    p.add_argument(
        "--detect_only",
        action="store_true",
        help="Print the colliding groups and stop.",
    )
    p.add_argument("--model", default="claude-sonnet-5", help="Anthropic model id.")
    p.add_argument(
        "--effort",
        default="medium",
        help="Effort level: low | medium | high | xhigh | max.",
    )
    p.add_argument("--max_tokens", type=int, default=4000, help="Answer token limit.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo groups whose cached answer had the same input.",
    )
    p.add_argument(
        "--llm_backend",
        default="session",
        choices=["api", "session"],
        help="Default session (no API cost): write each prompt to --session_dir, answer it in Claude Code, run again.",
    )
    p.add_argument(
        "--session_dir", default=None, help="Default: <run_dir>/Session_LLM."
    )
    p.add_argument(
        "--session_model",
        default="claude-code-session",
        help="Model name recorded for session answers.",
    )
    add_budget_args(p, reserve_default=0.0)
    return p


def main():
    from env import load_env, make_handler

    args = build_parser().parse_args()
    load_env()
    run_dir = Path(args.run_dir).resolve()
    dirs = {d.name: d for d in list_programs(run_dir) if (d / "label.json").exists()}
    labels = {pid: read_json(d / "label.json") for pid, d in dirs.items()}
    if not labels:
        sys.exit(f"no Programs/P<k>/label.json under {run_dir}")
    groups = find_collisions(labels, args.min_jaccard)
    n_colliding = sum(len(g) for g in groups)
    print(
        f"{len(labels)} labels, {len(groups)} colliding group(s) covering {n_colliding} programs"
    )
    for i, members in enumerate(groups, 1):
        print(
            f"  group{i:02d}: "
            + "; ".join(f"{p} {agent_name(labels[p])!r}" for p in members)
        )
    if args.detect_only or not groups:
        return 0

    out_dir = Path(args.out_dir or run_dir / "Synthesis" / "Label_collisions")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolve_budget_dir(args, run_dir)
    if args.llm_backend == "session":
        from session_llm import session_llm

        handler = session_llm(
            args.session_dir or run_dir / "Session_LLM",
            "label_collisions",
            args.session_model,
        )
    else:
        handler = make_handler(args.model)
    model = getattr(handler, "model_label", args.model)
    prompts = prompt_record(SYSTEM, TASK)
    n_fail, n_rewritten, n_pending = 0, 0, 0
    for i, members in enumerate(groups, 1):
        key = f"group{i:02d}"
        cell_type = (read_json(dirs[members[0]] / "program.json") or {}).get(
            "cell_type", ""
        )
        text = (
            f"Cell type: {cell_type}\n{len(members)} colliding programs.\n\n"
            + "\n".join(
                program_block(dirs[p], labels[p], args.n_genes) for p in members
            )
        )
        input_sha = hashlib.sha256(text.encode()).hexdigest()
        out = out_dir / f"{key}.json"
        run = StepRun(
            out_dir,
            key,
            "label_collisions",
            model,
            prompts,
            attach(handler, args, key, "label_collisions"),
        )
        cached = json.loads(out.read_text()) if out.exists() else None
        if cached and not args.overwrite and cached.get("input_sha256") == input_sha:
            result = GroupResolution.model_validate(cached["answer"])
            run.finish("cached")
        else:
            handler.key = key
            result = handler.get_json(
                SYSTEM.text,
                f"{text}\n\n---\n\n{TASK.text}",
                GroupResolution,
                effort=args.effort,
                max_tokens=args.max_tokens,
            )
            if result is None:
                if getattr(handler, "last_status", None) == "pending":
                    run.finish("pending_session")
                    n_pending += 1
                    continue
                run.finish(
                    "failed",
                    errors=[
                        getattr(handler, "last_error", None) or "no answer returned"
                    ],
                )
                print(f"  {key}: [fail] no answer returned")
                n_fail += 1
                continue
            errors = check_resolution(result, members)
            if errors:
                run.finish(
                    "failed_validation", errors=errors, raw_answer=result.model_dump()
                )
                print(f"  {key}: [fail] " + "; ".join(errors))
                n_fail += 1
                continue
            out.write_text(
                json.dumps(
                    {
                        "members": members,
                        "answer": result.model_dump(),
                        "input_sha256": input_sha,
                        "model": model,
                        "prompts": prompts,
                        "created": datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            run.finish("completed", {"family": result.family, "members": members})
        for resolved in result.programs:
            pid = resolved.program_id
            resolved.label = capitalize_distinguisher(resolved.label)
            label = read_json(dirs[pid] / "label.json")
            if label["program_name"] != resolved.label:
                n_rewritten += 1
                print(f"  {pid}: {label['program_name']!r} -> {resolved.label!r}")
            apply_label(dirs[pid], label, resolved.label, resolved)

    if n_pending:
        handler.report()
        print(
            "[stop] labels are not final until every group is answered; the later synthesis steps would read old names"
        )
        return 2
    final = {
        pid: read_json(d / "label.json")["program_name"] for pid, d in dirs.items()
    }
    still = [
        (a, b)
        for i, a in enumerate(final)
        for b in list(final)[i + 1 :]
        if final[a] == final[b] or jaccard(final[a], final[b]) >= args.min_jaccard
    ]
    for a, b in still:
        print(f"  [warn] still close: {a} {final[a]!r} / {b} {final[b]!r}")
    print(
        f"[done] {len(groups) - n_fail}/{len(groups)} group(s) resolved, {n_rewritten} label(s) rewritten -> {out_dir}"
    )
    if args.llm_backend == "session":
        handler.report()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
