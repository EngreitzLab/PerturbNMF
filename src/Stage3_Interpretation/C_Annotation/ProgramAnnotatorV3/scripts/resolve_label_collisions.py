"""Detect and resolve near-duplicate program labels across a whole run.

Why this is a separate stage: every annotator sees exactly one program, which is what keeps the
comparison blinded — but it also means no annotator can know that another program already took
its label. Distinguishability is therefore a property of the SET, and can only be fixed once the
set exists.

Two steps, and only the second costs anything:

  detect  — deterministic. Groups programs that collide, by normalised `label_family` and by
            near-duplicate labels (word-token Jaccard >= 0.6, the GeneProgramExplorer threshold).
            Writes one isolated prompt per colliding group.
  apply   — reads the answers back and rewrites `label` in place, keeping the original as
            `label_before_disambiguation` so nothing is silently overwritten.

The disambiguation prompt is shown ONLY the colliding programs' own labels, families,
distinguishers and gene lists. It never sees the reference annotation.

Usage:
    python resolve_label_collisions.py detect --dispatch dispatch --arm v3 \
        --gene-loading gene_loading_top300_with_uniqueness.csv \
        --out-root dispatch_disambiguation
    python resolve_label_collisions.py apply --dispatch dispatch --arm v3 \
        --out-root dispatch_disambiguation
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

JACCARD_THRESHOLD = 0.6
STOPWORDS = {"and", "or", "of", "the", "in", "a", "an", "to", "via", "with"}


def load_answer(path: Path) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", path.read_text().strip(), flags=re.MULTILINE)
    return json.loads(raw)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).strip()


def tokens(text: str) -> set:
    return {t for t in normalize(text).split() if t and t not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def slot_claim(entry: dict, slot: str) -> str:
    value = (entry.get("interpretation") or {}).get(slot) or {}
    return (value.get("claim") if isinstance(value, dict) else str(value)) or "(not filled)"


def find_collisions(labels: Dict[int, dict]) -> List[List[int]]:
    """Union-find over two collision signals: same family, or near-duplicate label."""
    parent = {pid: pid for pid in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_family = defaultdict(list)
    for pid, entry in labels.items():
        family = normalize(entry.get("label_family") or "")
        if family:
            by_family[family].append(pid)
    for members in by_family.values():
        for other in members[1:]:
            union(members[0], other)

    ids = sorted(labels)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if jaccard(labels[a].get("label", ""), labels[b].get("label", "")) >= JACCARD_THRESHOLD:
                union(a, b)

    groups = defaultdict(list)
    for pid in ids:
        groups[find(pid)].append(pid)
    return [sorted(members) for members in groups.values() if len(members) > 1]


PROMPT = """You are standardising the labels of a set of gene programs from ONE single-cell \
CRISPRi Perturb-seq experiment in {cell_system}.

These {n} programs were each labelled independently, by someone who could not see the others. \
They have collided: their labels are near-duplicates, or they claim the same broad family. A \
label that could equally well name a sibling program is not doing its job — a reader must be \
able to tell which program it refers to.

Make the set mutually distinguishable, changing as little as possible. Rules:

0. DO NO HARM. If a program's current label is already specific and distinct from every other \
label in this group, KEEP IT VERBATIM. Only rewrite labels that actually collide. Two \
illustrative examples of what harm looks like — both replaced a good, \
established label with a worse one:
     "ATF4 Amino Acid Stress Response"      -> "Integrated stress response - Serine/tRNA synthesis"
     "Angiogenesis - Caveolar Endocytosis"  -> "Angiogenesis - SMAD3"
   The first swapped a specific, established name for a broad family plus one arbitrary \
identifier; the second swapped a process term that already separated it from "Angiogenesis - \
ECM proteolysis" for a single gene.

1. STANDARDISE THE SHARED PART. If these really are variations on one theme, give them all the \
same family string rather than paraphrases of it, then distinguish within it:
     Angiogenesis - Tip cell        Angiogenesis - Stalk cell
     Cell cycle - G2M               Cell cycle - G1/S
2. DISTINGUISH BY PROCESS, NOT BY GENE. The distinguisher must be a cell-process, pathway, \
compartment, phase or state term. Use each program's upstream trigger, co-regulation mechanism, \
cellular output, temporal window (time courses) and distinctive genes (all listed below) to find the process that sets it \
apart, then name the process. Do NOT use a bare gene symbol as the distinguisher and do NOT \
append a gene in parentheses: a named sub-state needs no gene tag ("Cell cycle - G2M" beats \
"Cell cycle - G2M mitotic exit (CDC20)"). The only exception is a gene that IS the accepted \
name of the process ("KLF2 flow response", "ATF4 amino acid stress response").
   PREFER THE ESTABLISHED NAME of a process over a family-plus-modifier you construct.
3. A BARE TRAILING NUMBER ("Angiogenesis 1", "Angiogenesis 2") is the LAST resort, allowed only \
when nothing in the evidence defensibly separates two programs. Using it is an honest admission, \
not a failure — but do not reach for it before trying rule 2.
4. BE CONSERVATIVE. Only claim what that program's genes support. A sharper label that \
overstates the evidence is worse than a duller one that does not. If you are inventing a \
distinction to avoid a number, use the number instead.
5. Keep every label <= 6 words. Do not use the words "program", "process", "regulation of", \
"grab bag", "incoherent", "heterogeneous", "mixed", "unclear" or "miscellaneous". A label that \
lists 2-3 distinct processes as a comma-separated list ("Basement membrane, lysosomal, \
mitochondrial OXPHOS") is fine and should normally be kept as is. If such a label separates its \
processes with " / ", change only the separators to commas.
6. If a program in this group does NOT actually belong to the family, say so: give it a label \
from its own genes and set `"belongs_to_family": false`.

## The colliding programs

{program_blocks}

# OUTPUT — JSON only, no preamble, no code fences

{{"family": "<the standardised shared family name, or null if they do not share one>",
  "programs": [
    {{"program_id": <int>, "label": "<rewritten>", "distinguisher": "<what separates it>",
      "evidence": "<the genes or state the distinguisher rests on>",
      "belongs_to_family": <true|false>,
      "kept_verbatim": <true|false>,
      "used_bare_number": <true|false>}}
  ]}}
"""


def cmd_detect(args: argparse.Namespace) -> int:
    answers: Dict[int, dict] = {}
    for path in sorted(Path(args.dispatch).glob(f"{args.arm}_p*/answer.json")):
        pid = int(re.search(r"_p(\d+)", path.parent.name).group(1))
        try:
            answers[pid] = load_answer(path)
        except json.JSONDecodeError:
            print(f"skipping P{pid}: answer is not valid JSON")

    if not answers:
        print("no answers found — run the annotation pass first")
        return 1

    loading = pd.read_csv(args.gene_loading)
    if "program_id" not in loading.columns:
        loading = loading.rename(columns={"RowID": "program_id"})

    groups = find_collisions(answers)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not groups:
        print(f"{len(answers)} labels, no collisions — nothing to disambiguate")
        return 0

    for index, members in enumerate(groups, start=1):
        blocks = []
        for pid in members:
            entry = answers[pid]
            frame = loading[loading["program_id"] == pid]
            top = frame.nlargest(12, "Score")["Name"].tolist()
            distinctive = (
                frame.nlargest(40, "UniquenessScore")["Name"].tolist()
                if "UniquenessScore" in frame.columns
                else []
            )
            distinctive = [g for g in distinctive if g not in top][:12]
            blocks.append(
                f"### Program {pid}\n"
                f"- current label: {entry.get('label','')}\n"
                f"- claimed family: {entry.get('label_family','')}\n"
                f"- claimed distinguisher: {entry.get('label_distinguisher','') or '(none given)'}\n"
                f"- distinguisher evidence: {entry.get('label_distinguisher_evidence','') or '(none given)'}\n"
                f"- one-line summary: {entry.get('brief_summary','')}\n"
                f"- upstream trigger: {slot_claim(entry, 'upstream_trigger')}\n"
                f"- co-regulation mechanism: {slot_claim(entry, 'coregulation_mechanism')}\n"
                f"- cellular output: {slot_claim(entry, 'cellular_output')}\n"
                + (
                    f"- temporal window: {slot_claim(entry, 'temporal_window')}\n"
                    if (entry.get("interpretation") or {}).get("temporal_window")
                    else ""
                )
                + (
                    f"- group dependence: {slot_claim(entry, 'group_dependence')}\n"
                    if (entry.get("interpretation") or {}).get("group_dependence")
                    else ""
                )
                + f"- top-loading genes: {', '.join(top)}\n"
                f"- distinctive genes (high here, low elsewhere): {', '.join(distinctive) or '(none)'}\n"
            )

        directory = out_root / f"{args.arm}_group{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt.md").write_text(
            PROMPT.format(
                cell_system=args.cell_system,
                n=len(members),
                program_blocks="\n".join(blocks),
            ),
            encoding="utf-8",
        )
        (directory / "members.json").write_text(json.dumps(members), encoding="utf-8")
        print(f"group {index:02d}: programs {members} -> {directory}")

    colliding = sum(len(g) for g in groups)
    print(
        f"\n{len(groups)} colliding group(s) covering {colliding}/{len(answers)} programs; "
        f"{len(answers) - colliding} labels already unique"
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    rewritten = 0
    for directory in sorted(Path(args.out_root).glob(f"{args.arm}_group*")):
        answer_path = directory / "answer.json"
        if not answer_path.exists():
            print(f"{directory.name}: no answer yet, skipping")
            continue
        payload = load_answer(answer_path)
        for entry in payload.get("programs", []):
            pid = int(entry["program_id"])
            target = Path(args.dispatch) / f"{args.arm}_p{pid}" / "answer.json"
            if not target.exists():
                print(f"  P{pid}: original answer missing, skipping")
                continue
            original = load_answer(target)
            if "label_before_disambiguation" not in original:
                original["label_before_disambiguation"] = original.get("label", "")
            original["label"] = entry["label"]
            original["label_distinguisher"] = entry.get("distinguisher", "")
            original["label_distinguisher_evidence"] = entry.get("evidence", "")
            original["disambiguation_used_bare_number"] = entry.get("used_bare_number", False)
            target.write_text(json.dumps(original, indent=2), encoding="utf-8")
            rewritten += 1
            print(
                f"  P{pid}: {original['label_before_disambiguation']!r} -> {entry['label']!r}"
            )
    print(f"\nrewrote {rewritten} label(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect")
    detect.add_argument("--dispatch", required=True)
    detect.add_argument("--arm", default="v3")
    detect.add_argument("--gene-loading", required=True)
    detect.add_argument("--out-root", required=True)
    detect.add_argument(
        "--cell-system",
        required=True, help="the cell system, as in the annotation config",
    )
    detect.set_defaults(func=cmd_detect)

    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--dispatch", required=True)
    apply_parser.add_argument("--arm", default="v3")
    apply_parser.add_argument("--out-root", required=True)
    apply_parser.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
