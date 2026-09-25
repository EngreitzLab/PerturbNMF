"""Write one blinded citation-selection prompt per program from build_citation_candidates.py output.

The annotator already fixed the label and named the genes and regulators it rests on. This pass
only picks, for each of those claims, the best support among candidates it is shown: a PubMed
sentence, an enrichment term (ideally carrying the PMID the GO annotation itself cites), or an
explicit "none". Each candidate carries an id (L1, D1, ...) so every choice can be checked
mechanically against what was offered (validate_citation_answers.py).

Usage:
    python build_citation_prompts.py --candidates <candidates_dir> --dispatch-root <citations_dispatch> \
        --arm cite --cell-system "<cell system>" --excluded-pmids <excluded.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT = """You are checking the literature and database support for an annotation that has \
already been written. {subject_intro} Your only job is to choose, for each of those claims, \
the best support from the candidates listed under that claim. You do not re-annotate the \
{subject_noun} and you do not judge the label.

RULES
1. SELECT, NEVER RECALL. Use only the candidate ids listed under the same claim (L1, D2, ...). \
Never write a PMID, term or sentence that is not listed there.
2. QUOTE VERBATIM. For a literature candidate, `quote` is copied character for character from \
its sentence (the shortest span that makes the point, at most 300 characters).
3. SUPPORT MEANS A STATED RELATIONSHIP. `direct`: the sentence or database entry itself states \
the gene's role in the labelled process (for a regulator: its effect on this process or on \
these genes). `indirect`: it states a related role (same pathway, related process, another \
cell type). A sentence that merely lists the gene next to others, or uses it only as a marker \
of something unrelated, is NOT support.
4. DISCOVERY FIRST. The citation a reader expects is the study that DISCOVERED the gene's role \
IN THE LABELLED PROCESS — the first identification of that role, the defining loss- or \
gain-of-function experiment, or the first mechanistic demonstration — not a later paper that \
restates it. The discovery must be of THIS relationship: the first paper on the gene in an \
unrelated function or organism context (e.g. embryonic patterning when the claim is about \
endothelial specification) is not it. Give that paper \
first, with `role` "discovery". Clues are printed with every paper: its year, total citations, \
how many of the other candidate papers cite it ("co-cited by N"), and how it was found \
("curated" = a database curator attached this finding to this paper). Older, highly cited, \
co-cited primary papers are the usual discovery papers, but judge by what the sentence says. \
A [REVIEW] is never a discovery paper, and neither is a paper that mentions the gene only as a \
marker or states the finding as background.
5. Then, optionally, ONE `role` "context" support: a primary study showing the relationship in \
the matching system ({cell_system}), or a database term (give the PMID its GO annotation or \
summary cites, if any). At most 2 supports per claim.
6. If none of the candidates is plausibly the original study, give the best available support \
with `role` "restatement" — never label a restatement as a discovery.
7. NONE IS A VALID ANSWER. If no candidate supports the claim, return an empty `supports` list \
and a one-line `none_reason`. A stretched citation is worse than none.
8. Record the cell or tissue system of a literature quote in `system` when the sentence or \
title states it; otherwise "not stated".
9. Respond with ONLY the JSON object specified. No preamble, no markdown fences.

# {subject_heading} {program_id}
- label: {label}
- family: {family}
- distinguisher: {distinguisher}

# CLAIMS AND THEIR CANDIDATES

{claim_blocks}

# OUTPUT — JSON only, one entry per claim above, in the same order

{{"program_id": {program_id},
  "claims": [
    {{"claim_id": "G1", "symbol": "",
      "supports": [{{"ref": "L1|D1", "type": "literature|database", "role": "discovery|context|restatement",
                    "pmid": "<pmid or empty>",
                    "quote": "<verbatim, literature only>", "strength": "direct|indirect",
                    "system": "", "why": "<one line: what it establishes>"}}],
      "none_reason": ""}}
  ]}}
"""

# The citation pass is the same for every annotator; only how the prompt names the annotated unit
# differs. The claims come from the answer's `label_evidence` either way.
SUBJECTS = {
    "program": {
        "subject_intro": "A gene program from a single-cell CRISPRi Perturb-seq screen in {cell_system} was "
                         "labelled, and the annotator named the genes and regulators the label rests on.",
        "subject_noun": "program",
        "subject_heading": "PROGRAM",
    },
    "regulator_group": {
        "subject_intro": "A group of perturbed genes from a single-cell CRISPRi Perturb-seq screen in "
                         "{cell_system} — genes whose knockdowns shift the cell's gene programs in a similar way — was "
                         "labelled, and the annotator named the member genes the label rests on (each member is a "
                         "`regulator` claim).",
        "subject_noun": "group",
        "subject_heading": "REGULATOR GROUP",
    },
}


def claim_block(claim: dict, subject_noun: str = "program") -> str:
    kind = claim["kind"]
    head = f"## {claim['claim_id']} — {kind} {claim['symbol']}"
    if kind == "gene" and claim.get("loading_rank"):
        head += f" (loading rank {claim['loading_rank']})"
    if kind == "regulator" and claim.get("role"):
        head += f" ({claim['role']})"
    lines = [head, f"Claimed role: {claim.get('context') or '(not stated)'}"]
    lines.append("Database:")
    if claim["database"]:
        for i, entry in enumerate(claim["database"], start=1):
            if entry["source"] == "NCBI Gene summary":
                lines.append(f"- D{i} NCBI Gene summary: \"{entry['term']}\"")
                continue
            term_id = f" {entry['term_id']}" if entry.get("term_id") and entry["term_id"] != "nan" else ""
            pmids = ", ".join(f"{p['pmid']} ({p['evidence_code']})" for p in entry["pmids"])
            go = f"; GO annotation of {claim['symbol']} cites PMID {pmids}" if pmids else ""
            lines.append(
                f"- D{i} {entry['source']}: {entry['term']}{term_id} (enriched in this {subject_noun}, "
                f"FDR {entry['fdr']:.1e}; {claim['symbol']} is in the overlap){go}"
            )
    else:
        lines.append("- none")
    lines.append("Literature (title/abstract sentences that name the gene; oldest first):")
    if claim["literature"]:
        for i, entry in enumerate(claim["literature"], start=1):
            clues = [entry.get("year") or "year ?"]
            if entry.get("journal"):
                clues.append(entry["journal"])
            if entry.get("cited_by") is not None:
                clues.append(f"{entry['cited_by']} citations")
            if entry.get("cocited"):
                clues.append(f"co-cited by {entry['cocited']}")
            if entry.get("channels"):
                clues.append("found: " + "+".join(entry["channels"]))
            review = " [REVIEW]" if entry.get("is_review") else ""
            lines.append(f"- L{i} PMID {entry['pmid']} ({'; '.join(clues)}){review} \"{entry['sentence']}\" — {entry['title']}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dispatch-root", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cell-system", required=True)
    parser.add_argument("--subject", choices=sorted(SUBJECTS), default="program",
                        help="what was annotated: a gene program (ProgramAnnotatorV3) or a regulator group "
                             "(RegulatorGroupAnnotator); only the wording of the prompt changes")
    parser.add_argument("--excluded-pmids", type=Path,
                        help="flag_retracted_pool_pmids.py output; retracted/unresolved candidates are dropped")
    args = parser.parse_args()
    excluded = set()
    if args.excluded_pmids:
        payload = json.loads(args.excluded_pmids.read_text())
        excluded = set(payload.get("retracted", [])) | set(payload.get("unresolved", []))

    written = 0
    for path in sorted(args.candidates.glob("program_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        candidates = json.loads(path.read_text())
        pid = candidates["program_id"]
        if not candidates["claims"]:
            continue
        for claim in candidates["claims"]:  # ids are assigned after this, so none point at a dropped paper
            claim["literature"] = [e for e in claim["literature"] if e["pmid"] not in excluded]
            for entry in claim["database"]:
                entry["pmids"] = [p for p in entry["pmids"] if p["pmid"] not in excluded]
        prompt = PROMPT.format(
            **{k: v.format(cell_system=args.cell_system) for k, v in SUBJECTS[args.subject].items()},
            cell_system=args.cell_system, program_id=pid, label=candidates["label"],
            family=candidates["label_family"] or "(none)",
            distinguisher=candidates["label_distinguisher"] or "(none)",
            claim_blocks="\n\n".join(claim_block(c, SUBJECTS[args.subject]["subject_noun"]) for c in candidates["claims"]),
        )
        directory = args.dispatch_root / f"{args.arm}_p{pid}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt.md").write_text(prompt, encoding="utf-8")
        (directory / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
        written += 1
    print(f"wrote {written} citation prompts under {args.dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
