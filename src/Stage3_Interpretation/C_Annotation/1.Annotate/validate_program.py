"""Checks a program step must pass before program_dir.py ticks it in plan.md (fail-closed).

    python validate_program.py --program_dir <run>/Programs/P10 [--step interpretation]

Each check returns a list of errors; an empty list means the step's outputs are complete and consistent with the
program's own tables. Without --step every step marked done is re-checked.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0.Prepare"))
from program_dir import STEP_IDS, read_json, read_jsonl, read_plan, read_tsv  # noqa: E402

TABLE_COLUMNS = {
    "genes_top300.tsv": ("rank", "gene", "loading", "distinctive_score"),
    "regulators_sig.tsv": ("regulator", "condition", "log2fc", "adj_p", "role"),
    "go_sig.tsv": ("term", "go_id", "adj_p", "combined_score", "overlap", "genes"),
    "string_sig.tsv": ("category", "term", "description", "fdr", "p_value", "n_genes", "n_background", "genes"),
}


def header(path):
    with Path(path).open() as fh:
        return tuple(fh.readline().rstrip("\n").split("\t"))


def check_prepare(d):
    errors = []
    for name in ("program.json", "literature.jsonl"):
        if not (d / name).exists():
            errors.append(f"{name} missing")
    for name, cols in TABLE_COLUMNS.items():
        path = d / name
        if not path.exists():
            errors.append(f"{name} missing")
        elif header(path) != cols:
            errors.append(f"{name} columns {header(path)} != {cols}")
    if (d / "genes_top300.tsv").exists() and not read_tsv(d / "genes_top300.tsv"):
        errors.append("genes_top300.tsv has no genes")
    return errors


def check_confounder_screens(d):
    block = read_json(d / "confounder_screens.json")
    if block is None:
        return ["confounder_screens.json missing"]
    return [f"confounder_screens.json lacks {k}" for k in ("label", "screens", "params") if k not in block]


def check_jsonl(d, name, required):
    path = d / name
    if not path.exists():
        return [f"{name} missing"]
    errors = []
    try:
        rows = read_jsonl(path)
    except json.JSONDecodeError as e:
        return [f"{name} is not valid JSON lines: {e}"]
    for i, row in enumerate(rows, 1):
        missing = [k for k in required if k not in row]
        if missing:
            errors.append(f"{name} row {i} lacks {missing}")
    return errors[:20]


def regulator_names(d):
    return list(dict.fromkeys(r["regulator"] for r in read_tsv(d / "regulators_sig.tsv")))


def top_genes(d, n):
    return [g["gene"] for g in read_tsv(d / "genes_top300.tsv")[:n]]


def check_gene_info(d, n_top=20):
    errors = check_jsonl(d, "gene_info.jsonl", ("gene",))
    if errors:
        return errors
    have = {r["gene"] for r in read_jsonl(d / "gene_info.jsonl")}
    required = list(dict.fromkeys([*top_genes(d, n_top), *regulator_names(d)]))
    missing = [g for g in required if g not in have]
    return [f"gene_info.jsonl has no row for {len(missing)} of the {len(required)} required genes (top {n_top} "
            f"genes + regulators): {', '.join(missing[:10])}"] if missing else []


def check_interactions(d):
    errors = check_jsonl(d, "interactions.jsonl", ("regulator", "gene", "found"))
    if errors:
        return errors
    tested = {r["regulator"] for r in read_jsonl(d / "interactions.jsonl")}
    missing = [r for r in regulator_names(d) if r not in tested]
    return [f"interactions.jsonl has no tested pair for regulators {', '.join(missing[:10])}"] if missing else []


STANCES = {"supports", "contradicts", "mixed", "context_only"}


def check_literature(d):
    errors = check_jsonl(d, "literature.jsonl", ("title", "genes", "source", "status"))
    if errors:
        return errors
    rows = read_jsonl(d / "literature.jsonl")
    read = [r for r in rows if r.get("status") == "read"]
    if not read:
        errors.append("literature.jsonl has no paper with status read (search hits are only candidates)")
    for r in read:
        if not r.get("finding"):
            errors.append(f"read paper {r.get('pmid') or r.get('title')} has no finding")
        if r.get("stance") not in STANCES:
            errors.append(f"read paper {r.get('pmid') or r.get('title')} has stance {r.get('stance')!r}")
    return errors[:20]


def check_regulator_context(d):
    return [] if (d / "regulator_context.md").exists() else ["regulator_context.md missing"]


def check_label(d):
    label = read_json(d / "label.json")
    if label is None:
        return ["label.json missing"]
    errors = [f"label.json lacks {k}" for k in ("program_name", "function_sentence") if not label.get(k)]
    if errors:
        return errors
    if len(label["program_name"].split()) > 8:
        errors.append("program_name has more than 8 words")
    known = set(top_genes(d, 300)) | set(regulator_names(d))
    if not any(g in known for g in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]+", label["function_sentence"])):
        errors.append("function_sentence names no gene of the program (top 300 genes or regulators)")
    return errors


def check_driver_genes(d):
    block, label = read_json(d / "driver_genes.json"), read_json(d / "label.json") or {}
    if block is None:
        return ["driver_genes.json missing"]
    errors = []
    if block.get("annotation") != label.get("program_name"):
        errors.append(f"driver_genes.json was made for label {block.get('annotation')!r}, "
                      f"not {label.get('program_name')!r}")
    read = {str(r.get("pmid")) for r in read_jsonl(d / "literature.jsonl") if r.get("status") == "read" and r.get("pmid")}
    genes = block.get("genes") or {}
    for g, v in genes.items():
        if v.get("verdict") not in VERDICTS:
            errors.append(f"driver gene {g} has verdict {v.get('verdict')!r}")
        cited = [str(p) for p in v.get("pmids") or []] + [str(f.get("pmid")) for f in v.get("paper_findings") or []
                                                           if f.get("pmid")]
        unread = sorted(set(cited) - read)
        if unread:
            errors.append(f"driver gene {g} cites PMIDs that are not read papers in literature.jsonl: {unread[:5]}")
    for g in block.get("explained") or []:
        if g not in genes:
            errors.append(f"explained gene {g} is not among the judged genes")
        elif genes[g].get("verdict") not in ("supported", "mixed"):
            errors.append(f"explained gene {g} has verdict {genes[g].get('verdict')!r}")
    return errors[:20]


CONFOUNDERS = ("positional", "cell_cycle", "technical_qc", "essentiality_or_growth_arrest", "rna_processing_or_decay",
               "cis_target_effects", "ribosome_translation_housekeeping", "other_technical")
CONFOUNDER_STATUS = {"primary_explanation", "contributing", "ruled_out", "cannot_assess"}
SLOTS = ("upstream_trigger", "coregulation_mechanism", "cellular_output")
CONFIDENCE = {"high", "medium", "low"}
STRENGTH = {"supported", "suggestive", "speculative"}
VERDICTS = {"supported", "mixed", "contradicted", "insufficient"}


def check_interpretation(d):
    interp, label = read_json(d / "interpretation.json"), read_json(d / "label.json") or {}
    ev = read_json(d / "evidence.json")
    if interp is None:
        return ["interpretation.json missing"]
    if ev is None:
        return ["evidence.json missing: run the program-annotator build_evidence script first"]
    errors = [f"interpretation.json lacks {k}" for k in ("label", "confounders", "interpretation", "modules",
                                                          "coherence") if k not in interp]
    if errors:
        return errors
    if interp["label"] != label.get("program_name"):
        errors.append(f"interpretation.json was made for label {interp['label']!r}, not {label.get('program_name')!r}")
    if ev.get("label") != label.get("program_name"):
        errors.append("evidence.json was built for another label: rebuild it")
    allowed = {k: set(v) for k, v in ev["allowed"].items()}
    ids = set(ev["evidence"])

    def grounded(where, genes=(), regs=(), pmids=(), eids=()):
        for kind, values, pool in (("gene", genes, allowed["genes"]), ("regulator", regs, allowed["regulators"]),
                                   ("PMID", pmids, allowed["pmids"]), ("evidence id", eids, ids)):
            bad = [str(v) for v in values if str(v).removeprefix("PMID").strip(": ") not in pool]
            if bad:
                errors.append(f"{where}: {kind}(s) not in the evidence: {', '.join(bad[:8])}")

    statuses = {c.get("confounder"): c.get("status") for c in interp["confounders"]}
    missing = [c for c in CONFOUNDERS if c not in statuses]
    if missing:
        errors.append(f"confounders missing: {', '.join(missing)}")
    errors += [f"confounder {c} has status {s!r}" for c, s in statuses.items() if s not in CONFOUNDER_STATUS]
    for slot in SLOTS:
        s = (interp["interpretation"] or {}).get(slot)
        if s is None:
            continue
        if not s.get("claim") or s.get("confidence") not in CONFIDENCE:
            errors.append(f"{slot}: needs a claim and a confidence in {sorted(CONFIDENCE)}")
        if not s.get("evidence_ids"):
            errors.append(f"{slot}: cites no evidence id")
        grounded(slot, s.get("support_genes", []), s.get("support_regulators", []), s.get("pmids", []),
                 s.get("evidence_ids", []))
    for i, m in enumerate(interp["modules"], 1):
        if m.get("strength") not in STRENGTH:
            errors.append(f"module {i}: strength {m.get('strength')!r}")
        grounded(f"module {i}", m.get("genes", []), (), (), m.get("evidence_ids", []))
    if interp["coherence"] not in {"strong", "partial", "weak", "none"}:
        errors.append(f"coherence {interp['coherence']!r}")
    return errors[:30]


CHECKS = {"prepare": check_prepare, "confounder_screens": check_confounder_screens, "gene_info": check_gene_info,
          "interactions": check_interactions, "literature": check_literature,
          "regulator_context": check_regulator_context, "label": check_label, "driver_genes": check_driver_genes,
          "interpretation": check_interpretation}
assert set(CHECKS) == set(STEP_IDS)


def validate_step(program_dir, step_id):
    return CHECKS[step_id](Path(program_dir))


def main():
    p = argparse.ArgumentParser(description="Check a program's step outputs.")
    p.add_argument("--program_dir", required=True)
    p.add_argument("--step", choices=STEP_IDS, default=None, help="One step. Default: every step marked done.")
    args = p.parse_args()
    d = Path(args.program_dir)
    steps = [args.step] if args.step else [s.id for s in read_plan(d).steps if s.status == "done"]
    n_bad = 0
    for sid in steps:
        errors = validate_step(d, sid)
        n_bad += bool(errors)
        print(f"{sid}: " + ("ok" if not errors else "FAILED\n" + "\n".join(f"  - {e}" for e in errors)))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
