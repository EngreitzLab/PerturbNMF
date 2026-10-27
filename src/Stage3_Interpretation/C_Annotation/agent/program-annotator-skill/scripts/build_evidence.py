#!/usr/bin/env python3
"""Write the numbered evidence block of one program for the interpretation step.

    Programs/P<k>/evidence.md    sections A-H; every bullet of C-G carries an id [E01], [E02], ...
    Programs/P<k>/evidence.json  {"evidence": {E01: text}, "allowed": {genes, regulators, pmids}, "inputs": {file: sha256}}

interpretation.json may name only genes, regulators and PMIDs in "allowed" and cite only these evidence ids;
the interpretation step's checks (1.Annotate/validate_program.py) enforce it. Needs label.json and the outputs of the
earlier steps (missing optional inputs are reported as "Not run"). Reads one JSON object on stdin
({"program_dir": ...}) and writes one JSON object on stdout.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.program_files import family_name, gene_card, read_json, read_jsonl, read_tsv, regulators  # noqa: E402
from _lib.skill_lib import SkillError, require_str, run  # noqa: E402

TAGGED = ("## C", "## D", "## E", "## F", "## G")
INPUTS = ("program.json", "label.json", "genes_top300.tsv", "regulators_sig.tsv", "go_sig.tsv", "string_sig.tsv",
          "string_genes.json", "confounder_screens.json", "gene_info.jsonl", "interactions.jsonl", "literature.jsonl",
          "regulator_context.md", "driver_genes.json")


def screen_block(cs):
    """Section E: the deterministic confounder screens, as numbers."""
    if not cs:
        return "Not run."
    sc, lab = cs["screens"], cs["label"]
    lines = [f"Screened: the top {sc.get('n_top_genes_screened')} genes: {', '.join(sc.get('top_genes') or [])}"]
    empty = []
    for kind in ("symbol_families", "marker_sets"):
        for key, hit in (sc.get(kind) or {}).items():
            if hit.get("n_overlap"):
                lines.append(f"{family_name(key)}: {hit['n_overlap']} of the top genes (expected {hit.get('expected')}, "
                             f"p = {hit.get('p_value')}): {', '.join(hit.get('genes') or [])}")
            else:
                empty.append(family_name(key))
    if empty:   # tested with no overlap: evidence against, not missing evidence
        lines.append(f"Tested, 0 of the top genes: {', '.join(empty)}")
    lines.append(f"Called families: {', '.join(family_name(d['family']) for d in lab.get('family_details', [])) or 'none'}")
    pos = sc.get("positional") or {}
    if pos.get("assessable"):
        lines.append(f"Positional: most located top genes on {pos['top_chromosome']} ({pos['genes_on_top_chromosome']} of "
                     f"{pos['n_genes_located']}, expected {pos['expected_on_top_chromosome']}, fold {pos['fold_enrichment']}, "
                     f"binomial p = {pos['binomial_p']}): {', '.join(pos.get('top_chromosome_genes') or [])}"
                     + (f"; flagged positional ({lab['positional_flag']})" if lab.get("positional_flag") else ""))
    else:
        lines.append("Positional: not assessable")
    if sc.get("gene_biotypes"):
        lines.append("Gene biotypes: " + ", ".join(f"{k} {v}" for k, v in sc["gene_biotypes"].items()))
    cis = sc.get("cis_targets_in_top_genes") or {}
    lines.append(f"Cis targets (top genes that are themselves CRISPRi targets in this screen): {cis.get('n_overlap', 0)}"
                 + (f": {', '.join(cis.get('genes') or [])}" if cis.get("genes") else ""))
    regs = sc.get("regulators") or {}
    if regs:
        lines.append(f"Regulators: {regs.get('n_significant')} of {regs.get('n_tested')} knockdowns significant")
    return "\n".join(f"- {x}" for x in lines)


def tag(parts):
    """Number every bullet of the TAGGED sections; returns (parts, {id: text})."""
    ledger, out = {}, []
    for part in parts:
        if not part.startswith(TAGGED):
            out.append(part)
            continue
        lines = []
        for line in part.split("\n"):
            m = re.match(r"^(\s*)- (.*)$", line)
            if m:
                eid = f"E{len(ledger) + 1:02d}"
                ledger[eid] = m.group(2)
                line = f"{m.group(1)}- [{eid}] {m.group(2)}"
            lines.append(line)
        out.append("\n".join(lines))
    return out, ledger


def handle(req):
    d = Path(require_str(req, "program_dir"))
    label = read_json(d / "label.json")
    if not label:
        raise SkillError("not_found", f"{d / 'label.json'} is missing: the label step comes before the evidence")
    prog = read_json(d / "program.json") or {}
    n_genes, n_distinct = int(req.get("n_genes", 15)), int(req.get("n_distinctive", 8))
    n_terms = int(req.get("n_terms", 15))
    genes = read_tsv(d / "genes_top300.tsv")
    top = [g["gene"] for g in genes[:n_genes]]
    distinct = [g["gene"] for g in sorted((g for g in genes if g["distinctive_score"] and g["gene"] not in top),
                                          key=lambda g: -float(g["distinctive_score"]))[:n_distinct]]
    info = {r["gene"]: r for r in read_jsonl(d / "gene_info.jsonl")}
    regs = regulators(d)

    def gene_line(g):
        name = gene_card(info.get(g, {}))["name"]
        return f"{g} ({name})" if name else g

    parts = [f"# PROGRAM EVIDENCE - {d.name} in {prog.get('organism', '')} {prog.get('cell_type', '')}".rstrip(),
             f"## A. Current label (explain it; do not rename it)\n{label['program_name']}: {label['function_sentence']}",
             "## B. Genes\nTop-loading genes:\n" + "\n".join(f"- {gene_line(g)}" for g in top)
             + "\nDistinctive genes (high in this program, low in others):\n"
             + "\n".join(f"- {gene_line(g)}" for g in distinct),
             "## C. Significant perturbation regulators (log2FC of the program on knockdown)\n"
             + ("\n".join(f"- {g}: log2FC {float(r['log2fc']):+.2f} on knockdown ({r['condition']}, adj p "
                          f"{float(r['adj_p']):.2g}) -> " + ("activator (knockdown lowers the program)"
                                                            if r["role"] == "activator"
                                                            else "repressor (knockdown raises the program)")
                          for g, r in list(regs.items())[:15]) or "- none")]
    ctx = (d / "regulator_context.md")
    if ctx.exists():
        body = ctx.read_text().split("\n", 2)[-1].strip()
        parts += [s.strip() for s in re.split(r"\n(?=## )", body) if s.strip()]
    gos = read_tsv(d / "go_sig.tsv")[:n_terms]
    parts.append("## D. Enriched GO terms (top by adjusted p)\n"
                 + ("\n".join(f"- {t['term']} (adj p {t['adj_p']}, {t['overlap']})" for t in gos) or "- none"))
    sg = read_json(d / "string_genes.json") or {}
    ppi = sg.get("ppi_enrichment") or {}
    terms = read_tsv(d / "string_sig.tsv")[:n_terms]
    parts.append("## D2. STRING enrichment of the top genes"
                 + (f" (PPI: {ppi.get('number_of_edges')} edges vs {ppi.get('expected_number_of_edges')} expected, "
                    f"p = {ppi.get('p_value')})" if ppi else "") + "\n"
                 + ("\n".join(f"- {t['category']}: {t['description']} (FDR {float(t['fdr']):.2g}, {t['n_genes']} genes)"
                              for t in terms) or "- none"))
    parts.append("## E. Pre-annotation screens (deterministic)\n" + screen_block(read_json(d / "confounder_screens.json")))
    dg = read_json(d / "driver_genes.json")
    if dg and dg.get("annotation") == label["program_name"]:
        parts.append("## F. Driving genes: does each gene's own literature explain the label?\n"
                     + "\n".join(f"- {g}: {v.get('verdict')}" + (f" - {v['reason']}" if v.get("reason") else "")
                                 for g, v in (dg.get("genes") or {}).items()))
    lit = [r for r in read_jsonl(d / "literature.jsonl") if r.get("status") == "read" and r.get("finding")]
    parts.append("## G. Literature findings (papers read in the literature step)\n"
                 + ("\n".join(f"- [{('PMID ' + r['pmid']) if r.get('pmid') else (r.get('doi') or 'no id')}] "
                              f"{', '.join(r.get('genes') or [])}: {r['finding']} ({r.get('stance', 'context_only')}"
                              + (f"; {r['context']}" if r.get("context") else "") + f"; read from {r.get('read_from', '?')})"
                              for r in lit) or "- none read"))
    pool = {}
    for r in lit:
        if r.get("pmid"):
            pool.setdefault(str(r["pmid"]), f"PMID {r['pmid']} [{', '.join(r.get('genes') or [])}] {r.get('title', '')} "
                                            f"({r.get('year', '')}): {r['finding']}")
    for p in read_jsonl(d / "interactions.jsonl"):
        if p.get("found") and p.get("regulator") in regs:
            for x in (p.get("pmids") or [])[:3]:
                pool.setdefault(str(x), f"PMID {x} [OmniPath curated reference] {p['regulator']} - {p['gene']}: "
                                        f"{', '.join(p.get('interaction_categories') or [])}")
    parts.append("## H. REFERENCE POOL - the only citable PMIDs\n" + ("\n".join(f"- {x}" for x in pool.values()) or "- none"))

    parts, ledger = tag(parts)
    cs = read_json(d / "confounder_screens.json") or {}
    allowed_genes = ({g["gene"] for g in genes} | set(regs) | set(info)
                     | set((cs.get("screens") or {}).get("top_genes") or []) | set((dg or {}).get("genes") or {}))
    (d / "evidence.md").write_text("\n\n".join(parts) + "\n")
    (d / "evidence.json").write_text(json.dumps({
        "label": label["program_name"], "evidence": ledger,
        "allowed": {"genes": sorted(allowed_genes), "regulators": sorted(regs), "pmids": sorted(pool)},
        "inputs": {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in INPUTS if (d / f).exists()}},
        indent=2))
    return {"program_dir": str(d), "n_evidence": len(ledger), "n_citable_pmids": len(pool),
            "written": [str(d / "evidence.md"), str(d / "evidence.json")]}


if __name__ == "__main__":
    sys.exit(run(handle))
