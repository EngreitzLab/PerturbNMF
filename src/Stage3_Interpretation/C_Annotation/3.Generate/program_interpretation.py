"""Interpret each labeled program in layers with Claude: confounder rule-outs, the biology at three levels
(upstream trigger, co-regulation mechanism, cellular output) and the program's modules.

Reads '<info_dir>/P<k>.json' (AGeneTic bundle) and '<label_dir>/P<k>.json' (program_label.py) and writes
'<out_dir>/P<k>.json':

    {"program_id", "label", "confounders": [{confounder, status, evidence}],
     "interpretation": {"upstream_trigger" | "coregulation_mechanism" | "cellular_output":
                        {claim, mechanism_kind, support_genes, support_regulators, support_terms,
                         confidence, pmids} | null},
     "modules": [{name, genes, mechanism, strength}], "coherence", "dropped", "model", "source", "created"}

The label is given, not chosen: the interpretation explains the current program_label.py label. The prompt
holds the top-loading, distinctive and screened genes (MyGene names when present), the regulators with their
log2FC sign, the GO terms, the pre-annotation screens (0.Prepare/build_confounder_screens.py), the driving
gene verdicts (2.Evidence_curation/query_driver_gene.py), the literature summary, optionally the TF motif
enrichment (--motif_enrichment_path, selected by motif_selection.py), and a reference pool of the papers the
pipeline retrieved. Genes, regulators and PMIDs the answer names that are not in the evidence are removed
and listed under "dropped". The input bundles are never modified. Cached files are reused unless
--overwrite, or when the label they were made for has changed. Env: ANTHROPIC_API_KEY (or C_Annotation/.env).
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "1.Search"))
sys.path.insert(0, str(HERE.parent / "api"))
from budget import add_budget_args, attach  # noqa: E402
from html_report import gene_entries, gene_name, load_bundles, select_regulators  # noqa: E402
from program_label import gene_line, literature_summary  # noqa: E402
from program_site import driver_block, family_name, load_labels  # noqa: E402

CONFOUNDERS = ("positional", "cell_cycle", "technical_qc", "essentiality_or_growth_arrest",
               "rna_processing_or_decay", "cis_target_effects", "ribosome_translation_housekeeping",
               "other_technical")
SLOTS = ("upstream_trigger", "coregulation_mechanism", "cellular_output")

SYSTEM_PROMPT = """You are a cell biologist interpreting gene programs found by consensus NMF on a CRISPRi \
Perturb-seq screen. Each program already has a label; your job is to explain it, not to rename it.

A program is a set of genes whose expression co-varies across cells. Co-variation has many possible \
causes and only some of them are pathways. So work in two passes: first rule out the confounders that \
masquerade as biology, then read the remaining biology in layers: what triggers the program, what makes \
these particular genes move together, and what the cells end up doing. Those layers are usually all true \
at once; do not choose between them.

RULES
1. EVIDENCE ONLY. Every gene, regulator, term and PMID you name must appear in the PROGRAM EVIDENCE. Never \
add one from memory.
2. NAME YOUR SUPPORT. Every claim names the genes, regulators or terms behind it. A claim resting on fewer \
than 2 items is a speculation and must be labeled as one (confidence low).
3. REGULATORS are the knockdowns in section C only; a motif candidate TF is not a regulator.
4. CITATIONS ARE SELECTED, NOT RECALLED. Cite only PMIDs from the REFERENCE POOL, and only where the listed \
finding states what you claim. No citation is better than a guessed one.
5. PERTURBATION SIGN. log2FC is the effect of KNOCKING DOWN the regulator on program activity. Negative = \
the regulator activates the program; positive = it represses it. Every mechanism must be consistent with \
the signs, or say why not.
6. NO SIGN-RATIONALIZATION. A shared direction is a correlation, not a mechanism.
7. DO NOT FORCE-FIT. Chromosomal segments, technical artifacts and essentiality drag produce programs too, \
and some programs combine several unrelated processes. Say so through the confounder statuses, the \
coherence field and the modules, rather than telling a single confident story the genes do not support."""

TASK = """# TASK

Step 1 - RULE OUT THE CONFOUNDERS. Give one entry for each of: {confounders}. Status is one of \
primary_explanation, contributing, ruled_out, cannot_assess. A gene set tested with 0 of the top genes is \
evidence against that confounder (ruled_out), not missing evidence; use cannot_assess only when nothing \
in the evidence bears on it. In `evidence`, cite the number from section E \
(or the genes) that decided it. Section E is deterministic; trust its numbers over intuition. The cis-target \
overlap is for artifact detection only (cis_target_effects). For other_technical, name what you assessed.

Step 2 - READ THE BIOLOGY IN LAYERS. Fill whichever of these slots the evidence supports and leave the others \
null:
  - upstream_trigger: what turns this program on in this system
  - coregulation_mechanism: why THESE genes move together (a named TF or TF family, a chromatin complex, a \
shared RNA-decay route, a shared locus, a signaling cascade, or another mechanism you can describe); put a \
short name of the kind of mechanism in mechanism_kind
  - cellular_output: what the cells are doing as a result
Each slot: one claim (1-3 sentences), its support genes / regulators / terms, a confidence, and PMIDs from \
the reference pool (may be empty). If a confounder is the primary explanation, say so in the slots rather \
than building a pathway story on it.

Step 3 - MODULES. 0-5 subsets of the program's genes that share a narrower biological process or mechanism \
than the label: a name, the genes (from the evidence), the mechanism in one sentence, and a strength \
(supported = several genes and a term back it; suggestive; speculative). A module is biology: do not make \
modules of leftover / unassigned genes or of artifacts (those belong in the confounders and coherence).

Step 4 - COHERENCE: strong (one theme explains most genes), partial, weak, or none."""


class Confounder(BaseModel):
    confounder: str
    status: Literal["primary_explanation", "contributing", "ruled_out", "cannot_assess"]
    evidence: str


class Slot(BaseModel):
    claim: str
    mechanism_kind: str = ""
    support_genes: List[str]
    support_regulators: List[str]
    support_terms: List[str]
    confidence: Literal["high", "medium", "low"]
    pmids: List[str]


class Module(BaseModel):
    name: str
    genes: List[str]
    mechanism: str
    strength: Literal["supported", "suggestive", "speculative"]


class ProgramInterpretation(BaseModel):
    confounders: List[Confounder]
    upstream_trigger: Optional[Slot]
    coregulation_mechanism: Optional[Slot]
    cellular_output: Optional[Slot]
    modules: List[Module]
    coherence: Literal["strong", "partial", "weak", "none"]


def screen_block(bundle):
    """Section E: the deterministic pre-annotation screens, as numbers."""
    cs = bundle.get("confounder_screens")
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


def driver_lines(bundle, label):
    """Driving-gene verdicts (query_driver_gene.py) for the current label: verdict and the judge's reason."""
    dg = driver_block(bundle, label)
    if not dg:
        return "Not run."
    out = []
    for g, a in dg.get("genes", {}).items():
        j = a.get("judge") or {}
        why = " ".join((j.get("why") or j.get("reason") or a.get("short_excerpt") or "").split())
        out.append(f"- {g}: {a.get('verdict') or 'not asked'}{' (explains the label)' if g in dg.get('explained', []) else ''}"
                   f"{' - ' + why if why else ''}")
    return "\n".join(out)


def reference_pool(bundle):
    """{pmid: line} of the papers the pipeline retrieved for the program's genes (gene function search)."""
    pool = {}
    for g, e in gene_entries(bundle).items():
        papers = ((e.get("gene_info") or {}).get("Gene_function_categorization") or {}).get("papers") or []
        for p in papers:
            pmid = str(p.get("pmid") or "").strip()
            if pmid and pmid not in pool:
                claim = " ".join((p.get("claim") or "").split())
                pool[pmid] = (f"PMID {pmid} [{g}] {p.get('title', '')} ({p.get('year', '')}; "
                              f"{p.get('cell_line_or_tissue') or 'system not stated'}): {claim}")
    return pool


def motif_lines(selection):
    """TF motif enrichment (motif_selection.select_program_motifs) as text, one line per family."""
    if not selection:
        return ""
    labels = {key: label for key, _, label in selection.get("sections", [])}
    out = []
    for key, families in selection["families"].items():
        head = f"{key.split('_')[0]} ({labels.get(key, selection['section_sources'].get(key) or '')})"
        out.append(f"- {head}: {selection['n_significant'][key]} of {selection['n_tested'][key]} motifs significant")
        for f in families:
            motifs = ", ".join(f"{m} {e}x FDR {q:.1e}" for m, e, q in f["motifs"])
            cands = ", ".join(f"{c['tf']} ({c['tier']})" for c in f["candidates"])
            out.append(f"  - family {f['family']}: {motifs}" + (f"; candidate TFs: {cands}" if cands else ""))
    return "\n".join(out)


def evidence(pid, bundle, label, top_regulator, motifs=None):
    """(prompt evidence text, allowed genes, allowed regulators, allowed PMIDs)."""
    entries = gene_entries(bundle)
    regs = select_regulators(bundle, top_regulator)
    reg_lines = [f"{r['gene']}: log2FC {r['log2fc']:+.2f} on knockdown ({r['condition']}, adj p {r['adj_pval']:.2g}) -> "
                 + ("activator (knockdown lowers the program)" if r["log2fc"] < 0 else "repressor (knockdown raises the program)")
                 for r in regs if r.get("log2fc") is not None]
    pool = reference_pool(bundle)
    parts = [
        f"# PROGRAM EVIDENCE - Program {pid} in {bundle.get('organism', '')} {bundle.get('cell_type', '')}",
        f"## A. Current label (explain it; do not rename it)\n{label['program_name']}: {label['function_sentence']}",
        "## B. Genes\nTop-loading genes:\n" + "\n".join(f"- {gene_line(e)}" for e in bundle.get("program_genes", []))
        + "\nDistinctive genes (high in this program, low in others):\n"
        + "\n".join(f"- {gene_line(e)}" for e in bundle.get("distinctive_genes", [])),
        "## C. Significant perturbation regulators\n" + ("\n".join(f"- {x}" for x in reg_lines) or "- none"),
        "## D. Enriched GO terms\n" + ("\n".join(f"- {t}" for t in bundle.get("GO", [])) or "- none"),
        "## E. Pre-annotation screens (deterministic)\n" + screen_block(bundle),
    ]
    if motifs:
        parts.append("## E2. TF motif enrichment near the program genes (correlative)\n" + motif_lines(motifs))
    parts.append("## F. Driving genes: does each gene's own literature explain the label?\n" + driver_lines(bundle, label))
    lit = literature_summary(bundle)
    if lit:
        parts.append("## G. Literature summary (from retrieved papers)\n" + lit)
    parts.append("## H. REFERENCE POOL - the only citable PMIDs\n" + ("\n".join(f"- {x}" for x in pool.values()) or "- none"))

    genes = set(entries) | {gene_name(e) for e in bundle.get("program_genes", []) + bundle.get("distinctive_genes", [])}
    cs = bundle.get("confounder_screens") or {}
    genes |= set((cs.get("screens") or {}).get("top_genes") or [])
    dg = driver_block(bundle, label)
    genes |= set((dg or {}).get("genes", {}))
    if motifs:
        genes |= {c["tf"] for fams in motifs["families"].values() for f in fams for c in f["candidates"]}
    return "\n\n".join(parts), genes, {r["gene"] for r in regs}, set(pool)


def clean(result, genes, regulators, pmids):
    """Answer as stored: the genes / regulators / PMIDs outside the evidence removed and listed in dropped."""
    dropped = {"genes": set(), "regulators": set(), "pmids": set()}

    def keep(values, allowed, kind):
        out = []
        for v in values:
            v = re.sub(r"^PMID[:\s]*", "", str(v).strip(), flags=re.I) if kind == "pmids" else str(v).strip()
            (out.append(v) if v in allowed else dropped[kind].add(v))
        return out

    interpretation = {}
    for name in SLOTS:
        s = getattr(result, name)
        interpretation[name] = None if s is None else {
            "claim": s.claim.strip(), "mechanism_kind": s.mechanism_kind.strip(),
            "support_genes": keep(s.support_genes, genes, "genes"),
            "support_regulators": keep(s.support_regulators, regulators, "regulators"),
            "support_terms": [t.strip() for t in s.support_terms], "confidence": s.confidence,
            "pmids": keep(s.pmids, pmids, "pmids")}
    modules = [{"name": m.name.strip(), "genes": keep(m.genes, genes, "genes"), "mechanism": m.mechanism.strip(),
                "strength": m.strength} for m in result.modules]
    order = {c: i for i, c in enumerate(CONFOUNDERS)}
    confounders = sorted(({"confounder": c.confounder.strip(), "status": c.status, "evidence": c.evidence.strip()}
                          for c in result.confounders), key=lambda c: order.get(c["confounder"], len(order)))
    return {"confounders": confounders, "interpretation": interpretation, "modules": modules,
            "coherence": result.coherence, "dropped": {k: sorted(v) for k, v in dropped.items() if v}}


def build_parser():
    p = argparse.ArgumentParser(description="Write a Claude-generated layered interpretation per labeled program.")

    # IO
    p.add_argument("--info_dir", required=True, help="Folder of P<k>.json bundles (Literature_info_extended_PerturbNMF_Info recommended).")
    p.add_argument("--label_dir", default=None, help="program_label.py output. Default: <info_dir>/../Program_label.")
    p.add_argument("--out_dir", default=None, help="Output folder. Default: <info_dir>/../Program_interpretation.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids, space separated (e.g. 1 2 3). Default: every P<k>.json in --info_dir.")
    p.add_argument("--top_regulator", type=int, default=6, help="Regulators passed to the prompt (best adj_pval over conditions).")
    p.add_argument("--motif_enrichment_path", default=None, help="Stage 2 motif enrichment table (<K>_motif_enrichment.txt); adds the TF motif evidence (needs pandas). Default: none.")
    p.add_argument("--candidate_tfs_path", default=None, help="Stage 2 candidate TF table (<K>_candidate_tfs.txt); adds candidate TFs to the motif evidence.")

    # LLM
    p.add_argument("--model", default="claude-sonnet-5", help="Anthropic model id.")
    p.add_argument("--effort", default="medium", help="Effort level: low | medium | high | xhigh | max.")
    p.add_argument("--max_tokens", type=int, default=8000, help="Answer token limit.")
    p.add_argument("--overwrite", action="store_true", help="Redo programs that already have an interpretation for their current label.")
    p.add_argument("--llm_backend", default="api", choices=["api", "session"], help="api = Anthropic API (ANTHROPIC_API_KEY); session = write each prompt to --session_dir and read the answer written there by a Claude Code session (api/session_llm.py; run twice: export, then import).")
    p.add_argument("--session_dir", default=None, help="Request / answer folder for --llm_backend session.")
    p.add_argument("--session_model", default="claude-code-session", help="Model name recorded for session answers.")
    add_budget_args(p, reserve_default=0.0)
    return p


def main():
    from src import load_env, make_handler
    args = build_parser().parse_args()
    load_env()
    bundles = load_bundles(args.info_dir, args.programs)
    label_dir = Path(args.label_dir or Path(args.info_dir).resolve().parent / "Program_label")
    labels = load_labels(label_dir, bundles)
    skipped = [f"P{p}" for p in bundles if p not in labels]
    if skipped:
        print(f"  [warn] no label in {label_dir} for {skipped} (run program_label.py); skipped")
    out_dir = Path(args.out_dir or Path(args.info_dir).resolve().parent / "Program_interpretation")
    out_dir.mkdir(parents=True, exist_ok=True)
    motif_tables = None
    if args.motif_enrichment_path:
        from motif_selection import read_motif_tables, select_program_motifs
        motif_tables = read_motif_tables(args.motif_enrichment_path, args.candidate_tfs_path)
    if args.llm_backend == "session":
        sys.path.insert(0, str(HERE.parent / "api"))
        from session_llm import session_llm
        handler = session_llm(args.session_dir, "interpretation", args.session_model)
    else:
        handler = make_handler(args.model)
    model = getattr(handler, "model_label", args.model)
    task = TASK.format(confounders=" · ".join(CONFOUNDERS))

    n_fail = 0
    for pid, label in labels.items():
        out = out_dir / f"P{pid}.json"
        if out.exists() and not args.overwrite:
            if json.loads(out.read_text()).get("label") == label["program_name"]:
                print(f"  P{pid}: cached ({out})")
                continue
            print(f"  P{pid}: label changed since the cached interpretation; redoing")
        motifs = select_program_motifs(pid, *motif_tables) if motif_tables else None
        text, genes, regulators, pmids = evidence(pid, bundles[pid], label, args.top_regulator, motifs)
        attach(handler, args, f"P{pid}", "interpretation")
        handler.key = f"P{pid}"
        result = handler.get_json(SYSTEM_PROMPT, f"{text}\n\n---\n\n{task}", ProgramInterpretation,
                                  effort=args.effort, max_tokens=args.max_tokens)
        if result is None:
            if args.llm_backend == "session":
                continue
            print(f"  P{pid}: [fail] no interpretation returned")
            n_fail += 1
            continue
        record = {"program_id": f"P{pid}", "label": label["program_name"], **clean(result, genes, regulators, pmids),
                  "model": model, "source": str(Path(args.info_dir).resolve() / f"P{pid}.json"),
                  "created": datetime.now().isoformat(timespec="seconds")}
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"  P{pid}: coherence {record['coherence']}, {len(record['modules'])} module(s)"
              + (f", dropped {record['dropped']}" if record["dropped"] else ""))
    print(f"[done] {len(labels) - n_fail}/{len(labels)} program(s) -> {out_dir}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
