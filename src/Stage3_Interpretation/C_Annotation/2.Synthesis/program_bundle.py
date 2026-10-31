"""Read a run's program folders (Programs/P<k>/) in the shape the report code was written for.

The viewer, site and report (program_viewer.py, program_site.py, html_report.py) render a per-program "bundle" dict
(program_genes, perturbation_regulators, GO, gene_info, gene_interactions, program_enrichment, literature_search,
literature_evidence.driver_gene, ...). load_run() builds those bundles from the program folders, so the rendering code
stays as it is and only its loaders read the new layout:

    bundles, labels, interpretations = load_run(run_dir, programs=None, top_gene=15, top_unique=8, top_go=10)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0.Prepare"))
from program_dir import list_programs, read_json, read_jsonl, read_tsv  # noqa: E402

GENE_SOURCES = ("MyGene", "NCBI", "UniProt")


def _gene_entry(gene, info, function_papers):
    gi = {k: v for k, v in (info.get(gene) or {}).items() if k in GENE_SOURCES}
    if function_papers.get(gene):
        gi["Gene_function_categorization"] = {
            "found": True,
            "papers": function_papers[gene],
        }
    return {"gene": gene, "gene_info": gi}


def program_bundle(d, top_gene=15, top_unique=8, top_go=10):
    """The legacy bundle of one program folder."""
    prog = read_json(d / "program.json") or {}
    genes = read_tsv(d / "genes_top300.tsv")
    info = {r["gene"]: r for r in read_jsonl(d / "gene_info.jsonl")}
    lit = read_jsonl(d / "literature.jsonl")
    read = [r for r in lit if r.get("status") == "read"]
    function_papers = {}
    for r in read:  # papers read for a gene's own function, shown per gene
        if r.get("source") in ("gene-function-papers-skill", "generif-papers-skill"):
            for g in r.get("genes") or []:
                function_papers.setdefault(g, []).append(
                    {
                        "pmid": r.get("pmid"),
                        "title": r.get("title"),
                        "year": r.get("year"),
                        "journal": r.get("journal"),
                        "doi": r.get("doi"),
                        "claim": r.get("finding"),
                        "cell_line_or_tissue": r.get("context"),
                        "pdf_path": r.get("pdf_path"),
                    }
                )
    top = [g["gene"] for g in genes[:top_gene]]
    distinct = [
        g["gene"]
        for g in sorted(
            (g for g in genes if g["distinctive_score"] and g["gene"] not in top),
            key=lambda g: -float(g["distinctive_score"]),
        )[:top_unique]
    ]
    regs = {}
    for r in read_tsv(d / "regulators_sig.tsv"):
        regs.setdefault(r["condition"], []).append(
            {
                "gene": r["regulator"],
                "log2fc": round(float(r["log2fc"]), 2),
                "adj_pval": float(r["adj_p"]),
                "gene_info": _gene_entry(r["regulator"], info, function_papers)[
                    "gene_info"
                ],
            }
        )
    bundle = {
        "program_id": d.name,
        "organism": prog.get("organism", "human"),
        "cell_type": prog.get("cell_type", ""),
        "conditions": prog.get("conditions", []),
        "program_specificity": prog.get("program_specificity", {}),
        "GO": [t["term"] for t in read_tsv(d / "go_sig.tsv")[:top_go]],
        "program_genes": [_gene_entry(g, info, function_papers) for g in top],
        "distinctive_genes": [_gene_entry(g, info, function_papers) for g in distinct],
        "perturbation_regulators": regs,
    }
    cs = read_json(d / "confounder_screens.json")
    if cs:
        bundle["confounder_screens"] = cs
    pairs = read_jsonl(d / "interactions.jsonl")
    if pairs:
        tested, found = (
            {},
            {},
        )  # per query category (program_gene-regulator, distinctive_gene-regulator)
        for p in pairs:
            cat = p.get("query_category", "program_gene-regulator")
            tested[cat] = tested.get(cat, 0) + 1
            found[cat] = found.get(cat, 0) + bool(p.get("found"))
        bundle["gene_interactions"] = {
            "OmniPath": {
                "pairs": {
                    p.get("pair", f"{p['gene']}-{p['regulator']}"): p for p in pairs
                },
                "n_regulators": len({p["regulator"] for p in pairs}),
                "n_pairs_tested": tested,
                "n_pairs_found": found,
            }
        }
    sg = read_json(d / "string_genes.json")
    if sg:
        terms = [
            {
                **t,
                "fdr": float(t["fdr"]),
                "genes": t["genes"].split(",") if t["genes"] else [],
            }
            for t in read_tsv(d / "string_sig.tsv")
        ]
        bundle["program_enrichment"] = {
            "STRING": {
                **sg,
                "enrichment": terms,
                "n_top_genes": sg.get("n_genes"),
                "n_terms_significant": len(terms),
            }
        }
    if read:
        bundle["literature_search"] = {
            "agent": {
                "papers": [
                    {
                        "title": r.get("title"),
                        "finding": r.get("finding"),
                        "genes_mentioned": r.get("genes") or [],
                        "context_match": r.get("context"),
                        "pmid": r.get("pmid"),
                        "doi": r.get("doi"),
                        "url": r.get("url"),
                        "year": r.get("year"),
                        "stance": r.get("stance"),
                    }
                    for r in read
                ]
            }
        }
    dg = read_json(d / "driver_genes.json")
    if dg:
        agent = read_json(d / "_runs" / "agent.json") or {}
        judge = (
            f"{agent['model']} (Claude agent, from the papers it read)"
            if dg.get("method") == "agent-literature" and agent.get("model")
            else (dg.get("params") or {}).get("judge_model", "")
        )
        bundle["literature_evidence"] = {
            "driver_gene": {
                "annotation": dg.get("annotation"),
                "explained": dg.get("explained", []),
                "params": {"judge_model": judge},
                "genes": {
                    g: {
                        "verdict": v.get("verdict"),
                        "cell_type_evidence": v.get("cell_type_evidence"),
                        "judge": {
                            "why": v.get("reason"),
                            "reason": v.get("reason"),
                            "paper_findings": [
                                {"title": f["title"], "finding": f["finding"]}
                                for f in v.get("paper_findings") or []
                                if f.get("title") and f.get("finding")
                            ],
                        },
                    }
                    for g, v in (dg.get("genes") or {}).items()
                },
            }
        }
    return bundle


def load_run(run_dir, programs=None, top_gene=15, top_unique=8, top_go=10):
    """({pid: bundle}, {pid: label}, {pid: interpretation}) of the program folders that have a label."""
    bundles, labels, interps = {}, {}, {}
    for d in list_programs(run_dir):
        pid = int(d.name[1:])
        if programs and pid not in programs:
            continue
        bundles[pid] = program_bundle(d, top_gene, top_unique, top_go)
        label = read_json(d / "label.json")
        if label:
            if not label.get(
                "model"
            ):  # agent-written labels: the model is in the agent's run record
                agent = read_json(d / "_runs" / "agent.json") or {}
                label["model"] = (
                    f"{agent['model']} (Claude agent)"
                    if agent.get("model")
                    else "Claude agent"
                )
            labels[pid] = label
        interp = read_json(d / "interpretation.json")
        if interp and label and interp.get("label") == label.get("program_name"):
            interps[pid] = interp
    return bundles, labels, interps
