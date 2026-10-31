"""Write one self-contained HTML viewer of every labeled program (no external files or libraries).

Reads the run's program folders (<run_dir>/Programs/P<k>/, through program_bundle.load_run) and writes one
file: a top bar (search, previous / next, light / dark theme), a rail of programs grouped by the
pre-annotation family label, and one program at a time. The content is the same as program_site.py's pages
(it shares their selection helpers); the layout follows the single-file annotation viewer.

Per program, in page order: the name and one-sentence function; program activity per condition
(program_specificity); top genes (outlined = driving gene) and top unique (distinctive) genes; driving genes
when the optional driver_genes step has run for the current label (verdict, why, supporting
papers, the gene's STRING terms shown to the judge, other literature searched); significant regulators (an
inline SVG volcano per condition and a regulator x condition log2FC table, with --perturbation_path_base;
then the top regulators given to the annotator); TF motif enrichment (with --motif_enrichment_path: per
element type and motif source, the top significant motif families with logos and candidate TFs, selected
by motif_selection.py); confounder rule-outs and layered interpretation when interpretation.json was made
for the current label; regulator-gene interactions (an interactive Cytoscape.js graph, inlined, of the top
regulators x top and unique genes; edges = every OmniPath interaction found between them, from interactions.jsonl,
and the STRING links among the shown regulators, dotted and colored by type, with a table of the links among all
significant regulators, from string_genes.json); the interpretation's modules; top GO terms (a bar plot of -log10 adjusted p per term
from the GO enrichment table, --GO_path or the run's Programs/meta.json). A regulator with negative log2FC on knockdown is an
activator (knocking it down lowers the program); positive is a repressor.
With --home_url the top bar links back to every project (publish_viewer.py sets it).
Programs without label.json are skipped. Without --perturbation_path_base / --motif_enrichment_path only
the standard library is needed (pandas otherwise).
"""

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from html_report import (
    CYTOSCAPE_PATH,
    _js_safe,
    gene_card,
    gene_entries,
    gene_name,
    literature_edge,  # noqa: E402
    omnipath_edge,
    select_regulators,
)
from program_bundle import load_run  # noqa: E402
from program_site import (
    CELL_TYPE_TEXT,
    PUBMED,
    driver_block,
    driver_records,
    family_name,  # noqa: E402
    gene_paper_meta,
    other_paper_records,
    paper_records,
)

TEMPLATE_PATH = HERE / "program_viewer_template.html"
NO_SCREEN, NO_FAMILY = "Not screened", "No family label"


def paper(title, text, meta, same=False):
    """One paper as data: the link rule of program_site.paper_li (PubMed, else DOI, else URL)."""
    pmid = meta.get("pmid")
    url = (
        f"{PUBMED}{pmid}/"
        if pmid
        else f"https://doi.org/{meta['doi']}"
        if meta.get("doi")
        else meta.get("url")
    )
    return {
        "title": title,
        "text": text or "",
        "pmid": str(pmid or ""),
        "year": str(meta.get("year") or ""),
        "url": url or "",
        "same": bool(same),
    }


def other_papers(gene, shown, paper_meta, bundle):
    gf, lit = other_paper_records(gene, shown, paper_meta, bundle)
    return {
        "gene_function": [paper(t, text, m) for t, text, m in gf],
        "literature": [paper(t, text, m, same) for t, text, m, same in lit],
    }


STRING_SOURCE = {
    "Process": "STRING Process",
    "Function": "STRING Function",
    "Component": "STRING Component",
    "KEGG": "STRING KEGG",
    "RCTM": "STRING Reactome",
    "WikiPathways": "STRING WikiPathways",
}


def string_supports(bundle, gene, max_terms=8):
    """The gene's enriched STRING terms the driver-gene judge saw (search_STRING.py, most significant first), as
    database supports {source, term, fdr}; [] without a STRING block."""
    s = (bundle.get("program_enrichment") or {}).get("STRING") or {}
    g = (s.get("genes") or {}).get(gene) or {}
    terms = {t["term"]: t for t in s.get("enrichment", [])}
    return [
        {
            "source": STRING_SOURCE.get(t["category"], "STRING " + t["category"]),
            "term": t["description"],
            "id": t["term"],
            "fdr": t["fdr"],
        }
        for t in (terms[x] for x in g.get("terms", []) if x in terms)
    ][:max_terms]


def driver_data(dg, paper_meta, bundle, min_score=5):
    """Driving genes (the explained genes, with supporting and other papers) and how many genes were searched."""
    genes, others = driver_records(dg)

    def why(a):
        j = a.get("judge") or {}
        return j.get("why") or j.get("reason") or a.get("short_excerpt") or ""

    driving = []
    for g in genes:
        a = dg["genes"][g]
        records, shown = paper_records(a, paper_meta.get(g, {}), min_score)
        driving.append(
            {
                "gene": g,
                "verdict": a.get("verdict") or "",
                "why": why(a),
                "cell_type": CELL_TYPE_TEXT.get(a.get("cell_type_evidence"), ""),
                "string": string_supports(bundle, g),
                "string_used": bool((a.get("judge") or {}).get("uses_string")),
                "supporting": [paper(t, f, m) for t, f, m in records],
                "other": other_papers(g, shown, paper_meta.get(g, {}), bundle),
            }
        )
    return {
        "driving": driving,
        "n_searched": len(genes) + len(others),
        "judge_model": (dg.get("params") or {}).get("judge_model") or "",
    }


def rail_group(bundle):
    cs = bundle.get("confounder_screens")
    if not cs:
        return NO_SCREEN
    fam = cs["label"].get("primary_family")
    return family_name(fam) if fam and fam != "none" else NO_FAMILY


def load_associations(base, conditions, args):
    """{condition: table} of the perturbation association files that exist."""
    import pandas as pd

    out = {}
    for cond in conditions:
        path = f"{base}_{cond}.txt"
        if not Path(path).exists():
            print(f"  [warn] no association file {path}; condition {cond} skipped")
            continue
        df = pd.read_csv(path, sep="\t")
        df[args.program_col] = df[args.program_col].astype(str)
        df[args.target_col] = df[args.target_col].astype(str)
        out[cond] = df
    return out


def regulator_plots(pid, tables, names, index, args):
    """Volcano per condition, compact (regulator index, log2FC x100, -log10 adj p x100, significant), with
    the significance rule of html_Program_QC_plots._build_volcano (adj p <= p_value and log2FC past the up /
    down thresholds); and the regulators with adj p < p_value in any condition (as _build_heatmap), log2FC
    and adj p in every condition, ordered by number of significant conditions then best adj p."""
    volcano, grid = [], {}
    for ci, (cond, df) in enumerate(tables.items()):
        sub = df[df[args.program_col] == str(pid)]
        v = {"condition": cond, "g": [], "x": [], "y": [], "s": []}
        for gene, fc, q in zip(
            sub[args.target_col], sub[args.log2fc_key], sub["adj_pval"]
        ):
            if fc is None or q is None or math.isnan(fc) or math.isnan(q):
                continue
            if gene not in index:
                index[gene] = len(names)
                names.append(gene)
            sig = q <= args.p_value and (
                fc >= args.up_thred_log or fc <= args.down_thred_log
            )
            v["g"].append(index[gene])
            v["x"].append(round(fc * 100))
            v["y"].append(round(-math.log10(max(q, 1e-300)) * 100))
            v["s"].append(int(sig))
            grid.setdefault(gene, [None] * len(tables))[ci] = [
                round(float(fc), 4),
                bool(q < args.p_value),
                float(q),
            ]
        volcano.append(v)
    rows = [
        {"gene": g, "cells": cells}
        for g, cells in grid.items()
        if any(c and c[1] for c in cells)
    ]
    rows.sort(
        key=lambda r: (
            -sum(bool(c and c[1]) for c in r["cells"]),
            min(c[2] for c in r["cells"] if c),
        )
    )
    return volcano, rows


def string_edge(e):
    """One STRING link between two regulators (undirected, unsigned; sign = its interaction type)."""
    return {
        "id": f"e_st_{e['a']}_{e['b']}",
        "source": f"n_{e['a']}",
        "target": f"n_{e['b']}",
        "kind": "string",
        "sign": e["type"],
        "tarrow": "none",
        "sarrow": "none",
        "lstyle": "dotted",
        "width": round(1.2 + 3 * max(e["score"] - 0.7, 0) / 0.3, 2),
        "regulator": e["a"],
        "gene": e["b"],
        "score": e["score"],
        "channels": e.get("channels", []),
        "channel_scores": e.get("channel_scores", {}),
    }


def interaction_network(bundle, regs, genes, uniq):
    """Nodes = the shown top regulators, top genes and unique genes; edges = every found OmniPath pair and paper-qa
    relation between a regulator and one of these genes (html_report's edge records), and the STRING links among the
    shown regulators. regulator_links = the STRING links among all significant regulators (the card's table). None
    without gene_interactions or a STRING regulator network."""
    gi = bundle.get("gene_interactions") or {}
    omnipath, literature = gi.get("OmniPath"), gi.get("Literature")
    rn = ((bundle.get("program_enrichment") or {}).get("STRING") or {}).get(
        "regulator_network"
    )
    if not omnipath and not literature and not rn:
        return None
    reg_names = [r["gene"] for r in regs]
    kind = {g: "gene" for g in genes}
    for g in uniq:
        kind.setdefault(g, "distinctive")
    nodes = [
        {
            "id": f"n_{r['gene']}",
            "label": r["gene"],
            "kind": "regulator",
            "log2fc": r["log2fc"],
            "adj_p": "" if r.get("adj_pval") is None else f"{r['adj_pval']:.2g}",
            "condition": r["condition"],
            "in_program": r["gene"] in kind,
        }
        for r in regs
    ]
    for g, k in kind.items():
        if (
            g not in reg_names
        ):  # a regulator that is also a shown gene is drawn once, as a regulator
            nodes.append(
                {
                    "id": f"n_{g}",
                    "label": g,
                    "kind": k,
                    "rank": (genes if k == "gene" else uniq).index(g) + 1,
                }
            )

    included = [
        p
        for p in ((omnipath or {}).get("pairs") or {}).values()
        if p["regulator"] in reg_names and p["gene"] in kind
    ]
    edges = [omnipath_edge(p) for p in included if p["found"]]
    edges += [
        literature_edge(p)
        for p in ((literature or {}).get("pairs") or {}).values()
        if p.get("source") == "paper-qa"
        and p.get("category")
        and p["regulator"] in reg_names
        and p["gene"] in kind
    ]
    links = (rn or {}).get("edges") or []
    edges += [
        string_edge(e) for e in links if e["a"] in reg_names and e["b"] in reg_names
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "has_omnipath": omnipath is not None,
        "has_literature": literature is not None,
        "has_string": rn is not None,
        "n_tested": len(included),
        "n_found": sum(p["found"] for p in included),
        "regulator_links": links,
        "regulator_ppi": {
            **((rn or {}).get("ppi_enrichment") or {}),
            "background": (rn or {}).get("background"),
            "n_mapped": (rn or {}).get("n_mapped"),
        }
        if rn
        else None,
    }


def load_go_table(path):
    """{program number: {term without its parenthesized GO id: row}} from a GO enrichment table (program_name, Term,
    P-value, Adjusted P-value, Genes, overlap_numerator, overlap_denominator); the lowest adjusted p wins on ties."""
    table = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            term = re.sub(
                r"\s*\([^)]*\)", "", row["Term"]
            ).strip()  # as prepare_programs.significant_go
            prog = table.setdefault(int(row["program_name"]), {})
            if term not in prog or float(row["Adjusted P-value"]) < float(
                prog[term]["Adjusted P-value"]
            ):
                prog[term] = row
    return table


def go_stats(pid, terms, go_table):
    """Per shown GO term (bundle order): adjusted p, p, overlap and genes; None when the table has no row for it."""
    if not go_table:
        return None
    rows = go_table.get(pid, {})
    out = []
    for t in terms:
        r = rows.get(t)
        out.append(
            None
            if r is None
            else {
                "adj_p": float(r["Adjusted P-value"]),
                "p": float(r["P-value"]),
                "n": int(float(r.get("overlap_numerator") or 0)),
                "of": int(float(r.get("overlap_denominator") or 0)),
                "genes": [g for g in (r.get("Genes") or "").split(";") if g],
            }
        )
    return out


def program_motifs(pid, motif_tables):
    """motif_selection.select_program_motifs for this program; None without motif tables."""
    if not motif_tables:
        return None
    from motif_selection import select_program_motifs

    return select_program_motifs(pid, *motif_tables)


def load_interpretation(run_dir, pid, label):
    """Programs/P<k>/interpretation.json when it was made for this label, else None."""
    path = Path(run_dir) / "Programs" / f"P{pid}" / "interpretation.json"
    if not path.exists():
        return None
    it = json.loads(path.read_text())
    if it.get("label") != label["program_name"]:
        print(
            f'  [warn] P{pid}: interpretation was made for "{it.get("label")}", not the current label; not shown'
        )
        return None
    return {
        k: it.get(k)
        for k in ("confounders", "interpretation", "modules", "coherence", "model")
    }


def program_data(
    pid, bundle, label, args, tables, names, index, motif_tables=None, go_table=None
):
    cards = {g: gene_card(e) for g, e in gene_entries(bundle).items()}
    dg = driver_block(bundle, label)
    genes = [gene_name(e) for e in bundle.get("program_genes", [])][: args.top_gene]
    uniq = [gene_name(e) for e in bundle.get("distinctive_genes", [])]
    regs = select_regulators(bundle, args.top_regulator)
    if dg and any(
        (g.get("judge") or {}).get("error")
        for g in dg.get("genes", {}).values()
        if g.get("judge")
    ):
        print(
            f"  [warn] P{pid}: driving-gene judge verdicts still pending; section not shown"
        )
        dg = None
    drivers = driver_data(dg, gene_paper_meta(bundle), bundle) if dg else None
    shown_genes = set(genes) | set(uniq) | {r["gene"] for r in regs}
    if drivers:
        shown_genes |= {d["gene"] for d in drivers["driving"]}
    interpretation = load_interpretation(args.run_dir, pid, label)
    if interpretation:
        shown_genes |= {
            g
            for s in interpretation["interpretation"].values()
            if s
            for g in s.get("support_genes", []) + s.get("support_regulators", [])
        }
        shown_genes |= {
            g for m in interpretation["modules"] for g in m.get("genes", [])
        }
    volcano, grid = (
        regulator_plots(pid, tables, names, index, args) if tables else ([], [])
    )
    cs = bundle.get("confounder_screens") or {}
    return {
        "id": pid,
        "label": label["program_name"],
        "sentence": label["function_sentence"],
        "group": rail_group(bundle),
        "tag": "positional" if (cs.get("label") or {}).get("positional_flag") else "",
        "top_genes": genes,
        "distinctive": uniq,
        "genes": {
            g: {
                "name": cards[g].get("name") or "",
                "entrez": str(cards[g].get("entrez") or ""),
                "symbol": cards[g].get("symbol") or "",
            }
            for g in shown_genes
            if g in cards
        },
        "regulators": [
            {
                **{k: r[k] for k in ("gene", "log2fc", "condition")},
                "adj_p": "" if r.get("adj_pval") is None else f"{r['adj_pval']:.2g}",
            }
            for r in regs
        ],
        "go": (go := [re.sub(r"\s*\(GO:\d+\)", "", t) for t in bundle.get("GO", [])]),
        "go_stats": go_stats(pid, go, go_table),
        "drivers": drivers,
        "volcano": volcano,
        "grid": grid,
        "n_sig_by_condition": [sum(v["s"]) for v in volcano],
        "network": interaction_network(bundle, regs, genes, uniq),
        "activity": (bundle.get("program_specificity") or {}).get("per_condition")
        or {},
        "motifs": program_motifs(pid, motif_tables),
        "interpretation": interpretation,
        "label_model": label.get("model", ""),
        "source": f"{Path(args.run_dir).resolve().name}/Programs/P{pid}",
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Write one self-contained HTML viewer of every labeled program."
    )

    # IO
    p.add_argument(
        "--run_dir",
        required=True,
        help="Annotation run folder (Programs/P<k>/ with label.json, interpretation.json, ...).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output HTML. Default: <run_dir>/Report/program_viewer.html.",
    )

    # context info
    p.add_argument(
        "--dataset_name", required=True, help="Dataset label shown in the top bar."
    )
    p.add_argument(
        "--programs",
        type=int,
        nargs="+",
        default=None,
        help="Program ids, space separated (e.g. 1 2 3). Default: every program folder with a label.",
    )
    p.add_argument(
        "--home_url",
        default="",
        help="Link of the 'All projects' button in the top bar (e.g. ../index.html, the site root of publish_viewer.py). Default: no button; publish_viewer.py turns it on.",
    )

    # content
    p.add_argument(
        "--top_gene", type=int, default=15, help="Top-loading program genes shown."
    )
    p.add_argument(
        "--top_regulator",
        type=int,
        default=6,
        help="Regulators shown (best adj_pval over conditions).",
    )
    p.add_argument(
        "--GO_path",
        default=None,
        help="GO enrichment table (Evaluation/<K>_<thr>/<K>_GO_term_enrichment.txt) for the GO bar plot (-log10 adjusted p per term). Default: the GO_path recorded in <run_dir>/Programs/meta.json; without one the GO terms are listed.",
    )
    p.add_argument(
        "--membership_top",
        type=int,
        default=None,
        help="Top-loaded genes per program used for GO enrichment and unique genes (0.Prepare membership_top); shown in the card captions.",
    )

    # regulator-program association plots (optional)
    p.add_argument(
        "--perturbation_path_base",
        default=None,
        help="Perturbation association results without the _<condition>.txt suffix (e.g. Evaluation/<K>_<thr>/<K>_perturbation_association_results); adds the volcano + log2FC table. Default: no plots.",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions, space separated (files <base>_<condition>.txt). Default: the bundles' conditions.",
    )
    p.add_argument(
        "--target_col",
        default="target_name",
        help="Perturbed-gene column of the association files.",
    )
    p.add_argument(
        "--program_col",
        default="program_name",
        help="Program column of the association files (values match the P<k> number).",
    )
    p.add_argument(
        "--log2fc_key",
        default="log2FC",
        help="log2FC column of the association files (approx_log2FC for CRT results).",
    )
    p.add_argument(
        "--p_value",
        type=float,
        default=0.05,
        help="adj p cutoff for the volcano colors and the log2FC table regulators.",
    )
    p.add_argument(
        "--down_thred_log",
        type=float,
        default=-0.05,
        help="Volcano log2FC threshold for down.",
    )
    p.add_argument(
        "--up_thred_log",
        type=float,
        default=0.05,
        help="Volcano log2FC threshold for up.",
    )

    # TF motif enrichment (optional; Stage 2 run_motif_enrichment.py outputs)
    p.add_argument(
        "--motif_enrichment_path",
        default=None,
        help="Stage 2 motif enrichment table (e.g. Evaluation/<K>_<thr>/<K>_motif_enrichment.txt); adds the TF motif enrichment card. The run's <K>_motif_enrichment_config.yml next to it gives the test (t-test / correlation, top genes). Default: no motif card.",
    )
    p.add_argument(
        "--candidate_tfs_path",
        default=None,
        help="Stage 2 candidate TF table (<K>_candidate_tfs.txt); adds the candidate TFs under each motif family.",
    )
    p.add_argument(
        "--motif_logos_path",
        default=None,
        help="Stage 2 motif logos (<K>_motif_logos.json); adds a sequence logo per motif.",
    )
    p.add_argument(
        "--motif_source_label",
        default="",
        help="Free-text provenance of the motif results, shown under the motif card (e.g. motif database, promoter window, enhancer links).",
    )
    return p


def main():
    args = build_parser().parse_args()
    bundles, labels, _ = load_run(args.run_dir, args.programs, top_gene=args.top_gene)
    skipped = [f"P{p}" for p in bundles if p not in labels]
    if skipped:
        print(f"  [warn] no label.json for {skipped}; skipped")
    if not labels:
        raise SystemExit("No program has a label; nothing to write.")

    first = bundles[next(iter(labels))]
    conditions = args.conditions or first.get("conditions") or []
    tables = (
        load_associations(args.perturbation_path_base, conditions, args)
        if args.perturbation_path_base
        else {}
    )
    motif_tables = None
    if args.motif_enrichment_path:  # pandas only when motif results are shown
        from motif_selection import (
            read_motif_logos,
            read_motif_tables,
            read_motif_test,
            shown_logos,
        )

        motif_tables = read_motif_tables(
            args.motif_enrichment_path, args.candidate_tfs_path
        )
    go_path = args.GO_path
    meta_path = Path(args.run_dir) / "Programs" / "meta.json"
    if not go_path and meta_path.exists():
        go_path = json.loads(meta_path.read_text()).get("inputs", {}).get("GO_path")
    go_table = load_go_table(go_path) if go_path and Path(go_path).exists() else None
    if not go_table:
        print(
            "  [warn] no GO enrichment table (--GO_path); GO terms are listed without a bar plot"
        )
    names, index, programs = [], {}, {}
    for pid in labels:
        programs[pid] = program_data(
            pid,
            bundles[pid],
            labels[pid],
            args,
            tables,
            names,
            index,
            motif_tables,
            go_table,
        )
        net = programs[pid]["network"]
        print(
            f"  P{pid}: {labels[pid]['program_name']}"
            + (
                f" ({sum(e['kind'] != 'string' for e in net['edges'])} regulator-gene interactions, "
                f"{len(net['regulator_links'])} regulator-regulator STRING links)"
                if net
                else ""
            )
        )

    counts = {}
    for p in programs.values():
        counts[p["group"]] = counts.get(p["group"], 0) + 1
    last = {NO_FAMILY: 1, NO_SCREEN: 2}
    groups = sorted(counts, key=lambda g: (last.get(g, 0), -counts[g], g))
    meta = {
        "title": args.dataset_name,
        "conditions": list(tables) or conditions,
        "groups": groups,
        "regulator_names": names,
        "p_value": args.p_value,
        "up": args.up_thred_log,
        "down": args.down_thred_log,
        "membership_top": args.membership_top,
        "built": datetime.now().strftime("%Y-%m-%d"),
        "organism": first.get("organism") or "human",
        "motif_test": read_motif_test(args.motif_enrichment_path)
        if motif_tables
        else None,
        "motif_logos": (
            shown_logos(programs.values(), read_motif_logos(args.motif_logos_path))
            if motif_tables and args.motif_logos_path
            else {}
        ),
        "motif_source": args.motif_source_label,
    }
    out = Path(args.out or Path(args.run_dir) / "Report" / "program_viewer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    cyto = ""
    if any(
        p["network"] for p in programs.values()
    ):  # inline Cytoscape.js only when a graph is drawn
        lib = CYTOSCAPE_PATH.read_text()
        if "</script" in lib.lower():
            raise ValueError(f"{CYTOSCAPE_PATH} contains '</script'; cannot inline it")
        cyto = f"<script>{lib}</script>"
    # small placeholders first, then the data and library blobs (never re-scanned)
    html = (
        TEMPLATE_PATH.read_text()
        .replace(
            "__PAGE_TITLE__",
            args.dataset_name.replace("&", "&amp;").replace("<", "&lt;"),
        )
        .replace("__HOME_URL__", escape(args.home_url, quote=True))
        .replace("__HOME_HIDDEN__", "" if args.home_url else " hidden")
        .replace("__META_JSON__", _js_safe(json.dumps(meta, separators=(",", ":"))))
        .replace(
            "__PROGRAMS_JSON__", _js_safe(json.dumps(programs, separators=(",", ":")))
        )
    )
    html = html.replace("__CYTOSCAPE_TAG__", cyto, 1)
    out.write_text(html)
    print(
        f"[done] {len(programs)} program(s) -> {out} ({out.stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
