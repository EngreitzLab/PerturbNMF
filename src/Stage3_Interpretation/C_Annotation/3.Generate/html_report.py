"""Render the AGeneTic program JSONs as one self-contained HTML report.

Reads '<info_dir>/P<k>.json' (Gene_info_extended_PerturbNMF_Info: gene_info from MyGene /
NCBI / UniProt, gene_interactions.OmniPath, optionally gene_interactions.Literature from
2.Evidence_curation) and writes one HTML page with a program rail. Per program:

  * an interactive graph (Cytoscape.js, inlined so the file works offline) of the top
    regulators x top program genes; edges = OmniPath interactions (solid, colored by sign,
    arrow when directed) plus paper-qa literature relations (dashed). Nodes can be dragged;
    hover shows a tooltip, click opens the gene / interaction in the inspector
  * marker genes, per-gene information cards, regulators, GO terms, program specificity
    and the curated literature evidence

Style follows gene-program-interpreter's report (gpi/html_report.py, design A); node and
edge colors follow GeneProgramExplorer's viewer (viewer/src/lib/colorScale.js).
Only the standard library is needed.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "report_template.html"
CYTOSCAPE_PATH = HERE / "cytoscape.min.js"
CYTOSCAPE_CDN = '<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>'

# GeneProgramExplorer color scales
BLUE, WHITE, RED = (33, 102, 172), (247, 247, 247), (178, 24, 43)
INDIGO = (67, 56, 202)
EDGE_COLORS = {"activation": "#16a34a", "inhibition": "#dc2626", "mixed": "#64748b",
               "literature": "#d97706"}


# colors
def _mix(c1, c2, t):
    return "rgb(%d, %d, %d)" % tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def diverging_color(value, m):
    """Blue (down) -> white (0) -> red (up) over [-m, m]."""
    if value is None or not m:
        return _mix(WHITE, WHITE, 0)
    t = min(1.0, abs(value) / m)
    return _mix(WHITE, RED if value > 0 else BLUE, t)


def sequential_color(t):
    """White -> indigo for t in [0, 1]."""
    return _mix(WHITE, INDIGO, max(0.0, min(1.0, t)))


def text_on(rgb):
    """Black or white text, whichever reads on the fill."""
    r, g, b = (int(x) for x in re.findall(r"\d+", rgb)[:3])
    return "#1a1a1a" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


# loaders
def load_bundles(info_dir, programs=None):
    """{program number: bundle}; all P<k>.json in info_dir (numeric order) when programs is None."""
    info_dir = Path(info_dir)
    if not info_dir.is_dir():
        raise FileNotFoundError(f"Input not found: {info_dir}")
    if programs is None:
        found = [p for p in info_dir.glob("P*.json") if re.fullmatch(r"P\d+", p.stem)]
        programs = sorted(int(p.stem[1:]) for p in found)
    paths = {pid: info_dir / f"P{pid}.json" for pid in dict.fromkeys(programs)}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No program JSON for {len(missing)} program(s): {missing}")
    if not paths:
        raise FileNotFoundError(f"No P<k>.json in {info_dir}")
    return {pid: json.loads(p.read_text()) for pid, p in paths.items()}


def gene_name(entry):
    return entry if isinstance(entry, str) else entry["gene"]


def gene_entries(bundle):
    """{gene: entry} over program genes, distinctive genes and regulators (first wins)."""
    entries = {}
    for e in bundle.get("program_genes", []) + bundle.get("distinctive_genes", []):
        entries.setdefault(gene_name(e), e if isinstance(e, dict) else {"gene": e})
    for regs in bundle.get("perturbation_regulators", {}).values():
        for r in regs:
            entries.setdefault(r["gene"], r)
    return entries


# genes
def gene_card(entry):
    """Gene information of one entry, from its MyGene / NCBI / UniProt blocks."""
    info = entry.get("gene_info", {})
    mg, nc, up = (info.get(s, {}) for s in ("MyGene", "NCBI", "UniProt"))
    aliases = []
    for block in (mg, nc, up):
        aliases += block.get("alias") or []
    symbol = mg.get("symbol") or nc.get("symbol") or up.get("symbol") or entry["gene"]
    pmids = list(dict.fromkeys((up.get("function_pmids") or []) + (nc.get("generif_pmids") or [])))
    return {
        "gene": entry["gene"],
        "symbol": symbol,
        "name": mg.get("name") or nc.get("name") or up.get("protein_name"),
        "aliases": [a for a in dict.fromkeys(aliases) if a not in (symbol, entry["gene"])],
        "summary": mg.get("summary") or nc.get("summary"),
        "go_bp": mg.get("go_bp") or [],
        "entrez": mg.get("entrezgene") or nc.get("gene_id"),
        "map_location": nc.get("map_location"),
        "uniprot": up.get("accession"),
        "protein_name": up.get("protein_name"),
        "function": up.get("function"),
        "location": up.get("subcellular_location") or [],
        "pmids": pmids[:12],
        "sources": [s for s in ("MyGene", "NCBI", "UniProt") if info.get(s, {}).get("found")],
    }


def select_regulators(bundle, n=5):
    """Top-n regulators over all conditions: best adj_pval, then |log2FC|; per-condition stats kept."""
    best, per_cond = {}, {}
    for cond, regs in bundle.get("perturbation_regulators", {}).items():
        for r in regs:
            g = r["gene"]
            per_cond.setdefault(g, {})[cond] = {"log2fc": r.get("log2fc"), "adj_pval": r.get("adj_pval")}
            key = (r.get("adj_pval", 1.0), -abs(r.get("log2fc") or 0))
            if g not in best or key < best[g][0]:
                best[g] = (key, r, cond)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:n]
    return [{"gene": g, "log2fc": r.get("log2fc"), "adj_pval": r.get("adj_pval"), "condition": cond,
             "per_condition": per_cond[g]} for g, (_, r, cond) in ranked]


# graph
def node_width(label, regulator=False):
    """Node width in px for a monospace label (Cytoscape's width: label is deprecated)."""
    return round(len(label) * (8.2 if regulator else 7.4) + (30 if regulator else 16))


def _sign(signs):
    signs = set(signs or [])
    if signs == {"activation"}:
        return "activation"
    if signs == {"inhibition"}:
        return "inhibition"
    return "mixed"


def omnipath_edge(pair):
    dirs = set(pair.get("directions") or [])
    sign = _sign(pair.get("signs"))
    return {
        "id": f"e_op_{pair['regulator']}_{pair['gene']}",
        "source": f"n_{pair['regulator']}", "target": f"n_{pair['gene']}",
        "kind": "omnipath", "sign": sign, "color": EDGE_COLORS[sign],
        "tarrow": "triangle" if "regulator->gene" in dirs else "none",
        "sarrow": "triangle" if "gene->regulator" in dirs else "none",
        "lstyle": "solid",
        "width": round(1.6 + min(pair.get("n_references", 0), 20) / 4, 2),
        "regulator": pair["regulator"], "gene": pair["gene"],
        "categories": pair.get("interaction_categories", []),
        "directions": pair.get("directions", []),
        "resources": pair.get("resources", []),
        "n_references": pair.get("n_references", 0),
        "pmids": pair.get("pmids", []),
        "curation_effort": pair.get("curation_effort"),
        "interactions": pair.get("interactions", []),
    }


def literature_edge(pair):
    return {
        "id": f"e_lit_{pair['regulator']}_{pair['gene']}",
        "source": f"n_{pair['regulator']}", "target": f"n_{pair['gene']}",
        "kind": "literature", "sign": "literature", "color": EDGE_COLORS["literature"],
        "tarrow": "triangle" if pair.get("directed") else "none", "sarrow": "none",
        "lstyle": "dashed", "width": 2.2,
        "regulator": pair["regulator"], "gene": pair["gene"],
        "category": pair.get("category"), "short_excerpt": pair.get("short_excerpt"),
        "citation": pair.get("citation") or {}, "n_pdfs": pair.get("n_pdfs"),
    }


def build_graph(bundle, top_regulator=5, top_gene=15):
    """Nodes (top regulators + top program genes) and their OmniPath / literature edges."""
    regulators = select_regulators(bundle, top_regulator)
    genes = [gene_name(e) for e in bundle.get("program_genes", [])[:top_gene]]
    reg_names = [r["gene"] for r in regulators]
    m = max([abs(r["log2fc"] or 0) for r in regulators] or [0])

    nodes = []
    for r in regulators:
        fill = diverging_color(r["log2fc"], m)
        nodes.append({"id": f"n_{r['gene']}", "label": r["gene"], "kind": "regulator",
                      "w": node_width(r["gene"], regulator=True),
                      "fill": fill, "fg": text_on(fill), "log2fc": r["log2fc"],
                      "adj_pval": r["adj_pval"], "condition": r["condition"],
                      "per_condition": r["per_condition"], "in_program": r["gene"] in genes})
    for i, g in enumerate(genes):
        if g in reg_names:
            continue   # a regulator that is also a program gene is drawn once, as a regulator
        fill = sequential_color(0.75 * (1 - i / max(len(genes) - 1, 1)) + 0.1)
        nodes.append({"id": f"n_{g}", "label": g, "kind": "gene", "rank": i + 1, "w": node_width(g),
                      "fill": fill, "fg": text_on(fill)})

    node_ids = {n["id"] for n in nodes}
    gi = bundle.get("gene_interactions", {})
    omnipath = gi.get("OmniPath")
    edges = []
    if omnipath:
        for pair in omnipath["pairs"].values():
            if pair["found"] and pair["regulator"] in reg_names and pair["gene"] in genes:
                edge = omnipath_edge(pair)
                if edge["source"] in node_ids and edge["target"] in node_ids:
                    edges.append(edge)
    literature = gi.get("Literature")
    if literature:
        for pair in literature["pairs"].values():
            if (pair.get("source") == "paper-qa" and pair.get("category")
                    and pair["regulator"] in reg_names and pair["gene"] in genes):
                edges.append(literature_edge(pair))
    return {"nodes": nodes, "edges": edges, "has_omnipath": omnipath is not None,
            "has_literature": literature is not None}


# program
def omnipath_summary(bundle):
    op = bundle.get("gene_interactions", {}).get("OmniPath")
    if not op:
        return None
    return {"n_regulators": op.get("n_regulators"),
            "n_found": sum(op.get("n_pairs_found", {}).values()),
            "n_tested": sum(op.get("n_pairs_tested", {}).values()),
            "by_category": {c: [op["n_pairs_found"].get(c, 0), n]
                            for c, n in op.get("n_pairs_tested", {}).items()},
            "datasets": op.get("params", {}).get("datasets", [])}


def literature_rows(bundle):
    lit = bundle.get("gene_interactions", {}).get("Literature")
    if not lit:
        return None
    rows = [{"pair": k, **{f: p.get(f) for f in ("gene", "regulator", "status", "source", "category",
                                                  "directed", "short_excerpt", "citation", "n_pdfs")}}
            for k, p in lit["pairs"].items() if p.get("status") != "omnipath"]
    return {"n_by_status": lit.get("n_by_status", {}), "rows": rows}


def build_lead(bundle, regulators):
    spec = bundle.get("program_specificity", {})
    top = spec.get("top_condition")
    pct = spec.get("per_condition", {}).get(top, {}).get("pct_cells_expressed") if top else None
    n_regs = len({r["gene"] for regs in bundle.get("perturbation_regulators", {}).values() for r in regs})
    parts = [f"<b>{len(bundle.get('program_genes', []))}</b> top-loading and "
             f"<b>{len(bundle.get('distinctive_genes', []))}</b> distinctive genes"]
    parts.append(f"<b>{n_regs}</b> significant regulator{'' if n_regs == 1 else 's'}"
                 + (f", led by <b>{regulators[0]['gene']}</b>" if regulators else ""))
    lead = "; ".join(parts) + "."
    if top:
        lead += f" Usage is highest in <b>{top}</b>" + (f" ({pct}% of cells)." if pct is not None else ".")
    return lead


def build_program(pid, bundle, top_regulator=5, top_gene=15):
    graph = build_graph(bundle, top_regulator, top_gene)
    regulators = select_regulators(bundle, top_regulator)
    entries = gene_entries(bundle)
    go = bundle.get("GO", [])
    return {
        "id": pid,
        "label": bundle.get("program_id", f"P{pid}"),
        "title": re.sub(r"\s*\(GO:\d+\)", "", go[0]) if go else "No enriched GO term",
        "lead_html": build_lead(bundle, regulators),
        "organism": bundle.get("organism"),
        "cell_type": bundle.get("cell_type"),
        "conditions": bundle.get("conditions", []),
        "specificity": bundle.get("program_specificity", {}),
        "go": [re.sub(r"\s*\(GO:\d+\)", "", t) for t in go],
        "program_genes": [gene_name(e) for e in bundle.get("program_genes", [])],
        "distinctive_genes": [gene_name(e) for e in bundle.get("distinctive_genes", [])],
        "regulators_by_condition": {c: [{k: r.get(k) for k in ("gene", "log2fc", "adj_pval")} for r in regs]
                                    for c, regs in bundle.get("perturbation_regulators", {}).items()},
        "top_regulators": [r["gene"] for r in regulators],
        "genes": {g: gene_card(e) for g, e in entries.items()},
        "graph": graph,
        "omnipath": omnipath_summary(bundle),
        "literature": literature_rows(bundle),
    }


# html
def _js_safe(blob):
    """Keep embedded '</script>' or '<!--' from ending the inline script early."""
    return blob.replace("</", "<\\/").replace("<!--", "<\\!--")


def _html_esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(programs, dataset_name="", inline_cytoscape=True, top_regulator=5, top_gene=15):
    template = TEMPLATE_PATH.read_text()
    if inline_cytoscape:
        lib = CYTOSCAPE_PATH.read_text()
        if "</script" in lib.lower():
            raise ValueError(f"{CYTOSCAPE_PATH} contains '</script'; cannot inline it (use --cdn)")
        cyto = f"<script>{lib}</script>"
    else:
        cyto = CYTOSCAPE_CDN
    data = _js_safe(json.dumps({p["id"]: p for p in programs}, ensure_ascii=False))
    plist = _js_safe(json.dumps([[p["id"], p["label"], p["title"]] for p in programs], ensure_ascii=False))
    # small placeholders first, then the data and library blobs (never re-scanned)
    html = (template
            .replace("__DATASET_SUB__", _html_esc(dataset_name or "AGeneTic"))
            .replace("__DATASET_JSON__", json.dumps(dataset_name or ""))
            .replace("__NUM_PROGRAMS__", str(len(programs)))
            .replace("__GENERATED_ON__", datetime.now().strftime("%Y-%m-%d"))
            .replace("__TOP_REGULATOR__", str(top_regulator))
            .replace("__TOP_GENE__", str(top_gene))
            .replace("__PROGRAM_LIST_JSON__", plist))
    html = html.replace("__PROGRAMS_JSON__", data, 1)
    return html.replace("__CYTOSCAPE_TAG__", cyto, 1)


def build_parser():
    p = argparse.ArgumentParser(description="Write an HTML report with per-program regulator-gene interaction graphs.")

    # IO
    p.add_argument("--info_dir", required=True, help="Gene_info_extended_PerturbNMF_Info folder (P<k>.json).")
    p.add_argument("--out", default=None, help="Output HTML. Default: <info_dir>/../Report/AGeneTic_report.html.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids, space separated (e.g. 1 2 3). Default: every P<k>.json in --info_dir.")
    p.add_argument("--dataset_name", default="", help="Dataset label shown in the top bar and hero.")

    # graph
    p.add_argument("--top_regulator", type=int, default=5, help="Regulators per graph (best adj_pval over conditions).")
    p.add_argument("--top_gene", type=int, default=15, help="Top-loading program genes per graph.")
    p.add_argument("--cdn", action="store_true", help="Load Cytoscape.js from unpkg instead of inlining it (smaller file, needs internet).")
    return p


def main():
    args = build_parser().parse_args()
    bundles = load_bundles(args.info_dir, args.programs)
    programs = [build_program(pid, b, args.top_regulator, args.top_gene) for pid, b in bundles.items()]
    for p in programs:
        g = p["graph"]
        n_reg = sum(n["kind"] == "regulator" for n in g["nodes"])
        note = "" if g["has_omnipath"] else "  [warn] no OmniPath block (run OmniPath.py); graph has no edges"
        print(f"  {p['label']}: regulators={n_reg} genes={len(g['nodes']) - n_reg} "
              f"edges={len(g['edges'])}{note}")

    out = Path(args.out or Path(args.info_dir).resolve().parent / "Report" / "AGeneTic_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(programs, args.dataset_name, not args.cdn,
                               args.top_regulator, args.top_gene))
    print(f"[done] {len(programs)} program(s) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
