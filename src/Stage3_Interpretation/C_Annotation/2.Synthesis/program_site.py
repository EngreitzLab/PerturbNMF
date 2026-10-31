"""Write a static website with one page per program (e.g. for GitHub Pages).

Reads the run's program folders (<run_dir>/Programs/P<k>/, through program_bundle.load_run) and writes

    <site_dir>/index.html               every dataset folder in site_dir (from its site.json)
    <site_dir>/<slug>/index.html        program list: name + one-sentence function
    <site_dir>/<slug>/P<k>.html         one page per program
    <site_dir>/<slug>/site.json         dataset name and program list, read by the root index
    <site_dir>/.nojekyll
    <site_dir>/shared/plotly.min.js     only with --perturbation_path_base or --mdata_path

Per program, in page order: the name and one-sentence function; PerturbNMF input (top genes, top unique
(distinctive) genes, top regulators with their sign, top GO terms); the pre-annotation screen when
0.Prepare/build_confounder_screens.py has run (family label with its genes, enriched chromosome at p < 0.05);
driving genes when
the optional driver_genes step has run for the current label (a card per gene whose own papers
explain the label: verdict, why, supporting papers, other literature searched; the other genes searched in a
collapsed group); program interactive plots (Plotly, built with A_Plotting/src/html_Program_QC_plots.py; needs
pandas + plotly): with --perturbation_path_base a regulator volcano per condition and a regulator x condition
log2FC heatmap, with --mdata_path and --groupby a program activity violin per level of that obs column (e.g. day).
A regulator with negative log2FC on knockdown is
an activator (knocking it down lowers the program); positive is a repressor.
Programs without label.json are skipped. Without --perturbation_path_base / --mdata_path only the standard
library is needed.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from html import escape
from urllib.parse import quote
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from html_report import gene_card, gene_entries, gene_name, select_regulators  # noqa: E402

TEMPLATE_PATH = HERE / "program_page_template.html"
PLOTTING_DIR = HERE.parents[1] / "A_Plotting"  # src/ package with html_Program_QC_plots
NCBI_GENE = "https://www.ncbi.nlm.nih.gov/gene/"
PUBMED = "https://pubmed.ncbi.nlm.nih.gov/"
CELL_TYPE_TEXT = {
    "same": "evidence in this cell type",
    "related": "evidence in a related cell type",
    "other": "evidence from other cell types",
    "none": "no cell-type evidence",
}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "dataset"


def page(title, brand_sub, home_href, content, footer, nav=""):
    return (
        TEMPLATE_PATH.read_text()
        .replace("__PAGE_TITLE__", escape(title))
        .replace("__BRAND_SUB__", escape(brand_sub))
        .replace("__HOME_HREF__", home_href)
        .replace("__NAV__", nav)
        .replace("__FOOTER__", footer)
        .replace("__CONTENT__", content)
    )


# program page
def gene_chips(genes, cards, cls="gene", organism="human"):
    """Gene chips linked to NCBI Gene: the gene's page when the bundle knows its Entrez id, else an NCBI
    symbol search in the organism."""
    if not genes:
        return '<p class="empty">None.</p>'
    chips = []
    for g in genes:
        c = cards.get(g, {})
        tip = escape(c.get("name") or g, quote=True)
        href = (
            f"{NCBI_GENE}{c['entrez']}"
            if c.get("entrez")
            else f"{NCBI_GENE}?term={quote(g)}%5Bsym%5D+AND+{quote(organism)}%5Borgn%5D"
        )
        chips.append(
            f'<a class="{cls}" href="{escape(href, quote=True)}" title="{tip}" target="_blank" rel="noopener">{escape(g)}</a>'
        )
    return f'<div class="genes">{"".join(chips)}</div>'


def regulator_table(regulators, cards):
    if not regulators:
        return '<p class="empty">No significant regulator.</p>'
    rows = []
    for r in regulators:
        fc = r.get("log2fc")
        badge = (
            (
                '<span class="badge act">activator &darr;</span>'
                if fc < 0
                else '<span class="badge rep">repressor &uarr;</span>'
            )
            if fc is not None
            else ""
        )
        card = cards.get(r["gene"], {})
        name, symbol = card.get("name") or "", card.get("symbol")
        alt = (
            f' <span class="empty">({escape(symbol)})</span>'
            if symbol and symbol != r["gene"]
            else ""
        )
        p = r.get("adj_pval")
        rows.append(
            f'<tr><td class="mono" title="{escape(name, quote=True)}"><b>{escape(r["gene"])}</b>{alt}</td>'
            f"<td>{'' if fc is None else f'{fc:+.2f}'}</td><td>{badge}</td>"
            f"<td>{'' if p is None else f'{p:.2g}'}</td><td>{escape(str(r.get('condition', '')))}</td></tr>"
        )
    return (
        '<table class="regs"><thead><tr><th>Regulator</th><th>log2FC</th><th>Sign</th><th>adj p</th><th>Condition</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def driver_block(bundle, label):
    """literature_evidence.driver_gene when it was computed for this label, else None."""
    dg = (bundle.get("literature_evidence") or {}).get("driver_gene")
    if not dg:
        return None
    if dg.get("annotation") != label["program_name"]:
        print(
            f'  [warn] driver genes were computed for "{dg.get("annotation")}", not the current label; not shown'
        )
        return None
    return dg


def paper_records(answer, paper_meta, min_score):
    """Supporting papers as [(title, finding, meta)] plus the set of their normalized titles: the judge's
    paper_findings (several findings for the same title joined), else paper-qa's evidence papers with
    relevance >= min_score (0-10). paper_meta {lowercased title: paper} from
    gene_info.Gene_function_categorization gives PMID / year."""
    merged = {}  # one line per paper: several findings for the same title are joined
    for f in (answer.get("judge") or {}).get("paper_findings") or []:
        merged.setdefault(f["title"].strip(), []).append(f["finding"].strip())
    findings = [(t, " ".join(fs)) for t, fs in merged.items()]
    if not findings:
        best = {}
        for e in answer.get("evidence") or []:
            t = (e.get("title") or "").strip()
            if (
                t
                and (e.get("score") or 0) >= min_score
                and (t not in best or e["score"] > best[t]["score"])
            ):
                best[t] = e
        findings = [
            (t, qa_first_sentence(e["summary"]))
            for t, e in sorted(best.items(), key=lambda kv: -kv[1]["score"])
        ]
    records = [
        (title, finding, paper_meta.get(norm_title(title), {}))
        for title, finding in findings
    ]
    return records, {norm_title(t) for t, _ in findings}


def paper_items(answer, paper_meta, min_score):
    """One line per supporting paper (paper_records): title (PubMed link), year / PMID, the one-line finding."""
    records, shown = paper_records(answer, paper_meta, min_score)
    items = [paper_li(title, finding, meta) for title, finding, meta in records]
    html = (
        f'<ul class="papers">{"".join(items)}</ul>'
        if items
        else '<p class="empty">No supporting paper listed.</p>'
    )
    return html, shown


def norm_title(title):
    return (title or "").lower().strip().rstrip(".")


def paper_li(title, text, meta, tag=""):
    """One paper: title (PubMed link when the PMID is known, else DOI / URL), text, year / PMID, optional tag."""
    pmid = meta.get("pmid")
    url = (
        f"{PUBMED}{pmid}/"
        if pmid
        else f"https://doi.org/{meta['doi']}"
        if meta.get("doi")
        else meta.get("url")
    )
    link = (
        f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener">{escape(title)}</a>'
        if url
        else escape(title)
    )
    pmid_link = (
        f'<a href="{PUBMED}{escape(pmid)}/" target="_blank" rel="noopener">PMID {escape(pmid)}</a>'
        if pmid
        else ""
    )
    info = " &middot; ".join(
        x for x in (escape(str(meta.get("year") or "")), pmid_link, tag) if x
    )
    return (
        f'<li><div class="ptitle">{link}</div>{f"<div class=pex>{escape(text)}</div>" if text else ""}'
        f"{f'<div class=pmeta>{info}</div>' if info else ''}</li>"
    )


def other_paper_records(gene, shown, paper_meta, bundle):
    """The other papers searched for a gene: its gene function search papers not cited as support
    [(title, claim, meta)], then program literature search papers that mention the gene
    [(title, finding, paper, same_cell_type)]."""
    gf = [
        (p["title"], p.get("claim"), p) for t, p in paper_meta.items() if t not in shown
    ]
    seen = shown | set(paper_meta)
    lit = []
    for task in (bundle.get("literature_search") or {}).values():
        for p in task.get("papers") or []:
            t = norm_title(p.get("title"))
            if t and t not in seen and gene in str(p.get("genes_mentioned") or ""):
                seen.add(t)
                lit.append(
                    (
                        p["title"],
                        p.get("finding"),
                        p,
                        p.get("context_match") == "same_cell_type",
                    )
                )
    return gf, lit


def other_literature(gene, shown, paper_meta, bundle, wrap=True):
    """Collapsed list of the other papers searched for a gene (other_paper_records); program literature
    search papers are tagged when they study the same cell type."""
    gf_rec, lit_rec = other_paper_records(gene, shown, paper_meta, bundle)
    gf = [paper_li(t, text, meta) for t, text, meta in gf_rec]
    lit = [
        paper_li(
            t,
            text,
            meta,
            '<span class="badge sup">same cell type</span>' if same else "",
        )
        for t, text, meta, same in lit_rec
    ]
    if not gf and not lit:
        return "" if wrap else '<p class="empty">No paper retrieved.</p>'
    parts = (
        [
            f'<div class="psub">Gene function search</div><ul class="papers">{"".join(gf)}</ul>'
        ]
        if gf
        else []
    ) + (
        [
            f'<div class="psub">Program literature search</div><ul class="papers">{"".join(lit)}</ul>'
        ]
        if lit
        else []
    )
    if not wrap:
        return "".join(parts)
    return (
        f'<details class="more"><summary>Other literature searched ({len(gf) + len(lit)})</summary>'
        f"{''.join(parts)}</details>"
    )


def qa_first_sentence(text):
    flat = " ".join((text or "").split())
    m = re.search(r"(.+?[.!?])(\s|$)", flat)
    return m.group(1) if m else flat


def gene_paper_meta(bundle):
    """{gene: {lowercased title: paper}} from gene_info.Gene_function_categorization."""
    out = {}
    for g, e in gene_entries(bundle).items():
        papers = (
            (e.get("gene_info") or {}).get("Gene_function_categorization") or {}
        ).get("papers") or []
        out[g] = {
            (p.get("title") or "").lower().rstrip("."): p
            for p in papers
            if p.get("title")
        }
    return out


def driver_records(dg):
    """(driving genes: the explained genes, supported before mixed; other genes searched, in asked order)."""
    order = {"supported": 0, "mixed": 1}
    genes = sorted(
        dg.get("explained", []),
        key=lambda g: order.get(dg["genes"][g].get("verdict"), 2),
    )
    return genes, [g for g in dg.get("genes", {}) if g not in genes]


def driving_genes(dg, cards, paper_meta, bundle, min_score=5):
    """Collapsible group: one card per gene whose own papers explain the label."""
    genes, others = driver_records(dg)
    gene_cards = []
    for g in genes:
        a = dg["genes"][g]
        verdict = a.get("verdict") or ""
        j = a.get("judge") or {}
        supporting, shown = paper_items(a, paper_meta.get(g, {}), min_score)
        gene_cards.append(
            f'<section class="section"><div class="dhead">{gene_chips([g], cards)}'
            f'<span class="badge {"sup" if verdict == "supported" else "mix"}">{escape(verdict)}</span></div>'
            f'<p class="reason">{escape(j.get("why") or j.get("reason") or a.get("short_excerpt") or "")}</p>'
            f"{supporting}{other_literature(g, shown, paper_meta.get(g, {}), bundle)}"
            f'<div class="hmeta ctype">{escape(CELL_TYPE_TEXT.get(a.get("cell_type_evidence"), ""))}</div></section>'
        )
    body = (
        f'<div class="grid">{"".join(gene_cards)}</div>'
        if gene_cards
        else '<p class="empty">No gene\'s literature explains this label.</p>'
    )
    other_cards = []
    for g in others:  # genes whose papers did not explain the label, collapsed below the driving genes
        a = dg["genes"][g]
        verdict = a.get("verdict") or "not asked"
        j = a.get("judge") or {}
        other_cards.append(
            f'<section class="section"><div class="dhead">{gene_chips([g], cards)}'
            f'<span class="badge ns">{escape(verdict)}</span></div>'
            f'<p class="reason">{escape(j.get("why") or j.get("reason") or a.get("short_excerpt") or "")}</p>'
            f"{other_literature(g, set(), paper_meta.get(g, {}), bundle, wrap=False)}</section>"
        )
    rest = (
        f'<details class="group sub"><summary class="ghead">Other genes searched ({len(others)})</summary>'
        f'<div class="grid">{"".join(other_cards)}</div></details>'
        if others
        else ""
    )
    return f"""
        <details class="group" open>
        <summary class="ghead">Driving genes</summary>
        {body}{rest}
        </details>"""


def load_qc_module():
    """A_Plotting/src/html_Program_QC_plots (pandas, plotly, scanpy): imported only when plots are requested."""
    sys.path.insert(0, str(PLOTTING_DIR))
    from src import html_Program_QC_plots as qc  # noqa: E402

    return qc


def association_cards(qc, pid, conditions, args):
    """Two full-width cards: the regulator volcanoes of every condition on one line (one panel per condition,
    in condition order), then the regulator x condition log2FC heatmap below (Plotly divs)."""
    base = args.perturbation_path_base
    panels = []
    for cond in conditions:
        path = f"{base}_{cond}.txt"
        if not Path(path).exists():
            print(f"  [warn] P{pid}: no association file {path}; volcano skipped")
            continue
        data = qc._build_volcano(
            path,
            pid,
            args.program_col,
            args.target_col,
            args.log2fc_key,
            args.down_thred_log,
            args.up_thred_log,
            args.p_value,
            None,
        )
        n_sig = sum(c != "ns" for c in data["category"])
        fig = qc._make_volcano_fig(data)
        fig.update_layout(height=280, margin=dict(l=45, r=10, t=10, b=40))
        panels.append(
            f'<div class="vpanel"><div class="vhead"><b>{escape(str(cond))}</b> &middot; {n_sig} of '
            f"{len(data['regulators'])} significant</div>"
            f"{qc._fig_to_div(fig, f'volcano-p{pid}-{qc._safe_sample_key(str(cond))}')}</div>"
        )
    cards = []
    if panels:
        cards.append(
            f'<section class="section full"><h2>Regulator volcano</h2>'
            f'<div class="hmeta">Program log2FC on knockdown per perturbed gene; colored = adj p &le; {args.p_value} '
            f"and |log2FC| past the dashed lines</div>"
            f'<div class="vrow" style="--n:{len(panels)}">{"".join(panels)}</div></section>'
        )
    present = [c for c in conditions if Path(f"{base}_{c}.txt").exists()]
    if present:
        data = qc._build_heatmap(
            base,
            None,
            pid,
            present,
            args.program_col,
            args.target_col,
            args.log2fc_key,
            args.p_value,
            None,
        )
        cards.append(
            f'<section class="section full"><h2>Regulator heatmap</h2>'
            f'<div class="hmeta">log2FC of regulators significant (adj p &lt; {args.p_value}) in any condition; * = significant</div>'
            f"{qc._fig_to_div(qc._make_heatmap_fig(data), f'heatmap-p{pid}')}</section>"
        )
    return cards


def load_program_scores(mdata_path, prog_key):
    """The program modality only (cells x programs + obs) of an h5mu: the RNA modality is never read."""
    import mudata

    return {prog_key: mudata.read_h5ad(mdata_path, prog_key)}


def activity_card(qc, pid, scores, args):
    """Card: program activity (usage) per level of --groupby, as a violin; at most --max_points_per_group
    cells per level are embedded (summary stats use every cell). "" when it cannot be drawn."""
    if args.groupby not in scores[args.prog_key].obs.columns:
        print(f"  [warn] obs has no column {args.groupby!r}; activity violin skipped")
        return ""
    if str(pid) not in map(str, scores[args.prog_key].var_names):
        print(
            f"  [warn] program {pid} is not a var name of mod {args.prog_key!r}; activity violin skipped"
        )
        return ""
    data = qc._build_violin(
        scores, pid, args.groupby, data_key=args.prog_key, prog_key=args.prog_key
    )
    rng = __import__("random").Random(0)
    for g, vals in data["per_group_expression"].items():
        if len(vals) > args.max_points_per_group:
            data["per_group_expression"][g] = rng.sample(
                vals, args.max_points_per_group
            )
    fig = qc._make_violin_fig(data)
    fig.update_layout(xaxis_title=args.groupby, yaxis_title="Program activity (usage)")
    return (
        f'<section class="section full"><h2>Program activity by {escape(args.groupby)}</h2>'
        f'<div class="hmeta">Per-cell usage of P{pid}; box = quartiles, line = mean</div>'
        f"{qc._fig_to_div(fig, f'activity-p{pid}')}</section>"
    )


def plots_section(cards):
    """One collapsible group holding every interactive plot card, stacked full width."""
    if not cards:
        return ""
    return f"""
        <details class="group" open>
        <summary class="ghead">Program interactive plots</summary>
        <div class="stack">{"".join(cards)}</div>
        </details>"""


FAMILY_NAMES = {  # build_confounder_screens.py family / marker-set keys
    "cytoplasmic_ribosome": "Cytoplasmic ribosome",
    "mitochondrial_ribosome": "Mitochondrial ribosome",
    "mitochondrially_encoded": "Mitochondrially encoded (MT-)",
    "histone": "Histone",
    "small_nucleolar_or_nuclear_rna": "snoRNA / snRNA",
    "microrna": "microRNA",
    "long_noncoding_or_antisense": "lncRNA / antisense",
    "cell_cycle_s_phase": "Cell cycle, S phase",
    "cell_cycle_g2m_phase": "Cell cycle, G2/M phase",
    "heat_shock": "Heat shock",
    "integrated_stress_response": "Integrated stress response",
    "unfolded_protein_response": "Unfolded protein response",
    "interferon_response": "Interferon response",
}


def family_name(key):
    return FAMILY_NAMES.get(key, key.replace("_", " "))


def screen_section(bundle, cards):
    """Collapsible group from 0.Prepare/build_confounder_screens.py: the family label (genes behind each
    called family) and the most enriched chromosome when its binomial p < 0.05; "" without screens."""
    cs = bundle.get("confounder_screens")
    if not cs:
        return ""
    org = bundle.get("organism") or "human"
    lab, sc, prm = cs["label"], cs["screens"], cs.get("params", {})
    n_top = sc.get("n_top_genes_screened")
    rows = []
    for d in lab.get("family_details", []):
        fam = d["family"]
        hit = sc["symbol_families"].get(fam) or sc["marker_sets"].get(fam) or {}
        primary = (
            '<span class="badge sup">primary</span>'
            if fam == lab.get("primary_family")
            else ""
        )
        rows.append(
            f'<div class="frow"><div class="fhead"><b>{escape(family_name(fam))}</b>{primary}</div>'
            f'<div class="hmeta">{hit.get("n_overlap")} of {n_top} top genes (expected {hit.get("expected")}), '
            f"p = {escape(str(hit.get('p_value')))}</div>{gene_chips(hit.get('genes', []), cards, organism=org)}</div>"
        )
    fam_body = "".join(rows) or (
        f'<p class="empty">None: no family or marker set has &ge; {prm.get("min_genes", 3)} of the top {n_top} '
        f"genes at p &le; {prm.get('max_p', 1e-3):g}.</p>"
    )
    pos = sc.get("positional", {})
    p = float(pos["binomial_p"]) if pos.get("assessable") else None
    if p is not None and p < 0.05:
        flag = (
            '<span class="badge mix">flagged</span>'
            if lab.get("positional_flag")
            else ""
        )
        chrom_body = (
            f'<div class="frow"><div class="fhead"><b>{escape(pos["top_chromosome"])}</b>{flag}</div>'
            f'<div class="hmeta">{pos["genes_on_top_chromosome"]} of {pos["n_genes_located"]} located top genes '
            f"(expected {pos['expected_on_top_chromosome']}, fold {pos['fold_enrichment']}), "
            f"binomial p = {escape(pos['binomial_p'])}</div>"
            f"{gene_chips(pos.get('top_chromosome_genes', []), cards, organism=org)}</div>"
        )
    else:
        detail = (
            (
                f" (most genes on {escape(pos['top_chromosome'])}: {pos['genes_on_top_chromosome']} vs "
                f"{pos['expected_on_top_chromosome']} expected, p = {escape(pos['binomial_p'])})"
            )
            if p is not None
            else ""
        )
        chrom_body = (
            f'<p class="empty">None: no chromosome enriched at p &lt; 0.05{detail}.</p>'
        )
    all_genes = sc.get("top_genes") or []
    more = (
        (
            f'<details class="more"><summary>All {len(all_genes)} screened genes</summary>'
            f"{gene_chips(all_genes, cards, organism=org)}</details>"
        )
        if all_genes
        else ""
    )
    return f"""
        <details class="group" open>
        <summary class="ghead">Pre-annotation screen</summary>
        <div class="grid">
            <section class="section"><h2>Family label</h2><div class="hmeta">Gene families and marker sets enriched among the top {n_top} genes, against this run's gene universe</div>{fam_body}</section>
            <section class="section"><h2>Chromosome enrichment</h2><div class="hmeta">Chromosome holding more top genes than expected (binomial test)</div>{chrom_body}</section>
        </div>{more}
        </details>"""


def program_content(
    pid,
    bundle,
    label,
    dataset_name,
    top_gene,
    top_regulator,
    membership_top=None,
    extra="",
):
    top_n = f"top {membership_top} " if membership_top else ""
    cards = {g: gene_card(e) for g, e in gene_entries(bundle).items()}
    dg = driver_block(bundle, label)
    genes = [gene_name(e) for e in bundle.get("program_genes", [])][:top_gene]
    uniq = [gene_name(e) for e in bundle.get("distinctive_genes", [])]
    regs = select_regulators(bundle, top_regulator)
    go = [re.sub(r"\s*\(GO:\d+\)", "", t) for t in bundle.get("GO", [])]
    go_html = (
        f'<ol class="go">{"".join(f"<li>{escape(t)}</li>" for t in go)}</ol>'
        if go
        else '<p class="empty">No enriched GO term.</p>'
    )
    return f"""
        <section class="hero">
            <div class="kicker"><span class="pid">P{pid}</span></div>
            <h1 class="title">{escape(label["program_name"])}</h1>
            <p class="lead">{escape(label["function_sentence"])}</p>
        </section>
        <details class="group" open>
        <summary class="ghead">PerturbNMF input for annotation</summary>
        <div class="grid">
            <section class="section"><h2>Top genes</h2><div class="hmeta">Top {len(genes)} by program loading</div>{gene_chips(genes, cards)}</section>
            <section class="section"><h2>Top unique genes</h2><div class="hmeta">High in this program, low in others{f" among {top_n}loaded genes" if membership_top else ""}</div>{gene_chips(uniq, cards, "gene uniq")}</section>
            <section class="section"><h2>Top regulators</h2><div class="hmeta">Program usage log2FC on CRISPRi knockdown</div>{regulator_table(regs, cards)}</section>
            <section class="section"><h2>Top GO terms</h2><div class="hmeta">Gene ontology enrichment of {top_n}program genes</div>{go_html}</section>
        </div>
        </details>{screen_section(bundle, cards)}{driving_genes(dg, cards, gene_paper_meta(bundle), bundle) if dg else ""}
        {'<script src="../shared/plotly.min.js"></script>' if extra else ""}{extra}"""


def nav_links(pids, i):
    prev_ = f"P{pids[i - 1]}.html" if i > 0 else "#"
    next_ = f"P{pids[i + 1]}.html" if i < len(pids) - 1 else "#"
    return (
        f'<a class="iconbtn" href="index.html">All programs</a>'
        f'<a class="iconbtn{"" if i > 0 else " off"}" href="{prev_}" title="Previous program">&larr;</a>'
        f'<a class="iconbtn{"" if i < len(pids) - 1 else " off"}" href="{next_}" title="Next program">&rarr;</a>'
    )


# index pages
def dataset_index(dataset_name, labels):
    cards = "".join(
        f'<a class="pcard" href="P{pid}.html"><div class="kicker"><span class="pid">P{pid}</span></div>'
        f'<div class="pname">{escape(l["program_name"])}</div><div class="pdesc">{escape(l["function_sentence"])}</div></a>'
        for pid, l in labels.items()
    )
    return (
        f'<section class="hero"><div class="kicker"><span class="crumb">{len(labels)} programs</span></div>'
        f'<h1 class="title">{escape(dataset_name)}</h1></section><div class="plist">{cards}</div>'
    )


def root_index(site_dir):
    datasets = []
    for meta in sorted(Path(site_dir).glob("*/site.json")):
        d = json.loads(meta.read_text())
        datasets.append((meta.parent.name, d))
    cards = "".join(
        f'<a class="pcard" href="{slug}/index.html"><div class="pname">{escape(d["dataset_name"])}</div>'
        f'<div class="pdesc">{len(d["programs"])} programs &middot; updated {escape(d["updated"])}</div></a>'
        for slug, d in datasets
    )
    return f'<section class="hero"><h1 class="title">Program reports</h1></section><div class="plist">{cards}</div>'


def build_parser():
    p = argparse.ArgumentParser(
        description="Write a static website with one page per program."
    )

    # IO
    p.add_argument(
        "--run_dir",
        required=True,
        help="Annotation run folder (Programs/P<k>/ with label.json, ...).",
    )
    p.add_argument(
        "--site_dir",
        required=True,
        help="Website root (e.g. a GitHub Pages repo clone).",
    )

    # context info
    p.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset label; its slug names the dataset folder.",
    )
    p.add_argument(
        "--programs",
        type=int,
        nargs="+",
        default=None,
        help="Program ids, space separated (e.g. 1 2 3). Default: every program folder with a label.",
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
    # regulator-program association plots (optional)
    p.add_argument(
        "--perturbation_path_base",
        default=None,
        help="Perturbation association results without the _<condition>.txt suffix (e.g. Evaluation/<K>_<thr>/<K>_perturbation_association_results); adds the volcano + heatmap group. Default: no plots.",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions, space separated (files <base>_<condition>.txt). Default: the bundle's conditions.",
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
        help="adj p cutoff for the volcano colors and the heatmap regulators.",
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

    # program activity violin (optional)
    p.add_argument(
        "--mdata_path",
        default=None,
        help="cNMF h5mu; with --groupby adds a program activity violin per obs level. Only the program modality is read. Default: no violin.",
    )
    p.add_argument(
        "--groupby", default=None, help="obs column to group cells by (e.g. day)."
    )
    p.add_argument(
        "--prog_key",
        default="cNMF",
        help="Program modality of the h5mu (var names match the P<k> number).",
    )
    p.add_argument(
        "--max_points_per_group",
        type=int,
        default=2000,
        help="Cells per group embedded in the violin (random subsample; keeps pages small).",
    )

    p.add_argument(
        "--membership_top",
        type=int,
        default=None,
        help="Top-loaded genes per program used for GO enrichment and unique genes (0.Prepare membership_top); shown in the card captions.",
    )
    return p


def main():
    args = build_parser().parse_args()
    from program_bundle import load_run

    bundles, labels, _ = load_run(args.run_dir, args.programs, top_gene=args.top_gene)
    skipped = [f"P{p}" for p in bundles if p not in labels]
    if skipped:
        print(f"  [warn] no label.json for {skipped}; skipped")
    if not labels:
        raise SystemExit("No program has a label; nothing to write.")

    site_dir = Path(args.site_dir)
    out_dir = site_dir / slugify(args.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    pids = list(labels)
    if args.mdata_path and not args.groupby:
        raise SystemExit("--mdata_path needs --groupby")
    qc, scores = None, None
    if args.perturbation_path_base or args.mdata_path:
        qc = load_qc_module()
        qc._write_plotly_js(
            site_dir
        )  # <site_dir>/shared/plotly.min.js, matching the installed plotly
    if args.mdata_path:
        scores = load_program_scores(args.mdata_path, args.prog_key)
    for i, pid in enumerate(pids):
        plot_cards = []
        if args.perturbation_path_base:
            plot_cards += association_cards(
                qc, pid, args.conditions or bundles[pid].get("conditions") or [], args
            )
        if scores is not None:
            plot_cards += [c for c in [activity_card(qc, pid, scores, args)] if c]
        extra = plots_section(plot_cards)
        dg = driver_block(bundles[pid], labels[pid])
        judged = (
            (
                f" &middot; driving genes: paper-qa evidence, verdicts by "
                f"{escape((dg.get('params') or {}).get('judge_model') or '')}"
            )
            if dg
            else ""
        )
        footer = (
            f"Generated {today} &middot; name and summary by {escape(labels[pid].get('model', ''))}{judged} "
            f"&middot; source Programs/P{pid}/"
        )
        html = page(
            f"P{pid} {labels[pid]['program_name']} — {args.dataset_name}",
            args.dataset_name,
            "../index.html",
            program_content(
                pid,
                bundles[pid],
                labels[pid],
                args.dataset_name,
                args.top_gene,
                args.top_regulator,
                args.membership_top,
                extra,
            ),
            footer,
            nav_links(pids, i),
        )
        (out_dir / f"P{pid}.html").write_text(html)
        print(f"  P{pid}: {labels[pid]['program_name']}")

    (out_dir / "index.html").write_text(
        page(
            args.dataset_name,
            args.dataset_name,
            "../index.html",
            dataset_index(args.dataset_name, labels),
            f"Generated {today}",
        )
    )
    (out_dir / "site.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "programs": [f"P{p}" for p in pids],
                "updated": today,
            },
            indent=2,
        )
    )
    (site_dir / "index.html").write_text(
        page(
            "Program reports",
            "",
            "index.html",
            root_index(site_dir),
            f"Generated {today}",
        )
    )
    (site_dir / ".nojekyll").touch()
    print(f"[done] {len(pids)} page(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
