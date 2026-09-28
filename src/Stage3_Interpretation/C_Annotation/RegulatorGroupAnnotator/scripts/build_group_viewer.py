"""Build a self-contained HTML viewer for a set of regulator-group annotations.

Same layout as the ProgramAnnotatorV3 viewer (build_annotation_viewer.py) and the same stylesheet
(annotator_core/viewer_common.py): a sticky rail listing every annotated group on the left, grouped
by label family, and one group at a time in the main pane, with full-text search, #group-N deep
links, arrow-key navigation and dark mode. No CDN and no external files.

Per group, in reading order: label, family and distinguisher, and the two-sentence brief summary;
the effect signature as the main figure — every member's log2FC on the programs the group moves,
each row annotated with its clustering role (core / peripheral / rescued / recruited), the
annotator's role (core_explained / consistent / unexplained), stability, reliability and promoter
caveat, and the members the promoter screen excluded greyed at the bottom — with the member x
member correlation of effect profiles (raw or noise-corrected) in the same row order; the shared
function; the unexplained members' hypotheses; then why the group forms here, the curated
complexes, STRING network and enrichment, citations (plus the citation pass per member when
given), confounder assessment, alternative annotations, open questions and the gate's QC.

Inputs are the config build_group_prompts.py used, the dispatch directory and, optionally, the
citation-pass dispatch prefix and the ProgramAnnotatorV3 viewer to link program ids to.

Usage:
    python build_group_viewer.py --config my_group_config.json --dispatch dispatch_groups --arm rg \
        [--citations dispatch_citations/rgcite] [--program-viewer annotation_viewer.html] \
        --output group_viewer.html
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from conditions import normalise_conditions  # noqa: E402
from validate_group_answers import symbol, validate  # noqa: E402
from viewer_common import VIEWER_CSS, bare_pmid, fetch_titles, load_answer, load_support, to_js  # noqa: E402

SIGNIFICANCE = 0.05
ENRICHMENT_SHOWN = 10
# A named confounder as the primary explanation files the group under it in the rail; "other" is
# left out because annotators use it for the shared biology itself ("they are one complex").
CONFOUNDER_GROUPS = {
    "generic_fitness_or_stress": "Fitness / stress",
    "shared_state_shift": "Shared state shift",
    "differentiation_delay": "Shared state shift",  # the same confounder in answers to older prompts
    "promoter_neighbour": "Promoter neighbour",
    "weak_effect_noise": "Weak-effect noise",
}


def effect_rows(genes: list, features: list, effects: pd.DataFrame, adjusted: pd.DataFrame) -> dict:
    """gene -> [[log2FC, adjusted p] | None per signature feature]; None when not in the matrix."""
    rows = {}
    for gene in genes:
        cells = []
        for feature in features:
            if gene in effects.index and feature in effects.columns:
                fc = effects.at[gene, feature]
                q = adjusted.at[gene, feature] if (gene in adjusted.index and feature in adjusted.columns) else float("nan")
                cells.append(None if pd.isna(fc) else [round(float(fc), 3), None if pd.isna(q) else float(q)])
            else:
                cells.append(None)
        rows[gene] = cells
    return rows


def member_correlation(genes: list, effects: pd.DataFrame, reliability: dict) -> dict:
    """Fallback for evidence built before `member_correlation` existed: the same raw and
    noise-corrected (r / sqrt(rel_i rel_j)) correlations build_group_evidence.py writes."""
    genes = [g for g in genes if g in effects.index]
    raw = np.corrcoef(effects.loc[genes].to_numpy()) if len(genes) > 1 else np.ones((len(genes), len(genes)))
    rel = np.maximum(np.array([reliability.get(g, 1.0) for g in genes], dtype=float), 0.2)
    corrected = np.clip(raw / np.sqrt(np.outer(rel, rel)), -1, 1)
    return {"genes": genes, "raw": np.round(raw, 3).tolist(), "corrected": np.round(corrected, 3).tolist()}


def cluster_order(genes: list, correlation: dict) -> list:
    """Members in average-linkage order of 1 - raw r, so similar profiles sit together."""
    index = {g: i for i, g in enumerate(correlation["genes"])}
    known = [g for g in genes if g in index]
    if len(known) < 3:
        return known + [g for g in genes if g not in index]
    raw = np.array(correlation["raw"])[np.ix_([index[g] for g in known], [index[g] for g in known])]
    distance = np.clip(1 - raw, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform((distance + distance.T) / 2, checks=False), method="average")
    return [known[i] for i in leaves_list(tree)] + [g for g in genes if g not in index]


def member_rows(evidence: dict, grouped: dict, correlation: dict) -> list:
    """One row per member (clustered order), then the excluded members: everything the effect
    and correlation heatmaps annotate a row with."""
    rows = {m["gene"]: {**m, "excluded": False} for m in evidence.get("members", [])}
    for entry in evidence.get("excluded", []):
        stats = grouped.get(entry["gene"], {})
        rows[entry["gene"]] = {**stats, "gene": entry["gene"], "excluded": True, "reasons": entry.get("reasons", []),
                               "promoter": {"decision": "exclude", "reasons": entry.get("reasons", [])}}
    order = cluster_order([m["gene"] for m in evidence.get("members", [])], correlation)
    order += [e["gene"] for e in evidence.get("excluded", [])]
    return [{**rows[g], "summary": str(rows[g].get("summary", ""))[:300]} for g in order]


def group_fields(gid: int, evidence: dict, sources: dict) -> dict:
    directory = sources["dispatch"] / f"{sources['arm']}_p{gid}"
    answer = load_answer(directory / "answer.json")
    members = evidence.get("members", [])
    excluded = evidence.get("excluded", [])
    signature = evidence.get("signature", [])
    features = [s["feature"] for s in signature]
    genes = [m["gene"] for m in members] + [e["gene"] for e in excluded]
    grouped = sources["grouped"].get(gid, {})
    correlation = evidence.get("member_correlation") or member_correlation(
        genes, sources["effects"], {**sources["reliability"], **{m["gene"]: m["reliability"] for m in members
                                                                  if m.get("reliability") is not None}})
    rows = member_rows(evidence, grouped, correlation)
    try:
        problems, warnings = validate(gid, directory)
    except OSError as exc:  # no prompt.md next to the answer: the gate cannot run
        problems, warnings = [f"G{gid}: gate not run ({exc})"], []
    primary = [c.get("confounder") for c in answer.get("confounder_assessment", [])
               if c.get("status") == "primary_explanation"]
    pmids = {bare_pmid(c["pmid"]) for c in answer.get("citations", []) if c.get("pmid")}
    for block in [answer.get("shared_function") or {}, answer.get("why_here") or {}, *answer.get("regulators", [])]:
        pmids |= {bare_pmid(p) for p in block.get("pmids") or []}
    return {
        "id": gid,
        "label": answer.get("label", ""),
        "family": answer.get("label_family", ""),
        "distinguisher": answer.get("label_distinguisher", ""),
        "summary": answer.get("brief_summary", ""),
        "coherence": answer.get("coherence", ""),
        "confounders": answer.get("confounder_assessment", []),
        "primary_confounders": primary,
        "shared": answer.get("shared_function") or {},
        "why": answer.get("why_here") or {},
        "roles": [{**r, "symbol": symbol(r.get("symbol"))} for r in answer.get("regulators", [])],
        "label_evidence": answer.get("label_evidence") or {},
        "readings": answer.get("competing_readings", []),
        "citations": [{**c, "pmid": bare_pmid(c["pmid"])} for c in answer.get("citations", []) if c.get("pmid")],
        "cited_pmids": sorted(p for p in pmids if p),
        "open_questions": answer.get("open_questions", []),
        "members": members,
        "excluded": excluded,
        "rows": rows,
        "correlation": correlation,
        "stats": {k: evidence.get(k) for k in ("stability", "mean_raw_r", "mean_corrected_r", "strength_tiers")},
        "signature": signature,
        "effects": effect_rows([r["gene"] for r in rows], features, sources["effects"], sources["adjusted"]),
        "complexes": evidence.get("complexes", []),
        "edges": evidence.get("string_edges", []),
        "ppi": evidence.get("ppi_enrichment"),
        "enrichment": evidence.get("enrichment", [])[:ENRICHMENT_SHOWN],
        "pool": {p["pmid"]: {"title": p.get("title", ""), "year": p.get("year", ""), "sentence": p.get("sentence", ""),
                             "genes": p.get("genes", [])} for p in evidence.get("reference_pool", [])},
        "support": load_support(sources.get("citations"), gid),
        "qc": {
            "rejected": len(list(directory.glob("answer.rejected.*.json"))),
            "invalid": len(list(directory.glob("answer.invalid.*.json"))),
            "problems": problems,
            "warnings": warnings,
        },
    }


def load_from_config(args) -> tuple:
    """Everything dataset-specific comes from the prompt-builder config (paths relative to it)."""
    config = json.loads(args.config.read_text())
    groups_dir = Path(config["groups_dir"])
    if not groups_dir.is_absolute():
        groups_dir = args.config.parent / groups_dir
    conditions = normalise_conditions(config.get("conditions")) or [
        {"label": "all", "description": config["settings"].get("cell_system", "")}]
    payload = json.loads((groups_dir / "group_evidence.json").read_text())
    groups_file = groups_dir / "regulator_groups.json"
    grouping = json.loads(groups_file.read_text()) if groups_file.exists() else {}
    summary_file = groups_dir / "regulator_summary.tsv"
    reliability = (pd.read_csv(summary_file, sep="\t", index_col=0)["reliability"].to_dict()
                   if summary_file.exists() else {})
    sources = {
        # group id -> gene -> the clustering statistics, for members the promoter screen excluded
        "grouped": {g["group_id"]: {m["gene"]: m for m in g["members"]} for g in grouping.get("groups", [])},
        "reliability": reliability,
        "dispatch": args.dispatch, "arm": args.arm, "citations": args.citations,
        "effects": pd.read_csv(groups_dir / "effect_matrix.tsv", sep="\t", index_col=0),
        "adjusted": pd.read_csv(groups_dir / "significance.tsv", sep="\t", index_col=0),
    }
    settings = config["settings"]
    meta = {
        "title": settings.get("dataset_name", "Regulator groups"),
        "subtitle": "regulator-group annotations",
        "alpha": SIGNIFICANCE,
        "conditions": conditions,
        "significance": settings.get("significance_label", f"adjusted p < {SIGNIFICANCE}"),
        "programs_labelled": bool(payload.get("programs_labelled")),
        "calibration_used": bool(grouping.get("calibration_used_for_parameters")),
        "n_regulators": grouping.get("n_regulators"),
        "n_eligible": grouping.get("n_eligible"),
        "n_groups_defined": len(grouping.get("groups", [])) or len(payload["groups"]),
    }
    return payload["groups"], sources, meta, groups_dir


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
""" + VIEWER_CSS + """
.chip.unexplained { border: 1.5px dashed var(--warning); font-weight: 650; }
.chip.excluded { text-decoration: line-through; color: var(--muted); }
.heat th.ann, .heat td.ann { font-size: 11.5px; padding: 0 8px 0 0; white-space: nowrap; text-align: left; vertical-align: middle; }
.heat td.ann.num { font-variant-numeric: tabular-nums; color: var(--text-soft); }
.heat td.g, .heat td.ann { border-bottom: 1px solid var(--border); }
.heat .chip.unexplained { margin: 0; padding: 0 5px; }
.corr td.c { width: 44px; }
.corr th.rot { height: 70px; vertical-align: bottom; }
.corr th.rot span { writing-mode: vertical-rl; transform: rotate(180deg); font-family: var(--mono); font-size: 11px; }
.hyp { border-left: 3px solid var(--warning); }
.tip { white-space: pre-line; }
.heat tr.excl td.g { color: var(--muted); text-decoration: line-through; }
.heat tr.excl td.c { opacity: .45; }
.heat th a { text-decoration: none; }
</style>
</head>
<body>
<div class="top">
  <div class="brand" id="brand"></div>
  <input id="search" type="search" placeholder="Search labels, families, summaries, members, programs…" oninput="filterRail()">
  <div class="spacer"></div>
  <div class="meta" id="meta"></div>
  <button onclick="prev()" title="Previous (←)">←</button>
  <button onclick="next()" title="Next (→)">→</button>
  <button onclick="toggleTheme()" id="themeBtn">Dark</button>
</div>
<div class="shell">
  <nav class="rail" id="rail"></nav>
  <main class="canvas"><div class="wrap" id="main"></div></main>
</div>
<div class="tip" id="tip"></div>
<script>
const GROUPS = __GROUPS__;
const TITLES = __TITLES__;
const META = __META__;
const MULTI = META.conditions.length > 1;
const DESCRIPTIONS = Object.fromEntries(META.conditions.map(c => [c.label, c.description]));
document.getElementById("brand").innerHTML = `${META.title}<small>${META.subtitle}</small>`;
const IDS = Object.keys(GROUPS).map(Number).sort((a,b)=>a-b);
let currentId = null;

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const SEARCH_INDEX = {};
for (const id of IDS) {
  const g = GROUPS[id];
  SEARCH_INDEX[id] = [g.label, g.family, g.distinguisher, g.summary, g.rail,
    g.members.map(m => m.gene).join(" "), g.excluded.map(e => e.gene).join(" "),
    g.signature.map(s => s.program_label).join(" "), (g.why.programs || []).map(p => p.reading).join(" "),
    "G"+id, "group "+id].join(" ").toLowerCase();
}

function buildRail() {
  let out = "";
  for (const r of META.rails) {
    const ids = IDS.filter(id => GROUPS[id].rail === r);
    if (!ids.length) continue;
    out += `<h4>${esc(r)} (${ids.length})</h4>`;
    out += ids.map(id => { const g = GROUPS[id];
      return `<a data-id="${id}" href="#group-${id}" onclick="render(${id});return false;">` +
             `<span class="num">G${id}</span><span>${esc(g.label)}</span>${g.tag ? `<span class="tag">${esc(g.tag)}</span>` : ""}</a>`; }).join("");
  }
  document.getElementById("rail").innerHTML = out;
  filterRail();
}
function filterRail() {
  const q = document.getElementById("search").value.toLowerCase().trim();
  let shown = 0;
  document.querySelectorAll(".rail a").forEach(a => {
    const ok = !q || SEARCH_INDEX[a.dataset.id].includes(q); a.style.display = ok ? "" : "none"; if (ok) shown++; });
  document.querySelectorAll(".rail h4").forEach(h => {
    let el = h.nextElementSibling, any = false;
    while (el && el.tagName === "A") { if (el.style.display !== "none") any = true; el = el.nextElementSibling; }
    h.style.display = any ? "" : "none"; });
  document.getElementById("meta").textContent = q ? `${shown} of ${IDS.length} groups`
    : `${IDS.length} of ${META.n_groups_defined} groups annotated · ${META.built}`;
}

// ---- helpers ------------------------------------------------------------------------------
const pretty = s => String(s || "").replace(/_/g, " ");
const pmidLink = id => `<a href="https://pubmed.ncbi.nlm.nih.gov/${esc(id)}/" target="_blank" rel="noopener">PMID ${esc(id)}</a>`;
const chips = arr => (arr || []).map(x => `<span class="chip">${esc(x)}</span>`).join("");
function programRef(pid) {
  const label = META.program_labels[pid] || "";
  const text = `P${pid}`;
  return META.program_viewer
    ? `<a href="${esc(META.program_viewer)}#program-${pid}" target="_blank" rel="noopener" title="${esc(label)}">${text}</a>`
    : `<span title="${esc(label)}">${text}</span>`;
}
function mix(a, b, t) {
  const pa = a.match(/\\w\\w/g).map(h=>parseInt(h,16)), pb = b.match(/\\w\\w/g).map(h=>parseInt(h,16));
  return "rgb(" + pa.map((x,i)=>Math.round(x+(pb[i]-x)*t)).join(",") + ")";
}
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function divColor(v, scale) {
  const t = Math.min(1, Math.abs(v) / scale);
  return mix(cssVar("--div-mid"), cssVar(v < 0 ? "--div-neg" : "--div-pos"), 0.15 + 0.85 * t);
}
function inkFor(v, scale) { return Math.abs(v) / scale > 0.55 ? "#ffffff" : "var(--text)"; }
function memberInfo(g) { return Object.fromEntries(g.members.map(m => [m.gene, m])); }
function roleOf(g) { return Object.fromEntries(g.roles.map(r => [r.symbol, r])); }

// ---- members: row annotations shared by both heatmaps -------------------------------------
const ANNOTATED = {
  core_explained: ["core", "var(--accent-text)", "known function is the shared function"],
  consistent: ["consistent", "var(--bar)", "compatible with the shared function, not established"],
  unexplained: ["unexplained", "var(--serious)", "no evidence links it to the others — see the hypotheses"],
};
function clusterRole(m) {
  if (!m) return "";
  if (m.role === "rescued") return `rescued via ${m.rescued_via || "a complex"}`;
  return m.role || "";
}
function promoterFlag(m) {
  if (!m || !m.promoter || m.promoter.decision !== "flag") return "";
  const why = (m.promoter.reasons || []).join("; ");
  return `<span class="flag" style="color:var(--serious)" data-tip="${esc("Promoter caveat: " + why)}">⚠</span>`;
}
function annotatedCell(m, r) {
  if (m.excluded) return `<span class="small muted">excluded</span>`;
  if (!r) return `<span class="small" style="color:var(--critical)" data-tip="The answer gives this member no role">no role</span>`;
  const [text, colour, meaning] = ANNOTATED[r.role] || [r.role, "var(--muted)", ""];
  return `<span class="small${r.role === "unexplained" ? " chip unexplained" : ""}" style="color:${colour};font-weight:650" data-tip="${esc(`${pretty(r.role)}: ${meaning}${r.confidence ? " · " + r.confidence + " confidence" : ""}`)}">${text}</span>`;
}
function memberTip(m, r) {
  const f = x => x == null ? "—" : Number(x).toFixed(2);
  return [m.gene + (m.excluded ? " — EXCLUDED by the promoter screen: " + (m.reasons || []).join("; ") : ""),
    `clustering: ${clusterRole(m) || "—"}${m.recruited ? " (recruited: no significant program effect of its own)" : ""}`,
    `stability ${f(m.stability)} · r to group mean ${f(m.r_to_centroid)} · reliability ${f(m.reliability)}`,
    m.strength_tier ? `effect strength: ${m.strength_tier} (${m.n_significant} significant program effects)` : "",
    r ? `annotated: ${pretty(r.role)}${r.confidence ? ", " + r.confidence + " confidence" : ""}` : "",
    r && r.hypothesis ? `note: ${r.hypothesis}` : "",
    m.promoter && m.promoter.decision === "flag" ? `promoter caveat: ${(m.promoter.reasons || []).join("; ")}` : "",
    m.connected_to ? `connected to: ${m.connected_to.join(", ") || "none"}` : "",
    m.summary ? m.summary : ""].filter(Boolean).join("\\n");
}
function rowHead(g, m, roles) {
  const r = roles[m.gene], f = x => x == null ? "—" : Number(x).toFixed(2);
  const via = m.role === "rescued" ? `rescued<span class="muted"> via ${esc(String(m.rescued_via || "").slice(0, 18))}</span>` : esc(m.role || "");
  return `<td class="g" data-tip="${esc(memberTip(m, r))}">${esc(m.gene)} ${promoterFlag(m)}</td>
    <td class="ann">${via}${m.recruited ? ` <span class="flag" style="color:var(--muted)" data-tip="Recruited: no significant program effect of its own; grouped by its profile">recruited</span>` : ""}</td>
    <td class="ann">${annotatedCell(m, r)}</td><td class="ann num">${f(m.stability)}</td><td class="ann num">${f(m.reliability)}</td>`;
}
function hypothesesCard(g) {
  const list = g.roles.filter(r => r.role === "unexplained");
  if (!list.length) return "";
  const info = memberInfo(g);
  return `<div class="card hyp"><h3>Unexplained members — the hypotheses (${list.length})</h3>${list.map(r => `
    <div style="margin:0 0 10px"><span class="chip unexplained">${esc(r.symbol)}</span> <span class="small muted">${esc(clusterRole(info[r.symbol]))} · ${esc(r.confidence)} confidence</span> ${promoterFlag(info[r.symbol])}
      <p style="margin:4px 0 2px">${esc(r.hypothesis)}</p>
      <p class="small" style="margin:0"><b>What would test it:</b> ${esc(r.what_would_test_it)}</p>
      ${(r.pmids || []).length ? `<p class="small" style="margin:2px 0 0">${r.pmids.map(p => pmidLink(String(p).replace(/^PMID[:\\s]*/i, ""))).join(", ")}</p>` : ""}</div>`).join("")}</div>`;
}

// ---- effect signature + member correlation (the main figure) -------------------------------
function effectHeatmap(g) {
  if (!g.signature.length) return `<p class="muted">No effect signature.</p>`;
  const roles = roleOf(g);
  const all = g.rows.flatMap(m => (g.effects[m.gene] || []).filter(c => c).map(c => Math.abs(c[0])));
  const scale = Math.min(3, Math.max(1, ...all));
  const head = `<tr><th>Member</th><th class="ann">Clustering</th><th class="ann">Annotated</th><th class="ann" data-tip="Bootstrap co-assignment with the rest of the group">Stab.</th><th class="ann" data-tip="Share of the member's effect profile that is signal, not noise">Rel.</th>${g.signature.map(s => `<th data-tip="${esc(`P${s.program_id}${s.program_label ? " — " + s.program_label : ""}${MULTI ? " · " + s.condition + " · " + (DESCRIPTIONS[s.condition] || "") : ""}: mean log2FC ${s.mean_log2fc > 0 ? "+" : ""}${s.mean_log2fc.toFixed(2)}, ${s.members_significant_same_direction} of ${s.members} members significant in this direction`)}">
      ${programRef(s.program_id)}${MULTI ? `<br><span class="muted">${esc(s.condition)}</span>` : ""}<br><span class="muted">${s.members_significant_same_direction}/${s.members}</span></th>`).join("")}</tr>`;
  const body = g.rows.map(m => `<tr${m.excluded ? ' class="excl"' : ""}>${rowHead(g, m, roles)}` + (g.effects[m.gene] || g.signature.map(() => null)).map((c, i) => {
      const s = g.signature[i];
      if (!c) return `<td class="c na">n/a</td>`;
      const [v, q] = c, sig = q != null && q < META.alpha;
      const tip = `${m.gene} · P${s.program_id}${MULTI ? " " + s.condition : ""}: log2FC ${v > 0 ? "+" : ""}${v.toFixed(2)}${q != null ? ", adj p " + q.toExponential(1) : ""}${sig ? " — significant" : " — not significant"}${m.excluded ? " (excluded member)" : ""}`;
      return `<td class="c${sig ? " sig" : ""}" style="background:${divColor(v, scale)};color:${inkFor(v, scale)}" data-tip="${esc(tip)}">${v > 0 ? "+" : ""}${v.toFixed(2)}${sig ? "*" : ""}</td>`;
    }).join("") + `</tr>`).join("");
  return `<div class="legend"><span><span class="sw" style="background:var(--div-neg)"></span>negative = knockdown lowers the program (member needed for it)</span>
      <span><span class="sw" style="background:var(--div-pos)"></span>positive = knockdown raises it (member restrains it)</span>
      <span><b>bold, outlined, *</b> = significant (${esc(META.significance)})</span><span class="flag" style="color:var(--serious)">⚠</span> promoter caveat</div>
    <div style="overflow-x:auto"><table class="heat">${head}${body}</table></div>
    <p class="small muted">Columns: the group's effect signature, ranked by |mean log2FC| x the share of members moving it the same way significantly (count under each header). Rows: members in clustered order of their effect-profile correlation${g.excluded.length ? "; members the promoter screen excluded are greyed at the bottom and were not interpreted" : ""}. Colour saturates at |log2FC| = ${scale.toFixed(1)}. Hover a member for its statistics, gene summary and notes; a header for the program label${META.program_viewer ? " (click to open the program)" : ""}.</p>`;
}
let corrMode = "raw";
function correlationHeatmap(g) {
  const c = g.correlation;
  if (!c || !c.genes || c.genes.length < 2) return "";
  const index = Object.fromEntries(c.genes.map((x, i) => [x, i]));
  const rows = g.rows.filter(m => m.gene in index), matrix = c[corrMode];
  const head = `<tr><th></th>${rows.map(m => `<th class="rot"><span>${esc(m.gene)}</span></th>`).join("")}</tr>`;
  const body = rows.map(a => `<tr${a.excluded ? ' class="excl"' : ""}><td class="g">${esc(a.gene)}</td>` + rows.map(b => {
      if (a.gene === b.gene) return `<td class="c na"></td>`;
      const v = matrix[index[a.gene]][index[b.gene]];
      return `<td class="c" style="background:${divColor(v, 1)};color:${inkFor(v, 1)}" data-tip="${esc(`${a.gene} × ${b.gene}: ${corrMode === "raw" ? "raw" : "noise-corrected"} r = ${v.toFixed(2)}`)}">${v.toFixed(2)}</td>`;
    }).join("") + `</tr>`).join("");
  const button = (mode, text) => `<button class="link" style="${corrMode === mode ? "font-weight:700;text-decoration:none;color:var(--text)" : ""}" onclick="corrMode='${mode}';render(currentId,true)">${text}</button>`;
  return `<h3 style="margin-top:18px">Member × member correlation of effect profiles</h3>
    <p class="small" style="margin:0 0 6px">${button("raw", "raw r")} · ${button("corrected", "noise-corrected r")}</p>
    <div style="overflow-x:auto"><table class="heat corr">${head}${body}</table></div>
    <p class="small muted">Pearson r between members' log2FC profiles across every program${MULTI ? " x condition" : ""}; same row order as above. Noise-corrected r = r / √(reliability₁ × reliability₂), capped at ±1 — what the grouping used, so a group of weak regulators can be as coherent as a group of strong ones.</p>`;
}

// ---- STRING network -----------------------------------------------------------------------
function stringNetwork(g) {
  const nodes = g.members.map(m => m.gene), roles = roleOf(g);
  if (!g.edges.length) return `<p class="muted small">No STRING edge among members at combined score ≥ 0.4.</p>`;
  const W = 320, H = 260, cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 40;
  const pos = Object.fromEntries(nodes.map((n, i) => { const a = -Math.PI / 2 + 2 * Math.PI * i / nodes.length;
    return [n, [cx + R * Math.cos(a), cy + R * Math.sin(a)]]; }));
  const lines = g.edges.filter(e => pos[e.a] && pos[e.b]).map(e => { const [x1, y1] = pos[e.a], [x2, y2] = pos[e.b], phys = (e.physical_score || 0) >= 0.4;
    return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${phys ? "var(--accent)" : "var(--muted)"}" stroke-width="${(0.6 + 2.4 * e.score).toFixed(1)}" stroke-opacity="${phys ? 0.8 : 0.5}"${phys ? "" : ' stroke-dasharray="4,3"'}>
      <title>${esc(e.a)} – ${esc(e.b)}: combined ${e.score.toFixed(2)}${e.physical_score ? ", physical " + e.physical_score.toFixed(2) : ""}</title></line>`; }).join("");
  const fill = r => r === "core_explained" ? "var(--accent)" : r === "unexplained" ? "var(--warning)" : "var(--bar)";
  const dots = nodes.map(n => { const [x, y] = pos[n], left = x < cx - 1, right = x > cx + 1;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="7" fill="${fill((roles[n] || {}).role)}" stroke="var(--surface)" stroke-width="1.5"/>
      <text x="${(x + (left ? -10 : right ? 10 : 0)).toFixed(1)}" y="${(y + (y < cy ? -10 : 18)).toFixed(1)}" font-size="11" font-family="var(--mono)" fill="var(--text)" text-anchor="${left ? "end" : right ? "start" : "middle"}">${esc(n)}</text>`; }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="STRING network among members" style="max-width:100%">${lines}${dots}</svg>
    <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>physical ≥ 0.4</span><span><span class="sw" style="background:var(--muted)"></span>functional only</span>
      <span>node: <span style="color:var(--accent)">●</span> core <span style="color:var(--bar)">●</span> consistent <span style="color:var(--warning)">●</span> unexplained</span></div>`;
}

// ---- sections -----------------------------------------------------------------------------
const STATUS = {
  primary_explanation: ["Primary", "var(--critical)"],
  contributing: ["Contributing", "var(--warning)"],
  ruled_out: ["Ruled out", "var(--good)"],
  cannot_assess: ["Cannot assess", "var(--muted)"],
};
function statusCell(s) { const [t, c] = STATUS[s] || [s, "var(--muted)"];
  return `<span class="status"><span class="dot" style="background:${c}"></span>${esc(t)}</span>`; }

function sharedCard(g) {
  const s = g.shared;
  if (!s.claim) return `<div class="slot"><h4>Shared function</h4><p class="muted">Not filled.</p></div>`;
  return `<div class="slot"><h4>Shared function · <span class="muted">${esc(pretty(s.kind))}${s.confidence ? `, ${esc(s.confidence)} confidence` : ""}</span></h4>
    <p>${esc(s.claim)}</p>${(s.support_members || []).length ? `<div>${chips(s.support_members)}</div>` : ""}
    ${(s.support_complexes || []).length ? `<p class="small" style="margin:6px 0 0"><b>Complexes:</b> ${esc(s.support_complexes.join("; "))}</p>` : ""}
    ${(s.support_terms || []).length ? `<p class="small" style="margin:4px 0 0"><b>Terms:</b> ${esc(s.support_terms.join("; "))}</p>` : ""}
    ${(s.pmids || []).length ? `<p class="small">${s.pmids.map(x => pmidLink(String(x).replace(/^PMID[:\\s]*/i, ""))).join(", ")}</p>` : ""}</div>`;
}
function whyCard(g) {
  const w = g.why;
  if (!w.claim) return `<div class="slot"><h4>Why here</h4><p class="muted">Not filled.</p></div>`;
  const rows = (w.programs || []).map(p => `<tr><td>${programRef(p.program_id)} <span class="small muted">${esc(META.program_labels[p.program_id] || "")}</span></td>
      ${MULTI ? `<td>${esc(p.condition)}</td>` : ""}<td>${p.direction === "up" ? "↑ up" : p.direction === "down" ? "↓ down" : esc(p.direction)}</td><td>${esc(p.reading)}</td></tr>`).join("");
  return `<div class="slot"><h4>Why here${w.confidence ? ` · <span class="muted">${esc(w.confidence)} confidence</span>` : ""}</h4><p>${esc(w.claim)}</p>
    ${rows ? `<table><tr><th>Program</th>${MULTI ? "<th>Condition</th>" : ""}<th>Direction</th><th>Reading</th></tr>${rows}</table>` : ""}
    ${(w.pmids || []).length ? `<p class="small">${w.pmids.map(x => pmidLink(String(x).replace(/^PMID[:\\s]*/i, ""))).join(", ")}</p>` : ""}</div>`;
}
function evidenceCard(g) {
  const cx = g.complexes.map(c => { const outside = (c.members_perturbed || []).filter(x => !c.members_in_group.includes(x));
    return `<tr><td>${esc(c.name)}<br><span class="small muted">${esc(c.id)}${(c.sources || []).length ? " · " + esc(c.sources.join(", ")) : ""}</span></td>
      <td>${chips(c.members_in_group)}</td><td>${(c.members_perturbed || []).length} of ${c.size}${outside.length ? `<br><span class="small muted">perturbed, not in group: ${esc(outside.join(", "))}</span>` : ""}</td></tr>`; }).join("");
  const terms = g.enrichment.map(t => `<tr><td>${esc(t.category)}</td><td>${esc(t.description)} <span class="small muted">${esc(t.term)}</span></td>
      <td>${Number(t.fdr).toExponential(1)}</td><td>${t.number_of_genes} / ${t.number_of_genes_in_background}</td><td>${chips(t.genes)}</td></tr>`).join("");
  const ppi = g.ppi ? `PPI enrichment vs ${esc(g.ppi.background || "the screened genes")}: ${esc(g.ppi.number_of_edges)} edges observed, ${esc(g.ppi.expected_number_of_edges)} expected, p = ${Number(g.ppi.p_value).toExponential(1)}` : "PPI enrichment not available";
  const edges = g.edges.map(e => `<tr><td style="font-family:var(--mono)">${esc(e.a)} – ${esc(e.b)}</td><td>${e.score.toFixed(2)}</td><td>${e.physical_score ? e.physical_score.toFixed(2) : "—"}</td></tr>`).join("");
  return `<div class="card"><h3>Curated complexes with ≥ 2 members (${g.complexes.length})</h3>
      ${cx ? `<table><tr><th>Complex</th><th>Members here</th><th>Subunits perturbed / size</th></tr>${cx}</table>` : '<p class="muted small">None.</p>'}</div>
    <div class="card"><h3>STRING among members (${g.edges.length} edges)</h3>${stringNetwork(g)}<p class="small muted">${ppi}</p>
        ${edges ? `<details><summary class="small" style="cursor:pointer">Edge list</summary><table><tr><th>Pair</th><th>Combined</th><th>Physical</th></tr>${edges}</table></details>` : ""}</div>
      <div class="card"><h3>Functional enrichment (top ${g.enrichment.length})</h3>
        ${terms ? `<table><tr><th>Source</th><th>Term</th><th>FDR</th><th>Genes / bg</th><th>Members</th></tr>${terms}</table>` : '<p class="muted small">No term at FDR < 0.05.</p>'}
        <p class="small muted">STRING enrichment; background = the genes perturbed in this screen, not the genome.</p></div>`;
}

function supportLines(s) {
  if (!s.supports.length) return `<span class="flag" style="color:var(--muted)">○ none</span> <span class="small muted">${esc(s.none_reason)}</span>`;
  return s.supports.map(x => {
    const roleBadge = {discovery: ["★ discovery", "var(--good)", "The original study that established this link"],
                       context: ["◆ context", "var(--accent-text)", "Evidence in the matching system, or a database term"],
                       restatement: ["◐ restatement", "var(--serious)", "No original study was among the candidates; this paper states the link"]}[x.role];
    const badge = roleBadge
      ? `<span class="flag" style="color:${roleBadge[1]}" title="${roleBadge[2]}">${roleBadge[0]}</span>`
      : `<span class="flag" style="color:${x.strength === "direct" ? "var(--good)" : "var(--serious)"}">${x.strength === "direct" ? "● direct" : "◐ indirect"}</span>`;
    const t = TITLES[x.pmid] || {};
    if (x.type === "literature") return `<div class="small" style="margin-bottom:6px">${badge} ${pmidLink(x.pmid)} “${esc(x.quote)}”
        <span class="muted">${t.title ? esc(t.title) + " · " : ""}${t.journal ? esc(t.journal) + " " + esc(t.year) : esc(x.year || "")}${x.system && x.system !== "not stated" ? " · " + esc(x.system) : ""}</span></div>`;
    const pm = x.pmid ? ` · GO annotation cites ${x.pmid.split(",").map(pmidLink).join(", ")}` : "";
    const fdr = x.fdr != null ? ` (FDR ${Number(x.fdr).toExponential(1)})` : "";
    return `<div class="small" style="margin-bottom:6px">${badge} <b>${esc(x.source)}</b>: ${esc(String(x.term).slice(0, 160))}${fdr}${pm}</div>`;
  }).join("");
}
function citationsCard(g) {
  const title = pmid => { const t = TITLES[pmid] || {}, p = g.pool[pmid] || {};
    return { title: t.title || p.title || "", year: t.year || p.year || "", journal: t.journal || "" }; };
  const cited = new Set(g.citations.map(c => c.pmid));
  const cites = g.citations.map(c => { const t = title(c.pmid), p = g.pool[c.pmid];
    return `<li>${pmidLink(c.pmid)}${t.title ? ` — ${esc(t.title)} <span class="muted">(${esc([t.journal, t.year].filter(Boolean).join(" "))})</span>` : ""}
      <div class="small">${esc(c.supports)}${c.evidence_system ? ` <span class="muted">· system: ${esc(c.evidence_system)}${c.matched_cell_context === true ? " (matched)" : ""}</span>` : ""}</div>
      ${p && p.sentence ? `<div class="small muted">Pool sentence [${esc((p.genes || []).join(", "))}]: “${esc(p.sentence)}”</div>` : ""}</li>`; }).join("");
  const inline = g.cited_pmids.filter(p => !cited.has(p));
  const support = Object.entries(g.support).map(([k, s]) => `<tr><td style="font-family:var(--mono)">${esc(k.replace(/^regulator:/, ""))}</td><td>${supportLines(s)}</td></tr>`).join("");
  return `<div class="card"><h3>Citations (${g.citations.length})</h3>
      ${cites ? `<ul style="padding-left:18px;margin:0">${cites}</ul>` : '<p class="small muted" style="margin:0">None cited — no paper in the reference pool stated what the annotation claims (or the pool was empty).</p>'}
      ${inline.length ? `<p class="small">Also cited inline: ${inline.map(pmidLink).join(", ")}</p>` : ""}
      <p class="small muted">The annotator could cite only the reference pool: ${Object.keys(g.pool).length} papers naming two members together, plus the curated complexes' references.</p></div>
    ${META.has_citation_pass ? `<details class="card" open><summary>Per-member support — citation pass (${Object.keys(g.support).length})</summary>
      ${support ? `<table><tr><th>Member</th><th>Support</th></tr>${support}</table>` : '<p class="small muted">No citation-pass answer for this group.</p>'}</details>` : ""}`;
}

function qcCard(g) {
  const q = g.qc, notes = [];
  if (q.invalid) notes.push(`${q.invalid} answer(s) failed to parse as JSON and were re-dispatched automatically.`);
  if (q.rejected) notes.push(`${q.rejected} answer(s) failed the gate and were re-dispatched; the rejected versions are kept on disk.`);
  q.warnings.forEach(w => notes.push(`Gate warning: ${esc(w.replace(/^G\\d+: /, ""))}`));
  q.problems.forEach(w => notes.push(`<b>Gate failure:</b> ${esc(w.replace(/^G\\d+: /, ""))}`));
  const clean = !q.warnings.length && !q.problems.length;
  return `<div class="card qc${clean ? " clean" : ""}"><h3>QC</h3>${notes.length ? `<ul class="small" style="margin:0;padding-left:18px">${notes.map(n=>`<li>${n}</li>`).join("")}</ul>` : `<p class="small muted" style="margin:0">Passed the gate first time.</p>`}</div>`;
}

function render(id, keepScroll) {
  const g = GROUPS[id]; if (!g) return;
  currentId = id;
  document.querySelectorAll(".rail a").forEach(a => a.classList.toggle("active", +a.dataset.id === id));
  const active = document.querySelector(".rail a.active"); if (active) active.scrollIntoView({block: "nearest"});
  history.replaceState(null, "", "#group-" + id);
  if (!keepScroll) window.scrollTo(0, 0);
  const confRows = g.confounders.map(c => `<tr><td>${esc(pretty(c.confounder))}</td><td>${statusCell(c.status)}</td><td>${esc(c.evidence)}</td></tr>`).join("");
  const readings = g.readings.map(r => `<tr><td>${esc(r.reading)}</td><td>${esc(r.why_not_excluded)}</td><td>${esc(r.what_would_distinguish_it)}</td></tr>`).join("");
  const openQs = g.open_questions.map(q => `<li>${esc(q.claim)} <span class="muted">— test: ${esc(q.what_would_test_it)}</span></li>`).join("");
  const labelRegs = (g.label_evidence.regulators || []).map(r => `<tr><td style="font-family:var(--mono)">${esc(r.symbol)}</td><td>${esc(r.why)}</td></tr>`).join("");
  const st = g.stats, soft = "background:var(--surface-soft);color:var(--text-soft)";
  const coherenceColour = {strong: "var(--good)", partial: "var(--warning)", weak: "var(--serious)", none: "var(--critical)"}[g.coherence] || "var(--muted)";
  const stabilityTip = `Stability: mean bootstrap co-assignment of the core members. Mean effect-profile r ${st.mean_raw_r != null ? st.mean_raw_r.toFixed(2) : "—"} (${st.mean_corrected_r != null ? st.mean_corrected_r.toFixed(2) : "—"} noise-corrected).`;

  document.getElementById("main").innerHTML = `
    <span class="pill">Group ${id}</span>
    <span class="pill" style="${soft}"><span class="status"><span class="dot" style="background:${coherenceColour}"></span>coherence: ${esc(g.coherence || "—")}</span></span>
    <span class="pill" style="${soft}" data-tip="${esc(stabilityTip)}">stability ${st.stability != null ? st.stability.toFixed(2) : "—"}</span>
    <h1>${esc(g.label)}</h1>
    <p class="sub">${g.family ? `Family: <b>${esc(g.family)}</b>` : "No single family (several unrelated sets)"}${g.distinguisher ? ` · Distinguisher: <b>${esc(g.distinguisher)}</b>` : ""}</p>
    <p class="lead">${esc(g.summary)}</p>
    <div class="card"><h3>Effect signature — each member's knockdown effect on the programs the group moves</h3>${effectHeatmap(g)}${correlationHeatmap(g)}</div>
    <div class="card">${sharedCard(g)}</div>
    ${hypothesesCard(g)}
    <div class="card">${whyCard(g)}</div>
    ${labelRegs ? `<details class="card"><summary>Members the label rests on (${(g.label_evidence.regulators || []).length})</summary><table><tr><th>Member</th><th>Why</th></tr>${labelRegs}</table></details>` : ""}
    ${evidenceCard(g)}
    ${citationsCard(g)}
    <div class="card"><h3>Confounder assessment</h3><table><tr><th>Confounder</th><th>Status</th><th>Deciding evidence</th></tr>${confRows}</table></div>
    <details class="card"><summary>Alternative annotations (${g.readings.length})</summary><p class="small muted">Other annotations of the same group that the evidence does not rule out.</p><table><tr><th>Alternative annotation</th><th>Why it is not ruled out</th><th>What would distinguish it</th></tr>${readings}</table></details>
    ${openQs ? `<details class="card"><summary>Open questions (${g.open_questions.length})</summary><ul>${openQs}</ul></details>` : ""}
    ${qcCard(g)}
    <p class="small muted">Built ${esc(META.built)} from ${esc(META.source)}. Grouping: shared nearest neighbours of the noise-corrected correlation between regulators' effect profiles (log2FC on every program${MULTI ? " x condition" : ""}), kept when stable under bootstrap resampling of the programs${META.n_eligible ? `; ${META.n_eligible} of ${META.n_regulators} regulators were reliable enough to group` : ""}. Members sharing a curated complex with the core and correlating significantly with the group were rescued in. ${META.calibration_used ? "<b>CORUM complexes were partly used to calibrate the grouping parameters</b>, so complex recovery here is not an independent check." : "Curated complexes were not used to set the grouping parameters."} Guides whose effect a neighbouring promoter could explain were excluded before annotation.</p>`;
}

function prev() { const i = IDS.indexOf(currentId); if (i > 0) render(IDS[i-1]); }
function next() { const i = IDS.indexOf(currentId); if (i < IDS.length-1) render(IDS[i+1]); }
document.addEventListener("keydown", e => { if (/input|textarea/i.test(e.target.tagName)) return;
  if (e.key === "ArrowLeft") prev(); if (e.key === "ArrowRight") next(); });

// hover tooltips for chart marks
const tip = document.getElementById("tip");
document.addEventListener("mousemove", e => { const t = e.target.closest("[data-tip]");
  if (!t) { tip.style.opacity = 0; return; }
  tip.textContent = t.dataset.tip; tip.style.opacity = 1;
  tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 330) + "px"; tip.style.top = (e.clientY + 14) + "px"; });

function applyTheme(t) { document.documentElement.dataset.theme = t; document.getElementById("themeBtn").textContent = t === "dark" ? "Light" : "Dark"; }
function toggleTheme() { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("cc-viewer-theme", t); } catch (e) {} applyTheme(t); if (currentId !== null) render(currentId, true); }
let savedTheme = null; try { savedTheme = localStorage.getItem("cc-viewer-theme"); } catch (e) {}
applyTheme(savedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

buildRail();
function idFromHash() { return parseInt((location.hash.match(/group-(\\d+)/) || [])[1]); }
window.addEventListener("hashchange", () => { const id = idFromHash(); if (IDS.includes(id) && id !== currentId) render(id); });
if (IDS.length) render(IDS.includes(idFromHash()) ? idFromHash() : IDS[0]);
else document.getElementById("main").innerHTML = `<p class="muted">No annotated groups.</p>`;
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="the build_group_prompts.py config")
    parser.add_argument("--dispatch", required=True, type=Path, help="group annotation dispatch root")
    parser.add_argument("--arm", default="rg", help="dispatch directory prefix: <arm>_p<group id>")
    parser.add_argument("--citations", type=Path, help="citation-pass dispatch prefix, e.g. dispatch_citations/rgcite")
    parser.add_argument("--program-viewer", help="relative path or URL of the ProgramAnnotatorV3 viewer; program ids link to it")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    evidence, sources, meta, groups_dir = load_from_config(args)
    ids = sorted(
        int(re.search(r"_p(\d+)$", d.name).group(1))
        for d in args.dispatch.glob(f"{args.arm}_p*")
        if (d / "answer.json").exists()
    )
    groups = {}
    for gid in ids:
        entry = evidence.get(str(gid))
        if not entry or entry.get("skipped"):
            print(f"G{gid}: answer without usable evidence ({(entry or {}).get('skipped', 'not in group_evidence.json')}); skipped")
            continue
        groups[gid] = group_fields(gid, entry, sources)

    # Rail: a named confounder as the primary explanation files the group under it; otherwise the
    # label family. One-group families would each get a header of their own, so they are folded
    # into "Other themes" (as in the program viewer).
    for group in groups.values():
        confounder = next((c for c in group["primary_confounders"] if c in CONFOUNDER_GROUPS), None)
        group["rail"] = CONFOUNDER_GROUPS[confounder] if confounder else (group["family"] or "Several unrelated sets")
        group["tag"] = CONFOUNDER_GROUPS[confounder].split(" ")[0].lower() if confounder else ""
    confounder_rails = tuple(dict.fromkeys(CONFOUNDER_GROUPS.values()))
    counts = {}
    for group in groups.values():
        counts[group["rail"]] = counts.get(group["rail"], 0) + 1
    for group in groups.values():
        if group["rail"] not in confounder_rails and counts[group["rail"]] < 2:
            group["rail"] = "Other themes"
    shared = sorted({g["rail"] for g in groups.values()} - set(confounder_rails) - {"Other themes"}, key=str.lower)

    cited = {p for g in groups.values() for p in g["cited_pmids"]}
    cited |= {s["pmid"] for g in groups.values() for claim in g["support"].values()
              for s in claim["supports"] for pmid in [s.get("pmid", "")] if pmid}
    cited = {one for pmid in cited for one in pmid.split(",") if one}
    titles = fetch_titles(cited, groups_dir / "cited_pmid_titles.json") if cited else {}

    program_labels = {s["program_id"]: s["program_label"] for g in groups.values() for s in g["signature"] if s.get("program_label")}
    meta.update(
        rails=shared + ["Other themes", *confounder_rails],
        program_labels=program_labels,
        program_viewer=args.program_viewer or "",
        built=datetime.date.today().isoformat(),
        source=f"{args.dispatch}/{args.arm}_p*/answer.json (blinded claude -p, group gate applied)",
        has_citation_pass=any(g["support"] for g in groups.values()),
    )

    page = (PAGE.replace("__TITLE__", f"{meta['title']} — regulator groups").replace("__GROUPS__", to_js(groups))
            .replace("__TITLES__", to_js(titles)).replace("__META__", to_js(meta)))
    args.output.write_text(page, encoding="utf-8")
    print(f"wrote {len(groups)} groups -> {args.output} ({len(page) / 1e6:.2f} MB); "
          f"{sum(1 for g in groups.values() if g['qc']['problems'])} with gate failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
