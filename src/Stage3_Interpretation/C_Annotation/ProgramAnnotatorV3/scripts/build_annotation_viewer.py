"""Build a self-contained HTML viewer for a set of v3 gene-program annotations.

Layout (after the GPI "Reading Desk" viewer): a sticky rail listing every program on the left,
grouped by peak condition (multi-condition runs) or by label family (single condition), and one
program at a time in the main pane, with full-text search, #program-N deep links, arrow-key
navigation and dark mode. No CDN and no external files: everything is inline, so the HTML can be
emailed or dropped on any static host.

Per program: label (and the label before the collision pass), brief summary; activity by
condition and the condition dependence (multi-condition); top and distinctive genes (the same genes the
prompt showed); the genes the label rests on WITH the support the citation pass chose for each;
regulator volcano plot(s) with the regulators named in the annotation labelled, plus a table view;
regulator hypotheses with their support; TF motifs in the program's promoters / enhancers as one
table per element type, one row per motif (analysis method, motif (+ family if the name lacks it), sequence logo,
enrichment, FDR, candidate TFs with their evidence tier), strongest families first, logos drawn in
the page from Stage 2 `{K}_motif_logos.json` (config key `motif_logos`; when the config names Stage 2
motif tables; the same selection prompt section E2 showed); non-specific explanations checked; layered
interpretation; modules; alternative program annotations; QC.

Inputs are the same config the prompt builder used (build_annotation_prompts.py), plus the
annotation dispatch directory and, optionally, the citation-pass dispatch directory.

Usage:
    python build_annotation_viewer.py --config my_config.json --dispatch dispatch --arm v3 \
        [--citations dispatch_citations/cite] --output annotation_viewer.html
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from build_annotation_prompts import TOP_LOADING, TOP_UNIQUE, read_motif_tables, select_program_motifs  # noqa: E402
from validate_annotation_answers import validate  # noqa: E402
from conditions import normalise_conditions  # noqa: E402
from viewer_common import VIEWER_CSS, fetch_titles, load_answer, load_support, to_js  # noqa: E402

# The same genes the annotation prompt showed (its sections A and B).
TOP_GENES_SHOWN = TOP_LOADING
DISTINCTIVE_SHOWN = TOP_UNIQUE


def regulator_grid(program_regs: pd.DataFrame, conditions: list) -> tuple:
    labels = [c["label"] for c in conditions]
    significant = program_regs[program_regs["significant"]]
    grid = []
    for gene, rows in program_regs[program_regs["target_gene"].isin(significant["target_gene"])].groupby("target_gene"):
        rows = rows.drop_duplicates("condition").set_index("condition")
        cells = []
        for label in labels:
            if label in rows.index:
                r = rows.loc[label]
                cells.append([round(float(r["log2_fc"]), 3), bool(r["significant"]), float(r["adj_pval"])])
            else:
                cells.append(None)
        n_sig = sum(1 for c in cells if c and c[1])
        best_p = min(c[2] for c in cells if c and c[1])
        grid.append({"gene": gene, "cells": cells, "n_sig": n_sig, "best_p": best_p})
    grid.sort(key=lambda g: (-g["n_sig"], g["best_p"]))
    n_sig = [int(significant[significant["condition"] == label].shape[0]) for label in labels]
    return grid, n_sig


REGULATOR_NAMES: list = []
REGULATOR_INDEX: dict = {}


def volcano_points(program_regs: pd.DataFrame, conditions: list) -> list:
    """Every tested regulator per condition as compact parallel arrays:
    g = index into META.regulator_names, x = log2FC x100, y = -log10(adj p) x100, s = significant."""
    import math
    panels = []
    for condition in conditions:
        rows = program_regs[program_regs["condition"] == condition["label"]]
        panel = {"g": [], "x": [], "y": [], "s": []}
        for gene, fc, q, sig in zip(rows["target_gene"], rows["log2_fc"], rows["adj_pval"], rows["significant"]):
            if pd.isna(fc) or pd.isna(q):
                continue
            if gene not in REGULATOR_INDEX:
                REGULATOR_INDEX[gene] = len(REGULATOR_NAMES)
                REGULATOR_NAMES.append(gene)
            panel["g"].append(REGULATOR_INDEX[gene])
            panel["x"].append(int(round(float(fc) * 100)))
            panel["y"].append(int(round(-math.log10(max(float(q), 1e-300)) * 100)))
            panel["s"].append(1 if sig else 0)
        panels.append(panel)
    return panels


def common_fields(pid: int, sources: dict) -> dict:
    directory = sources["dispatch"] / f"{sources['arm']}_p{pid}"
    answer = load_answer(directory / "answer.json")
    loading = sources["loading"]
    frame = loading[loading["program_id"] == pid].sort_values("Score", ascending=False)
    frame = frame.assign(loading_rank=range(1, len(frame) + 1))
    top = frame.head(TOP_GENES_SHOWN)
    distinctive = (
        frame[~frame["Name"].isin(set(top["Name"]))]
        .sort_values("UniquenessScore", ascending=False)
        .head(DISTINCTIVE_SHOWN)
    )
    regulators = sources["regulators"]
    program_regs = regulators[regulators["program_id"] == pid]
    grid, n_sig = regulator_grid(program_regs, sources["conditions"])
    problems, warnings = validate(pid, directory)
    interpretation = answer.get("interpretation") or {}
    primary = [c.get("confounder") for c in answer.get("confounder_assessment", [])
               if c.get("status") == "primary_explanation"]
    return {
        "id": pid,
        "label": answer.get("label", ""),
        "family": answer.get("label_family", ""),
        "label_before": answer.get("label_before_disambiguation", ""),
        "used_bare_number": bool(answer.get("disambiguation_used_bare_number")),
        "summary": answer.get("brief_summary", ""),
        "coherence": answer.get("coherence", ""),
        "confounders": answer.get("confounder_assessment", []),
        "primary_confounders": primary,
        "slots": {k: interpretation.get(k) for k in ("upstream_trigger", "coregulation_mechanism", "cellular_output")},
        # condition_dependence; temporal_window / group_dependence in answers to older prompts
        "condition_dependence": interpretation.get("condition_dependence") or interpretation.get("temporal_window")
        or interpretation.get("group_dependence") or {},
        "label_genes": (answer.get("label_evidence") or {}).get("genes", []),
        "label_regulators": (answer.get("label_evidence") or {}).get("regulators", []),
        "modules": answer.get("modules", []),
        "readings": answer.get("competing_readings", []),
        "model_regulators": answer.get("regulators", []),
        "open_questions": answer.get("open_questions", []),
        "top_genes": [[g, round(float(v), 5)] for g, v in zip(top["Name"], top["Score"])],
        # [gene, loading rank in this program, number of programs whose gene list includes it]
        "distinctive": [[g, int(r), int(sources["programs_per_gene"][g])]
                        for g, r in zip(distinctive["Name"], distinctive["loading_rank"])],
        "n_program_genes": len(frame),
        "grid": grid,
        "n_sig_by_condition": n_sig,
        "volcano": volcano_points(program_regs, sources["conditions"]),
        "support": load_support(sources.get("citations"), pid),
        "motifs": (select_program_motifs(pid, sources["motif_enrichment"], sources.get("candidate_tfs"))
                   if sources.get("motif_enrichment") is not None else None),
        "activity": None,
        "qc": {
            "rejected": len(list(directory.glob("answer.rejected.*.json"))),
            "invalid": len(list(directory.glob("answer.invalid.*.json"))),
            "problems": problems,
            "warnings": warnings,
        },
    }


def read_motif_logos(config: dict, data: Path) -> dict:
    """Stage 2 logo matrices ({source: {motif: entry}}) from config key `motif_logos` (relative to data_dir);
    {} if not configured."""
    if not config.get("motif_logos"):
        return {}
    return json.loads((data / config["motif_logos"]).read_text()).get("logos", {})


def select_shown_logos(programs: dict, logos: dict) -> dict:
    """Only the logos of motifs the viewer shows, keyed '<source>|<motif>' (keeps the page small). A block
    without a motif source uses the logo file's only source."""
    default_source = next(iter(logos)) if len(logos) == 1 else None
    shown = {}
    for program in programs.values():
        motifs = program.get("motifs") or {}
        for key, families in (motifs.get("families") or {}).items():
            source = motifs.get("section_sources", {}).get(key) or default_source
            for family in families:
                for tf, _, _ in family["motifs"]:
                    entry = logos.get(source, {}).get(tf)
                    if entry is not None:
                        shown[f"{source}|{tf}"] = {"kind": entry["kind"], "matrix": entry["matrix"]}
    return shown


def technical_group(program: dict) -> Optional[str]:
    primary = program["primary_confounders"]
    if "positional" in primary:
        return "Positional"
    # Cell cycle, housekeeping, essentiality and RNA processing are biology; only true artifacts go here.
    if any(c in {"technical_qc", "cis_target_effects", "other_technical"} for c in primary):
        return "Technical artifact"
    return None


def load_from_config(args) -> tuple:
    """Everything dataset-specific comes from the prompt-builder config."""
    config = json.loads(args.config.read_text())
    data = Path(config["data_dir"])
    if not data.is_absolute():
        data = (args.config.parent / data) if not data.exists() else data
    conditions = normalise_conditions(config.get("conditions")) or [
        {"label": "all", "description": config["settings"].get("cell_system", "")}]
    multi = len(conditions) > 1
    regulators = pd.read_csv(data / config.get("regulators_by_condition", config.get("regulators")))
    regulators["significant"] = regulators["significant"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    if "condition" not in regulators.columns:
        regulators["condition"] = conditions[0]["label"]
    activity = pd.read_csv(data / config["program_activity"]) if config.get("program_activity") else None
    loading = pd.read_csv(data / config["gene_loading"])
    if "program_id" not in loading.columns:
        loading = loading.rename(columns={"RowID": "program_id"})
    sources = {"dispatch": args.dispatch, "arm": args.arm, "conditions": conditions,
               "citations": args.citations, "loading": loading, "regulators": regulators,
               "programs_per_gene": loading.groupby("Name")["program_id"].nunique()}
    sources.update(read_motif_tables(config, data))
    sources["motif_logos"] = read_motif_logos(config, data)

    def build(pid: int) -> dict:
        program = common_fields(pid, sources)
        group = technical_group(program)
        program["tag"] = (group or "").split(" ")[0].lower()
        if activity is not None:
            by_condition = activity[activity["program_id"] == pid].set_index("condition")["mean_score"]
            scores = [round(float(by_condition.get(c["label"], 0.0)), 5) for c in conditions]
            total = sum(scores) or 1.0
            peak = conditions[scores.index(max(scores))]
            program.update(activity=scores, share=[round(v / total, 3) for v in scores], peak=peak["label"])
        if multi and activity is not None:
            program["group"] = f"Peak {program['peak']} · {next(c['description'] for c in conditions if c['label'] == program['peak'])}"
        else:
            program["group"] = group or (program["family"] or "Several processes")
        return program

    settings = config["settings"]
    meta = {
        "title": settings.get("dataset_name", "Gene programs"),
        "subtitle": "v3 program annotations",
        "conditions": conditions,
        "groups": [f"Peak {c['label']} · {c['description']}" for c in conditions] if (multi and activity is not None) else None,
        "significance": settings.get("significance_label", "adjusted p < 0.05"),
        "k": int(loading["program_id"].nunique()),
        "motif_source": config.get("motif_enrichment_label") or config.get("motif_enrichment") or "",
        "motif_test": sources.get("motif_test"),
        "motif_logos_all": sources["motif_logos"],
        "logo_default_source": next(iter(sources["motif_logos"])) if len(sources["motif_logos"]) == 1 else None,
    }
    return build, meta, data


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
""" + VIEWER_CSS + """
.motif-table { margin: 4px 0 8px; }
.motif-table td { vertical-align: middle; padding: 2px 8px; }
.motif-table td.num { white-space: nowrap; }
svg.logo { display: block; background: #fff; border-radius: 3px; }
</style>
</head>
<body>
<div class="top">
  <div class="brand" id="brand"></div>
  <input id="search" type="search" placeholder="Search labels, summaries, genes, regulators…" oninput="filterRail()">
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
const PROGRAMS = __PROGRAMS__;
const TITLES = __TITLES__;
const META = __META__;
const LABELS = META.conditions.map(c => c.label);
const WORD = "condition";
const ON = "in";
const DESCRIPTIONS = Object.fromEntries(META.conditions.map(c => [c.label, c.description]));
document.getElementById("brand").innerHTML = `${META.title}<small>${META.subtitle}</small>`;
const IDS = Object.keys(PROGRAMS).map(Number).sort((a,b)=>a-b);
let currentId = null;

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function motifNames(p) {
  if (!p.motifs) return "";
  return Object.values(p.motifs.families).flat().flatMap(f => [f.family, ...f.motifs.map(m => m[0]),
    ...f.candidates.map(c => c.tf)]).join(" ");
}
const SEARCH_INDEX = {};
for (const id of IDS) {
  const p = PROGRAMS[id];
  SEARCH_INDEX[id] = [p.label, p.family, p.label_before, p.summary, p.group,
    p.top_genes.map(g=>g[0]).join(" "), p.distinctive.map(g=>g[0]).join(" "), p.grid.map(g=>g.gene).join(" "),
    (p.condition_dependence||{}).claim, motifNames(p), "P"+id, "program "+id].join(" ").toLowerCase();
}

function artifactTag(p) { return p.tag || ""; }

function buildRail() {
  let out = "";
  for (const g of META.groups) {
    const ids = IDS.filter(id => PROGRAMS[id].group === g);
    if (!ids.length) continue;
    out += `<h4>${esc(g)} (${ids.length})</h4>`;
    out += ids.map(id => { const p = PROGRAMS[id]; const t = artifactTag(p);
      return `<a data-id="${id}" href="#program-${id}" onclick="render(${id});return false;">` +
             `<span class="num">P${id}</span><span>${esc(p.label)}</span>${t ? `<span class="tag">${t}</span>` : ""}</a>`; }).join("");
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
  document.getElementById("meta").textContent = q ? `${shown} of ${IDS.length} programs` : `${IDS.length} programs · ${META.built}`;
}

// ---- charts -------------------------------------------------------------------------------
function activityChart(p) {
  if (!p.activity) return "";
  const max = Math.max(...p.activity) || 1;
  const cols = p.activity.map((v, i) => {
    const h = Math.max(1, Math.round(100 * v / max));
    const tip = `${LABELS[i]} · ${DESCRIPTIONS[LABELS[i]]}: mean score ${v.toFixed(5)} (${Math.round(100*p.share[i])}% of total)`;
    return `<div class="col" data-tip="${esc(tip)}"><div class="v">${Math.round(100*p.share[i])}%</div><div class="b" style="height:${h}%"></div></div>`;
  }).join("");
  const axis = LABELS.map(d => `<div class="${d===p.peak?"pk":""}">${d}<br><span class="muted">${DESCRIPTIONS[d]}</span></div>`).join("");
  return `<div class="act" aria-label="Mean program score by ${WORD}">${cols}</div><div class="axis">${axis}</div>
    <p class="small muted" style="margin:8px 0 0">Mean program score per ${WORD} (share of the total above each bar). Peak: <b>${p.peak}</b>.</p>`;
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

let gridShowAll = false;
function regulatorGrid(p) {
  if (!p.grid.length) return `<p class="muted">No knockdown reached significance.</p>`;
  const rows = gridShowAll ? p.grid : p.grid.slice(0, 30);
  const scale = Math.max(1, ...p.grid.flatMap(g => g.cells.filter(c=>c).map(c => Math.abs(c[0])))) ;
  const capped = Math.min(scale, 3);
  const head = `<tr><th></th>${LABELS.map((d,i)=>`<th>${LABELS.length > 1 ? d : "log2FC"}<br><span class="muted">${p.n_sig_by_condition[i]} sig</span></th>`).join("")}</tr>`;
  const body = rows.map(g => `<tr><td class="g">${esc(g.gene)}</td>` + g.cells.map((c, i) => {
      if (!c) return `<td class="c na">n/a</td>`;
      const [v, sig, q] = c;
      const role = v < 0 ? "activator (knockdown lowers program)" : "repressor (knockdown raises program)";
      const tip = `${g.gene} · ${LABELS[i]}: log2FC ${v>0?"+":""}${v.toFixed(2)}, adj p ${q.toExponential(1)}${sig ? " — significant, " + role : " — not significant"}`;
      return `<td class="c${sig?" sig":""}" style="background:${divColor(v, capped)};color:${inkFor(v, capped)}" data-tip="${esc(tip)}">${v>0?"+":""}${v.toFixed(2)}${sig?"*":""}</td>`;
    }).join("") + `</tr>`).join("");
  const more = p.grid.length > 30
    ? `<p class="small"><button class="link" onclick="gridShowAll=!gridShowAll;render(currentId,true)">${gridShowAll ? "Show top 30" : `Show all ${p.grid.length} regulators`}</button></p>` : "";
  return `<div class="legend"><span><span class="sw" style="background:var(--div-neg)"></span>negative log2FC = knockdown lowers program (activator)</span>
      <span><span class="sw" style="background:var(--div-pos)"></span>positive = knockdown raises program (repressor)</span>
      <span><b>bold, outlined, *</b> = significant (${esc(META.significance)})</span></div>
    <div style="overflow-x:auto"><table class="heat">${head}${body}</table></div>${more}
    <p class="small muted">${LABELS.length > 1 ? `Every regulator significant ${ON} ≥1 ${WORD}, ordered by number of significant ${WORD}s then best adjusted p. Non-significant values are shown so "no effect" can be told from "same direction, below threshold".` : "Every significant regulator, ordered by adjusted p."} Colour saturates at |log2FC| = ${capped.toFixed(1)}.</p>`;
}

// ---- volcano ------------------------------------------------------------------------------
const MAX_LABELS_PER_PANEL = 8;
function keyRegulators(p) {
  const strip = x => String(x || "").replace(/\\s*\\(.*\\)\\s*$/, "").trim();
  const keys = new Set();
  (p.label_regulators || []).forEach(r => keys.add(strip(r.symbol)));
  (p.model_regulators || []).forEach(r => keys.add(strip(r.symbol)));
  Object.keys(p.support || {}).filter(k => k.startsWith("regulator:")).forEach(k => keys.add(k.slice(10)));
  return keys;
}
function volcanoes(p) {
  if (!p.volcano || !p.volcano.some(v => v.x.length)) return `<p class="muted">No regulator results.</p>`;
  const names = META.regulator_names, keys = keyRegulators(p);
  const multi = LABELS.length > 1;
  const W = multi ? 250 : 560, H = multi ? 250 : 320, m = {l: 40, r: 10, t: 34, b: 30};
  // Robust x range: a handful of extreme non-significant knockdowns would otherwise squash the
  // plot. Significant points always fit; anything beyond the range is clamped to the edge.
  const absx = p.volcano.flatMap(v => v.x.map(a => Math.abs(a) / 100)).sort((a, b) => a - b);
  const q995 = absx[Math.floor(0.995 * (absx.length - 1))] || 1;
  const sigmax = Math.max(0, ...p.volcano.flatMap(v => v.x.filter((_, j) => v.s[j]).map(a => Math.abs(a) / 100)));
  const xmax = Math.min(8, Math.max(1, q995, 1.1 * sigmax));
  const ymax = Math.max(1.5, ...p.volcano.flatMap(v => v.y.map(a => a / 100)));
  const sx = x => m.l + (Math.max(-xmax, Math.min(xmax, x)) + xmax) / (2 * xmax) * (W - m.l - m.r);
  const sy = y => H - m.b - y / ymax * (H - m.t - m.b);
  const thr = -Math.log10(0.05);
  const panels = p.volcano.map((v, i) => {
    let dots = "", labels = [];
    const order = v.x.map((_, j) => j).sort((a, b) => v.s[a] - v.s[b]);  // significant drawn last, on top
    for (const j of order) {
      const x = v.x[j] / 100, y = v.y[j] / 100, gene = names[v.g[j]], sig = v.s[j];
      const fill = sig ? (x < 0 ? "var(--div-neg)" : "var(--div-pos)") : "var(--muted)";
      const r = sig ? 3.2 : 1.8, op = sig ? 0.95 : 0.35;
      const tip = `${gene} · ${LABELS[i]}: log2FC ${x > 0 ? "+" : ""}${x.toFixed(2)}, adj p ${Math.pow(10, -y).toExponential(1)}${sig ? (x < 0 ? " — activator" : " — repressor") : " — n.s."}`;
      dots += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="${r}" fill="${fill}" fill-opacity="${op}" stroke="var(--surface)" stroke-width="${sig ? 1 : 0}" data-tip="${esc(tip)}"/>`;
      if (sig) labels.push({gene, x, y, key: keys.has(gene)});
    }
    // Label the key regulators (named in the annotation) first, then the most significant others.
    labels.sort((a, b) => (b.key - a.key) || (b.y - a.y));
    const placed = [];
    let text = "";
    for (const l of labels.slice(0, MAX_LABELS_PER_PANEL)) {
      let lx = sx(l.x) + (l.x < 0 ? -5 : 5), ly = sy(l.y) - 4;
      while (placed.some(q => Math.abs(q.y - ly) < 11 && Math.abs(q.x - lx) < 48)) ly += 11;
      placed.push({x: lx, y: ly});
      text += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10.5" text-anchor="${l.x < 0 ? "end" : "start"}" fill="var(--text)" font-weight="${l.key ? 700 : 400}" font-family="var(--mono)">${esc(l.gene)}</text>`;
    }
    const ticks = [-Math.floor(xmax), 0, Math.floor(xmax)].filter((t, k, a) => a.indexOf(t) === k);
    const yticks = [0, Math.round(ymax / 2), Math.floor(ymax)].filter((t, k, a) => a.indexOf(t) === k);
    const axis = `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--border)"/>
      <line x1="${sx(0)}" x2="${sx(0)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--border)"/>
      <line x1="${m.l}" x2="${W - m.r}" y1="${sy(thr)}" y2="${sy(thr)}" stroke="var(--muted)" stroke-dasharray="3,3"/>
      <text x="${W - m.r}" y="${sy(thr) - 3}" font-size="9.5" text-anchor="end" fill="var(--muted)">adj p = 0.05</text>
      ${ticks.map(t => `<text x="${sx(t)}" y="${H - m.b + 12}" font-size="10" text-anchor="middle" fill="var(--text-soft)">${t}</text>`).join("")}
      ${yticks.map(t => `<text x="${m.l - 4}" y="${sy(t) + 3}" font-size="10" text-anchor="end" fill="var(--text-soft)">${t}</text>`).join("")}
      <text x="${(m.l + W - m.r) / 2}" y="${H - 4}" font-size="10" text-anchor="middle" fill="var(--text-soft)">log2FC (knockdown effect)</text>
      <text x="12" y="${(m.t + H - m.b) / 2}" font-size="10" text-anchor="middle" fill="var(--text-soft)" transform="rotate(-90 10 ${(m.t + H - m.b) / 2})">−log10 adj p</text>
      <text x="${m.l}" y="13" font-size="11.5" font-weight="700" fill="var(--text)">${multi ? LABELS[i] + " · " + esc(DESCRIPTIONS[LABELS[i]]) : ""}</text>
      <text x="${W - m.r}" y="13" font-size="10.5" text-anchor="end" fill="var(--text-soft)">${p.n_sig_by_condition[i]} significant</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Volcano plot ${esc(LABELS[i])}" style="max-width:100%">${axis}${dots}${text}</svg>`;
  }).join("");
  return `<div class="legend"><span><span class="sw" style="background:var(--div-neg)"></span>activator — knockdown lowers the program</span>
      <span><span class="sw" style="background:var(--div-pos)"></span>repressor — knockdown raises it</span>
      <span><span class="sw" style="background:var(--muted);opacity:.5"></span>not significant</span>
      <span><b>bold label</b> = regulator named in the annotation</span></div>
    <div style="display:flex;flex-wrap:wrap;gap:10px">${panels}</div>
    <p class="small muted">All ${p.volcano[0].x.length} tested knockdowns${multi ? ` per ${WORD}, same axes ${ON} every ${WORD}` : ""}; significance: ${esc(META.significance)}. Up to ${MAX_LABELS_PER_PANEL} significant regulators labelled per panel, those named in the annotation first. Hover a point for its values.</p>`;
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
const pretty = s => String(s || "").replace(/_/g, " ");
const pmidLink = id => `<a href="https://pubmed.ncbi.nlm.nih.gov/${esc(id)}/" target="_blank" rel="noopener">PMID ${esc(id)}</a>`;
const chips = (arr, hits) => (arr || []).map(g => `<span class="chip${hits && hits.has(g) ? " hit" : ""}">${esc(g)}</span>`).join("");

function distinctiveChips(p, hits) {
  return p.distinctive.map(([g, rank, n]) => `<span class="chip${hits.has(g) ? " hit" : ""}" data-tip="${esc(`${g}: loading rank ${rank} of ${p.n_program_genes} · in the top ${p.n_program_genes} of ${n} of ${META.k} programs`)}">${esc(g)}</span>`).join("");
}

function slotCard(title, s) {
  if (!s || !s.claim) return `<div class="slot"><h4>${title}</h4><p class="muted">Not filled.</p></div>`;
  const support = [...(s.support_genes||[]), ...(s.support_regulators||[])];
  return `<div class="slot"><h4>${title}${s.confidence ? ` · <span class="muted">${esc(s.confidence)} confidence</span>` : ""}</h4>
    <p>${esc(s.claim)}</p>${support.length ? `<div>${chips(support)}</div>` : ""}
    ${s.mechanism_kind ? `<p class="small muted">Mechanism kind: ${esc(s.mechanism_kind)}</p>` : ""}
    ${(s.pmids||[]).length ? `<p class="small">${s.pmids.map(x=>pmidLink(String(x).replace(/^PMID[:\\s]*/i,""))).join(", ")}</p>` : ""}</div>`;
}

function conditionCard(p) {
  const t = p.condition_dependence || {};
  const rows = (t.regulator_timing || t.regulator_pattern || []).map(r => `<tr><td class="g" style="font-family:var(--mono)">${esc(r.symbol)}</td>
      <td>${esc((r.conditions||[]).join(", "))}</td><td>${esc(pretty(r.pattern))}</td></tr>`).join("");
  const consistent = t.consistent_with_mechanism === true ? "yes" : t.consistent_with_mechanism === false ? "no" : "—";
  return `<p>${esc(t.claim || "")}</p>
    <p class="small muted">Model-reported peak: ${esc(t.peak_condition || "—")} · consistent with proposed mechanism: ${consistent}${t.confidence ? ` · ${esc(t.confidence)} confidence` : ""}</p>
    ${rows ? `<table><tr><th>Regulator</th><th>Condition(s)</th><th>Pattern</th></tr>${rows}</table>` : ""}`;
}

// ---- citation-pass support ----------------------------------------------------------------
function supportCell(p, kind, symbol) {
  const s = (p.support || {})[`${kind}:${symbol}`];
  if (!s) return META.has_citation_pass ? `<span class="small muted" title="The citation pass covers label-evidence regulators and high/medium-confidence hypotheses only.">not checked — low-confidence hypothesis</span>` : "";
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
        <span class="muted">${t.journal ? esc(t.journal) + " " + esc(t.year) : esc(x.year || "")}${x.cited_by ? " · " + x.cited_by + " citations" : ""}${x.system && x.system !== "not stated" ? " · " + esc(x.system) : ""}</span></div>`;
    const pm = x.pmid ? ` · GO annotation cites ${x.pmid.split(",").map(pmidLink).join(", ")}` : "";
    const fdr = x.fdr != null ? ` (FDR ${Number(x.fdr).toExponential(1)})` : "";
    return `<div class="small" style="margin-bottom:6px">${badge} <b>${esc(x.source)}</b>: ${esc(String(x.term).slice(0, 160))}${fdr}${pm}</div>`;
  }).join("");
}

// sequence logo as inline SVG: information content (bits, max 2) or a CWM (positive up, negative down)
const LOGO_COLORS = {A: "#109648", C: "#255C99", G: "#F7B32B", T: "#D62839"};
function logoSvg(entry, width, height) {
  if (!entry) return "";
  const m = entry.matrix, n = m.length, col = width / Math.max(n, 1), cwm = entry.kind === "cwm";
  const pos = m.map(r => r.reduce((a, v) => a + Math.max(v, 0), 0)), neg = m.map(r => r.reduce((a, v) => a + Math.max(-v, 0), 0));
  const top = cwm ? Math.max(...pos, 1e-9) : 2, bottom = cwm ? Math.max(...neg, 0) : 0;
  const scale = height / (top + bottom), base = top * scale;
  let out = "";
  m.forEach((row, i) => {
    const letters = row.map((v, j) => ["ACGT"[j], v]);
    let up = base, down = base;
    letters.filter(l => l[1] > 0).sort((a, b) => a[1] - b[1]).forEach(([b, v]) => { const h = v * scale; if (h < 0.3) return;
      out += `<text transform="translate(${(i * col).toFixed(1)},${up.toFixed(1)}) scale(${(col / 7.2).toFixed(3)},${(h / 7.2).toFixed(3)})" textLength="7.2" lengthAdjust="spacingAndGlyphs" fill="${LOGO_COLORS[b]}">${b}</text>`; up -= h; });
    letters.filter(l => l[1] < 0).sort((a, b) => a[1] - b[1]).forEach(([b, v]) => { const h = -v * scale; if (h < 0.3) return; down += h;
      out += `<text transform="translate(${(i * col).toFixed(1)},${down.toFixed(1)}) scale(${(col / 7.2).toFixed(3)},${(h / 7.2).toFixed(3)})" textLength="7.2" lengthAdjust="spacingAndGlyphs" fill="${LOGO_COLORS[b]}">${b}</text>`; });
  });
  const label = cwm ? "contribution weight matrix (TF-MoDISco pattern)" : "information content (bits)";
  return `<svg class="logo" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="Helvetica,Arial,sans-serif" font-size="10" font-weight="700" role="img"><title>${label}</title>${cwm && bottom ? `<line x1="0" x2="${width}" y1="${base}" y2="${base}" stroke="var(--border)" stroke-width="0.5"/>` : ""}${out}</svg>`;
}
function motifLogo(source, tf) {
  const key = `${source || META.logo_default_source}|${tf}`, entry = (META.motif_logos || {})[key];
  return entry ? logoSvg(entry, Math.min(entry.matrix.length * 7, 150), 26) : "";
}

function candidateChip(c) {
  const support = c.tier === "motif+regulator" ? `knockdown log2FC ${c.log2fc > 0 ? "+" : ""}${c.log2fc.toFixed(2)}, adj p ${c.adj_p.toExponential(1)}`
    : (c.loading_rank ? `loading rank ${c.loading_rank}` : "");
  const tier = {"motif+regulator": "regulator", "motif+expressed_in_program": "in program", "motif+expressed": "expressed"}[c.tier] || c.tier;
  if (c.tier === "motif+expressed") return `<span class="chip" style="opacity:.75" data-tip="${esc(c.tier)} via ${esc(c.motif)}"><i>${esc(c.tf)}</i> <span class="muted small">expressed</span></span>`;
  return `<span class="chip hit" data-tip="${esc(c.tier)} via ${esc(c.motif)}${support ? " · " + esc(support) : ""}"><i>${esc(c.tf)}</i> <span class="small">${tier}${support ? " · " + esc(support) : ""}</span></span>`;
}

function motifCard(p) {
  const m = p.motifs;
  if (!m) return "";
  if (!Object.values(m.n_tested).some(n => n)) return `<div class="card"><h3>TF motifs in program promoters / enhancers</h3><p class="muted small">No motif enrichment results for this program.${META.motif_source ? ` Source: ${esc(META.motif_source)}.` : ""}</p></div>`;
  // one table per element type; one row per significant motif (strongest families first), the analysis method
  // (motif source) as a column, never pooled across methods; candidate TFs on the row of the motif they came via
  const sections = m.sections || [["promoter", "promoter", ""], ["enhancer", "enhancer", ""]];
  const test = META.motif_test || {method: "ttest", n_top: 300}, corr = test.method === "correlation";
  const multiMethod = sections.some(s => s[2]);
  const tables = ["promoter", "enhancer"].map(et => {
    const mine = sections.filter(s => s[1] === et);
    if (!mine.length) return "";
    const counts = mine.map(([key, , source]) => {
      const nf = m.n_families ? m.n_families[key] : m.families[key].length;
      return `${source ? esc(source) + ": " : ""}${m.n_significant[key]} of ${m.n_tested[key]} motifs significant${nf ? ` in ${nf} famil${nf > 1 ? "ies" : "y"}` : ""}${nf > m.families[key].length ? ` (top ${m.families[key].length} families shown)` : ""}`;
    }).join(" · ");
    const rows = mine.flatMap(([key, , source]) => {
      const raw = (m.section_sources || {})[key];
      return m.families[key].flatMap(f => {
        const shown = new Set(f.motifs.map(x => x[0]));
        return f.motifs.map(([tf, e, q], i) => {
          // candidates reached via this motif; a family's candidates via motifs not shown go on its first row
          const cands = f.candidates.filter(c => c.motif === tf || (i === 0 && !shown.has(c.motif)));
          const strong = cands.filter(c => c.tier !== "motif+expressed"), weak = cands.filter(c => c.tier === "motif+expressed");
          const family = String(tf).startsWith(f.family) ? "" : `<div class="muted small">${esc(f.family)}</div>`;
          return `<tr>${multiMethod ? `<td class="small">${esc(source || "")}</td>` : ""}
            <td><span style="font-family:var(--mono);white-space:nowrap">${esc(tf)}</span>${family}</td><td>${motifLogo(raw, tf)}</td>
            <td class="num">${corr ? "r=" + e.toFixed(2) : e.toFixed(2) + "×"}</td><td class="num">${q.toExponential(1)}</td>
            <td class="small">${strong.map(candidateChip).join(" ")}${weak.length ? `${strong.length ? "<br>" : ""}<span class="muted">expressed: ${weak.map(c => `<i>${esc(c.tf)}</i>`).join(", ")}</span>` : ""}</td></tr>`;
        });
      });
    }).join("");
    return `<h4 style="margin:10px 0 4px">${et === "promoter" ? "Promoters" : "Enhancers"} <span class="muted small">· ${counts}</span></h4>
      ${rows ? `<table class="motif-table"><tr>${multiMethod ? "<th>Method</th>" : ""}<th>Motif</th><th>Logo</th><th>${corr ? "Correlation" : "Enrichment"}</th><th>FDR</th><th>Candidate TFs</th></tr>${rows}</table>`
             : `<p class="muted small">No significant motif.</p>`}`;
  }).join("");
  return `<div class="card"><h3>TF motifs in program promoters / enhancers</h3>${tables}
    <p class="small muted">Correlative: ${corr ? "a motif whose per-gene count correlates with the program loadings (FDR &lt; 0.05, r &gt; 0, over expressed genes)" : `a motif over-represented near the top ${test.n_top} genes (FDR &lt; 0.05, enrichment &gt; 1, vs expressed genes)`} nominates a TF family; it does not show that TF acts on the program. Method: FIMO scan of a motif database, or Fi-NeMo hits from ChromBPNet / TF-MoDISco; methods are tested separately. Family = MotifCompendium family (TFClass family for HOCOMOCO). Candidate TFs = expressed TFs on the motif's database TF list: regulator = the TF's knockdown also moves the program; in program = the TF is among the program genes; expressed = expressed only. Logos: information content (FIMO database motif) or the TF-MoDISco contribution weight matrix (Fi-NeMo). The annotator saw these same motifs (evidence section E2).${META.motif_source ? ` Source: ${esc(META.motif_source)}.` : ""}</p></div>`;
}

function qcCard(p) {
  const q = p.qc, notes = [];
  if (p.label_before && p.label_before !== p.label) notes.push(`Collision pass renamed this program: <b>${esc(p.label_before)}</b> → <b>${esc(p.label)}</b>${p.used_bare_number ? " (bare number: nothing in the evidence separated it from a sibling)" : ""}.`);
  else if (p.label_before) notes.push(`In a collision group; label kept verbatim.`);
  if (q.invalid) notes.push(`${q.invalid} answer(s) failed to parse as JSON and were re-dispatched automatically.`);
  if (q.rejected) notes.push(`${q.rejected} answer(s) failed the validator gate (or preceded a prompt fix) and were re-dispatched; the rejected versions are kept on disk.`);
  q.warnings.forEach(w => notes.push(`Validator warning: ${esc(w.replace(/^P\\d+: /, ""))}`));
  q.problems.forEach(w => notes.push(`<b>Validator failure:</b> ${esc(w.replace(/^P\\d+: /, ""))}`));
  const clean = !q.warnings.length && !q.problems.length;
  return `<div class="card qc${clean ? " clean" : ""}"><h3>QC</h3>${notes.length ? `<ul class="small" style="margin:0;padding-left:18px">${notes.map(n=>`<li>${n}</li>`).join("")}</ul>` : `<p class="small muted" style="margin:0">Passed the validator first time; no collision rewrite.</p>`}</div>`;
}

function render(id, keepScroll) {
  const p = PROGRAMS[id]; if (!p) return;
  if (id !== currentId) gridShowAll = false;
  currentId = id;
  document.querySelectorAll(".rail a").forEach(a => a.classList.toggle("active", +a.dataset.id === id));
  const active = document.querySelector(".rail a.active"); if (active) active.scrollIntoView({block: "nearest"});
  history.replaceState(null, "", "#program-" + id);
  if (!keepScroll) window.scrollTo(0, 0);
  const labelGenes = new Set((p.label_genes||[]).map(g => g.symbol));
  const confRows = p.confounders.map(c => `<tr><td>${esc(pretty(c.confounder))}</td><td>${statusCell(c.status)}</td><td>${esc(c.evidence)}</td></tr>`).join("");
  const sym = x => String(x || "").replace(/\\s*\\(.*\\)\\s*$/, "").trim();
  const geneRows = (p.label_genes||[]).map(g => `<tr><td style="font-family:var(--mono)">${esc(g.symbol)}</td><td>${esc(g.loading_rank ?? "")}</td><td>${esc(g.why)}</td><td style="min-width:320px">${supportCell(p, "gene", sym(g.symbol))}</td></tr>`).join("");
  const regRows = (p.model_regulators||[]).map(r => `<tr><td style="font-family:var(--mono)">${esc(r.symbol)}</td><td>${esc(r.role)}</td>
      <td>${r.log2fc != null ? esc(r.log2fc) : ""}</td><td>${esc(r.confidence)}</td><td>${esc(r.hypothesis)}${r.sign_note ? ` <span class="muted">(${esc(r.sign_note)})</span>` : ""}</td><td style="min-width:300px">${supportCell(p, "regulator", sym(r.symbol))}</td></tr>`).join("");
  const modules = (p.modules||[]).map((m, i) => `<div class="card" style="margin:8px 0;background:var(--surface-soft)">
      <b>${i+1}. ${esc(m.name)}</b> <span class="muted small">· ${esc(m.strength)}</span>
      <div style="margin:4px 0">${chips(m.genes)}</div><div class="small">${esc(m.mechanism)}</div></div>`).join("");
  const readings = (p.readings||[]).map(r => `<tr><td>${esc(r.reading)}</td><td>${esc(r.why_not_excluded)}</td><td>${esc(r.what_would_distinguish_it)}</td></tr>`).join("");
  const openQs = (p.open_questions||[]).map(q => `<li>${esc(q.claim)} <span class="muted">— test: ${esc(q.what_would_test_it)}</span></li>`).join("");

  document.getElementById("main").innerHTML = `
    <span class="pill">Program ${id}</span> ${p.peak ? `<span class="pill" style="background:var(--surface-soft);color:var(--text-soft)">peak ${p.peak} · ${DESCRIPTIONS[p.peak]}</span>` : ""} ${supportPill(p)}
    <h1>${esc(p.label)}</h1>
    <p class="sub">coherence: ${esc(p.coherence)}</p>
    <p class="lead">${esc(p.summary)}</p>
    ${p.activity ? `<div class="grid2">
      <div class="card"><h3>Program activity by ${WORD}</h3>${activityChart(p)}</div>
      <div class="card"><h3>Condition dependence</h3>${conditionCard(p)}</div>
    </div>` : ""}
    <div class="card"><h3>Genes</h3>
      <p class="small muted" style="margin:0 0 4px">Top ${p.top_genes.length} by loading (outlined = named as label evidence)</p><div>${chips(p.top_genes.map(g=>g[0]), labelGenes)}</div>
      <p class="small muted" style="margin:10px 0 4px">Most distinctive genes outside the top ${p.top_genes.length}</p><div>${distinctiveChips(p, labelGenes)}</div>
      <p class="small muted" style="margin:6px 0 0">From loading ranks ${p.top_genes.length + 1}–${p.n_program_genes}, the ${p.distinctive.length} genes with the highest uniqueness score = loading × ln((K+1)/(n+1)), where K = ${META.k} programs and n = the number of programs whose top ${p.n_program_genes} genes include the gene: genes that load well here and in few other programs. The annotator saw these same genes. Hover a gene for its rank and n.</p></div>
    <details class="card" open><summary>Genes that drove the label (${(p.label_genes||[]).length}) — with support</summary><table><tr><th>Gene</th><th>Rank</th><th>Why</th><th>Support (citation pass)</th></tr>${geneRows}</table></details>
    <div class="card"><h3>${LABELS.length > 1 ? `Regulators by ${WORD}` : "Significant regulators"}</h3>${volcanoes(p)}
      <details style="margin-top:8px"><summary class="small" style="cursor:pointer">Table view — ${LABELS.length > 1 ? `log2FC of every significant regulator ${ON} every ${WORD}` : "significant regulators"}</summary>${regulatorGrid(p)}</details></div>
    <details class="card" open><summary>Regulator hypotheses (${(p.model_regulators||[]).length}) — with support</summary><table><tr><th>Regulator</th><th>Role</th><th>log2FC</th><th>Conf.</th><th>Hypothesis</th><th>Support (citation pass)</th></tr>${regRows}</table>
      <p class="small muted">Support was sought for label-evidence regulators and high/medium-confidence hypotheses; low-confidence hypotheses were not checked.</p></details>
    ${motifCard(p)}
    <div class="card"><h3>Non-specific explanations checked</h3>
      <p class="small muted" style="margin:0 0 6px">Before reading the biology, the annotator checked whether a technical or non-specific cause explains why these genes vary together: genomic position (neighbouring genes), cell cycle, technical QC, essentiality or growth arrest, RNA processing, CRISPRi effects around the targeted genes, ${LABELS.length > 1 ? "ribosome / housekeeping, and, where assessed, a change in the mix of cells across conditions" : "and ribosome / housekeeping"}. Most statuses are decided by deterministic screens.</p>
      <table><tr><th>Explanation</th><th>Status</th><th>Deciding evidence</th></tr>${confRows}</table></div>
    <div class="card"><h3>Layered interpretation</h3><div class="grid3">
      ${slotCard("Upstream trigger", p.slots.upstream_trigger)}${slotCard("Co-regulation mechanism", p.slots.coregulation_mechanism)}${slotCard("Cellular output", p.slots.cellular_output)}</div></div>
    <details class="card"><summary>Modules (${(p.modules||[]).length})</summary>
      <p class="small muted">A module is a subset of this program's genes that the annotator grouped under a narrower process or mechanism than the label. A program can contain several. Strength says how well the genes and enrichment terms back it: supported, suggestive or speculative.</p>${modules || '<p class="muted">None.</p>'}</details>
    <details class="card"><summary>Alternative program annotations (${(p.readings||[]).length})</summary>
      <p class="small muted">Other annotations of the same gene set that the evidence does not rule out.</p><table><tr><th>Alternative annotation</th><th>Why it is not ruled out</th><th>What would distinguish it</th></tr>${readings}</table></details>
    ${openQs ? `<details class="card"><summary>Open questions</summary><ul>${openQs}</ul></details>` : ""}
    ${qcCard(p)}
    <p class="small muted">Built ${esc(META.built)} from ${esc(META.source)}. 
      ${META.has_citation_pass ? `Citation pass across all programs: ${META.coverage.any_pmid}/${META.coverage.claims} claims with a PMID (${META.coverage.direct_pmid} direct), ${META.coverage.database_only} database term only, ${META.coverage.none} none. A citation means a paper links the gene to the process named in the label — it was retrieved using the label's own words, so it does not test the label.` : ""}</p>`;
}

function supportPill(p) {
  const claims = Object.values(p.support || {});
  if (!claims.length) return "";
  const withPmid = claims.filter(c => c.supports.some(s => s.pmid)).length;
  const none = claims.filter(c => !c.supports.length).length;
  return `<span class="pill" style="background:var(--surface-soft);color:var(--text-soft)" title="Citation pass: claims with a PMID / all claims; ${none} with no support">${withPmid}/${claims.length} claims with a PMID</span>`;
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
function idFromHash() { return parseInt((location.hash.match(/program-(\\d+)/) || [])[1]); }
window.addEventListener("hashchange", () => { const id = idFromHash(); if (IDS.includes(id) && id !== currentId) render(id); });
render(IDS.includes(idFromHash()) ? idFromHash() : IDS[0]);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="the build_annotation_prompts.py config")
    parser.add_argument("--dispatch", required=True, type=Path, help="annotation dispatch root")
    parser.add_argument("--arm", default="v3", help="dispatch directory prefix: <arm>_p<N>")
    parser.add_argument("--citations", type=Path, help="citation-pass dispatch prefix, e.g. dispatch_citations/cite")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build, meta, data_dir = load_from_config(args)
    ids = sorted(
        int(re.search(r"_p(\d+)$", d.name).group(1))
        for d in args.dispatch.glob(f"{args.arm}_p*")
        if (d / "answer.json").exists()
    )
    programs = {pid: build(pid) for pid in ids}
    meta["motif_logos"] = select_shown_logos(programs, meta.pop("motif_logos_all"))
    if not meta["groups"]:
        # Single-condition runs group the rail by family; one-program families would each get a
        # header of their own, so they are folded into "Other themes".
        artifact_groups = ("Positional", "Technical artifact")
        counts = {}
        for program in programs.values():
            counts[program["group"]] = counts.get(program["group"], 0) + 1
        for program in programs.values():
            if program["group"] not in artifact_groups and counts[program["group"]] < 2:
                program["group"] = "Other themes"
        shared = sorted({p["group"] for p in programs.values()} - set(artifact_groups) - {"Other themes"}, key=str.lower)
        meta["groups"] = shared + ["Other themes", *artifact_groups]

    cited = {s["pmid"] for p in programs.values() for claim in p["support"].values()
              for s in claim["supports"] for pmid in [s.get("pmid", "")] if pmid}
    cited = {one for pmid in cited for one in pmid.split(",") if one}
    titles = fetch_titles(cited, data_dir / "cited_pmid_titles.json")

    claims = [c for p in programs.values() for c in p["support"].values()]
    meta.update(
        regulator_names=REGULATOR_NAMES,
        built=datetime.date.today().isoformat(),
        source=f"{args.dispatch}/{args.arm}_p*/answer.json (v3 prompt, blinded claude -p, collision pass applied)",
        has_citation_pass=bool(claims),
        coverage={
            "claims": len(claims),
            "direct_pmid": sum(any(s.get("pmid") and s["strength"] == "direct" for s in c["supports"]) for c in claims),
            "any_pmid": sum(any(s.get("pmid") for s in c["supports"]) for c in claims),
            "database_only": sum(bool(c["supports"]) and not any(s.get("pmid") for s in c["supports"]) for c in claims),
            "none": sum(not c["supports"] for c in claims),
        },
    )

    page = (PAGE.replace("__TITLE__", f"{meta['title']} — v3 program annotations").replace("__PROGRAMS__", to_js(programs)).replace("__TITLES__", to_js(titles))
            .replace("__META__", to_js(meta)))
    args.output.write_text(page, encoding="utf-8")
    print(f"wrote {len(programs)} programs -> {args.output} ({len(page) / 1e6:.1f} MB); "
          f"citation-pass claims: {meta['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
