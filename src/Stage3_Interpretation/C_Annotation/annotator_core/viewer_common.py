"""Pieces every annotation viewer shares: answer + citation-pass loading, PubMed titles, the
page stylesheet and safe JSON inlining.

The viewers are single self-contained HTML files (no CDN): data are inlined as JSON with
`to_js`, and every viewer uses VIEWER_CSS so a program page and a regulator-group page look alike.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from answer_io import load_answer
from validate_citation_answers import resolve_literature_ref
from verify_cited_pmids import fetch_pubmed_summaries


def fetch_titles(pmids: set, cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = sorted(p for p in pmids if p not in cache)
    if missing:
        try:
            records = fetch_pubmed_summaries(missing)
            for pmid in missing:
                record = records.get(pmid) or {}
                cache[pmid] = {
                    "title": record.get("title", ""),
                    "year": str(record.get("pubdate", ""))[:4],
                    "journal": record.get("source", ""),
                }
            cache_path.write_text(json.dumps(cache, indent=1))
        except Exception as exc:  # the viewer still builds; titles are a convenience
            print(f"PubMed title lookup failed ({exc}); citations will show PMIDs only")
    return cache


def bare_pmid(value) -> str:
    return re.sub(r"^\s*PMID[:\s]*", "", str(value), flags=re.IGNORECASE).strip()


def load_support(citations_prefix: Optional[Path], pid: int) -> dict:
    """{"gene:SYMBOL" | "regulator:SYMBOL": {"supports": [...], "none_reason": ...}} from the
    citation pass, with each chosen candidate expanded (sentence, title, term) for display."""
    if not citations_prefix:
        return {}
    directory = citations_prefix.parent / f"{citations_prefix.name}_p{pid}"
    if not (directory / "answer.json").exists():
        return {}
    candidates = json.loads((directory / "candidates.json").read_text())
    offered = {c["claim_id"]: c for c in candidates["claims"]}
    support = {}
    for entry in load_answer(directory / "answer.json").get("claims", []):
        claim = offered.get(str(entry.get("claim_id")))
        if not claim:
            continue
        expanded = []
        for chosen in entry.get("supports") or []:
            match = re.fullmatch(r"([LD])(\d+)", str(chosen.get("ref", "")))
            if not match:
                continue
            pool = claim["literature"] if match.group(1) == "L" else claim["database"]
            index = int(match.group(2)) - 1
            if match.group(1) == "L":  # same id-slip resolution as the gate
                pmid = (re.findall(r"\d{6,9}", str(chosen.get("pmid") or "")) or [""])[0]
                resolved, _ = resolve_literature_ref(claim, str(chosen.get("ref", "")), pmid, chosen.get("quote", ""))
                if resolved is None:
                    continue
                index = resolved
            if index >= len(pool):
                continue
            source = pool[index]
            expanded.append({
                "type": "literature" if match.group(1) == "L" else "database",
                "pmid": ",".join(re.findall(r"\d{6,9}", str(chosen.get("pmid") or source.get("pmid", "")))),
                "quote": chosen.get("quote", ""),
                "strength": chosen.get("strength", ""),
                "role": chosen.get("role", ""),
                "year": source.get("year", ""),
                "cited_by": source.get("cited_by"),
                "system": chosen.get("system", ""),
                "why": chosen.get("why", ""),
                "title": source.get("title", ""),
                "term": source.get("term", ""),
                "source": source.get("source", ""),
                "fdr": source.get("fdr"),
            })
        support[f"{claim['kind']}:{claim['symbol']}"] = {
            "supports": expanded, "none_reason": entry.get("none_reason", ""),
        }
    return support


DEFAULT_EFFECT_LABEL = "log2FC"


def read_effect_label(settings: dict) -> str:
    """What the regulator effect column holds, as the viewers label it (axis, headers, tooltips).

    settings.effect_label, default "log2FC". Screens whose `log2_fc` column holds another
    statistic (e.g. a calibrated t-statistic) set it so the pages do not call it a fold change.
    """
    return settings.get("effect_label") or DEFAULT_EFFECT_LABEL


def to_js(obj) -> str:
    """JSON for a <script> block: "</" inside the JSON would end the script block early."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


VIEWER_CSS = """:root {
  color-scheme: light;
  --bg: #f6f6f4; --surface: #fcfcfb; --surface-soft: #f0efec; --border: #e3e2de;
  --text: #0b0b0b; --text-soft: #52514e; --muted: #7a7974;
  --accent: #0d9488; --accent-soft: #d5f0ec; --accent-text: #0b5e57;
  --bar: #2a78d6; --div-neg: #1c5cab; --div-pos: #c43d3d; --div-mid: #f0efec;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --surface-soft: #242422; --border: #33332f;
  --text: #ffffff; --text-soft: #c3c2b7; --muted: #8f8e86;
  --accent: #2dd4bf; --accent-soft: #123b37; --accent-text: #7ee8da;
  --bar: #3987e5; --div-neg: #3987e5; --div-pos: #e66767; --div-mid: #383835;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif; }
a { color: var(--accent-text); }
.top { position: sticky; top: 0; z-index: 10; height: 53px; display: flex; align-items: center; gap: 14px;
  padding: 0 18px; background: var(--surface); border-bottom: 1px solid var(--border); }
.brand { font-weight: 700; white-space: nowrap; }
.brand small { font-weight: 500; color: var(--muted); margin-left: 6px; }
.top input { flex: 1; max-width: 420px; padding: 7px 11px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface-soft); color: var(--text); font: inherit; }
.top .meta { color: var(--muted); font-size: 12.5px; white-space: nowrap; }
.top button { border: 1px solid var(--border); background: var(--surface); color: var(--text-soft);
  border-radius: 8px; padding: 5px 10px; cursor: pointer; font: inherit; font-size: 13px; }
.top button:hover { background: var(--surface-soft); }
.spacer { flex: 1; }
.shell { display: grid; grid-template-columns: 290px minmax(0, 1fr); }
.rail { position: sticky; top: 53px; align-self: start; height: calc(100vh - 53px); overflow-y: auto;
  border-right: 1px solid var(--border); padding: 14px 10px 32px 14px; background: var(--surface); }
.rail h4 { margin: 16px 8px 6px; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.rail h4:first-child { margin-top: 4px; }
.rail a { display: flex; gap: 8px; padding: 6px 8px; border-radius: 8px; color: var(--text-soft);
  font-size: 12.5px; cursor: pointer; text-decoration: none; line-height: 1.35; }
.rail a .num { color: var(--muted); font-variant-numeric: tabular-nums; font-weight: 700; min-width: 26px; }
.rail a .tag { margin-left: auto; font-size: 10.5px; color: var(--muted); white-space: nowrap; }
.rail a:hover { background: var(--surface-soft); }
.rail a.active { background: var(--accent-soft); color: var(--accent-text); font-weight: 650; }
.canvas { padding: 26px clamp(18px, 3vw, 40px) 80px; }
.wrap { max-width: 1180px; margin: 0 auto; }
.pill { display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 9px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent-text); }
h1 { font-size: 27px; line-height: 1.2; margin: 10px 0 4px; }
.sub { color: var(--text-soft); margin: 0 0 12px; font-size: 13.5px; }
.lead { font-size: 15px; margin: 8px 0 18px; max-width: 900px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin: 14px 0; }
.card > h3, details > summary { margin: 0 0 10px; font-size: 14px; font-weight: 700; }
details.card > summary { cursor: pointer; margin: 0; list-style: none; }
details.card { overflow-x: auto; }
details.card > summary::before { content: "▸ "; color: var(--muted); }
details.card[open] > summary::before { content: "▾ "; }
details.card[open] > summary { margin-bottom: 10px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.chip { display: inline-block; font-family: var(--mono); font-size: 12px; padding: 1px 7px; margin: 2px 3px 2px 0;
  border-radius: 6px; background: var(--surface-soft); border: 1px solid var(--border); }
.chip.hit { border-color: var(--accent); color: var(--accent-text); font-weight: 650; }
.muted { color: var(--muted); }
.small { font-size: 12.5px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; }
.status { white-space: nowrap; font-weight: 600; }
.status .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.slot h4 { margin: 0 0 4px; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
.slot p { margin: 0 0 6px; }
.act { display: flex; align-items: flex-end; gap: 2px; height: 96px; margin-top: 6px; }
.act .col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; cursor: default; }
.act .b { width: 70%; background: var(--bar); border-radius: 4px 4px 0 0; min-height: 1px; }
.act .v { font-size: 11px; color: var(--text-soft); margin-bottom: 3px; font-variant-numeric: tabular-nums; }
.axis { display: flex; gap: 2px; border-top: 1px solid var(--muted); }
.axis div { flex: 1; text-align: center; font-size: 11px; color: var(--text-soft); padding-top: 3px; line-height: 1.25; }
.axis .pk { font-weight: 700; color: var(--text); }
.heat { border-collapse: separate; border-spacing: 2px; width: auto; }
.heat td, .heat th { border: 0; padding: 0; }
.heat th { font-size: 11px; text-align: center; padding: 0 2px 3px; text-transform: none; letter-spacing: 0; }
.heat td.g { font-family: var(--mono); font-size: 12px; padding-right: 8px; white-space: nowrap; }
.heat td.c { width: 62px; height: 22px; border-radius: 4px; text-align: center; font-size: 11px;
  font-variant-numeric: tabular-nums; color: var(--text); }
.heat td.c.sig { font-weight: 700; box-shadow: inset 0 0 0 2px var(--text); }
.heat td.c.na { background: transparent; color: var(--muted); }
.legend { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; font-size: 12px; color: var(--text-soft); margin: 4px 0 10px; }
.legend .sw { display: inline-block; width: 14px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.tip { position: fixed; pointer-events: none; z-index: 50; background: var(--text); color: var(--surface);
  font-size: 12px; padding: 5px 8px; border-radius: 6px; opacity: 0; transition: opacity .08s; max-width: 320px; }
.qc { border-left: 3px solid var(--warning); }
.qc.clean { border-left-color: var(--good); }
.flag { font-size: 12px; font-weight: 600; }
button.link { background: none; border: 0; color: var(--accent-text); cursor: pointer; font: inherit; padding: 0; text-decoration: underline; }
@media (max-width: 880px) { .shell { grid-template-columns: 1fr; } .rail { display: none; } }
"""
