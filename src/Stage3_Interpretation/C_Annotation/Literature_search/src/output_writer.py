"""JSON + Markdown + HTML output generation per program."""
from __future__ import annotations

import html as html_lib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_backend import LLMBackend
from .paper_fetcher import PaperData
from .verification import VerificationResult

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------
# JSON output
# ------------------------------------------------------------------


def write_program_json(
    output_dir: Path,
    program_id: int,
    papers: Dict[int, PaperData],
    verifications: Dict[int, List[VerificationResult]],
    search_queries: List[Dict[str, str]],
    program_context: Dict[str, Any],
) -> Path:
    """Write structured JSON for one program."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"program_{program_id}.json"

    data = {
        "program_id": program_id,
        "timestamp": datetime.now().isoformat(),
        "program_context": program_context,
        "search_queries": search_queries,
        "papers": [],
        "statistics": {
            "total_papers": len(papers),
            "total_queries": len(search_queries),
        },
    }

    for pmid, paper in papers.items():
        paper_dict = paper.to_dict()
        paper_dict["verification"] = [
            v.to_dict() for v in verifications.get(pmid, [])
        ]
        # Compute overall badge
        vr_list = verifications.get(pmid, [])
        if vr_list:
            failed = [v for v in vr_list if not v.passed]
            if failed:
                paper_dict["badge"] = failed[0].badge
            else:
                paper_dict["badge"] = "[VERIFIED]"
        else:
            paper_dict["badge"] = "[UNVERIFIED]"
        data["papers"].append(paper_dict)

    # Sort papers: verified first, then by number of evidence sentences
    data["papers"].sort(
        key=lambda p: (
            p["badge"] == "[VERIFIED]",
            sum(len(v) for v in p.get("evidence_sentences", {}).values()),
        ),
        reverse=True,
    )

    # Atomic write
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)
    logger.info("Wrote JSON: %s", path)
    return path


# ------------------------------------------------------------------
# Markdown output
# ------------------------------------------------------------------


def write_program_markdown(
    output_dir: Path,
    program_id: int,
    papers: Dict[int, PaperData],
    verifications: Dict[int, List[VerificationResult]],
    program_context: Dict[str, Any],
    llm: Optional[LLMBackend] = None,
) -> Path:
    """Write formatted Markdown for one program."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"program_{program_id}.md"

    lines = []
    lines.append(f"# Program {program_id} — Literature Evidence\n")

    # Context header
    lines.append("## Program Context\n")
    for key in ("top_genes", "regulators",
                "day_most_active", "cell_types", "go_enrichment"):
        val = program_context.get(key, "")
        if val:
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}**: {val}")
    lines.append("")

    # Per-paper evidence
    lines.append("## Paper Evidence\n")
    sorted_pmids = _sort_papers(papers, verifications)

    for rank, pmid in enumerate(sorted_pmids, 1):
        paper = papers[pmid]
        vr_list = verifications.get(pmid, [])
        badge = _overall_badge(vr_list)

        lines.append(f"### {rank}. {paper.title} {badge}\n")
        lines.append(f"**PMID**: [{paper.pmid}](https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/)")
        if paper.authors:
            lines.append(f"  |  **Authors**: {paper.authors}")
        if paper.journal:
            lines.append(f"  |  **Journal**: {paper.journal} ({paper.year})")
        lines.append("")

        # LLM summary
        if paper.summary:
            lines.append(f"**Summary**: {paper.summary}\n")

        # Evidence sentences
        if paper.evidence_sentences:
            lines.append("**Evidence sentences:**\n")
            for gene, sents in paper.evidence_sentences.items():
                for sent in sents:
                    lines.append(f'- **{gene}**: "{sent}" (PMID:{paper.pmid})')
            lines.append("")

        # Gene-gene relations
        if paper.relations:
            lines.append("**Gene-gene relations (PubTator3):**\n")
            for rel in paper.relations[:5]:
                lines.append(
                    f"- {rel['gene1']} — {rel['type']} — {rel['gene2']} "
                    f"(score: {rel.get('score', 'N/A')})"
                )
            lines.append("")

        # Verification details
        if vr_list:
            lines.append("**Verification:**\n")
            for vr in vr_list:
                status = "PASS" if vr.passed else "FAIL"
                lines.append(
                    f"- L{vr.level} ({vr.method}): {status} "
                    f"[{vr.confidence:.1f}] — {vr.details}"
                )
            lines.append("")

        lines.append("---\n")

    # LLM Synthesis
    if llm is not None and papers:
        lines.append("## Synthesis\n")
        synthesis = _synthesize(llm, program_id, papers, verifications, program_context)
        lines.append(synthesis)
        lines.append("")

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote Markdown: %s", path)
    return path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sort_papers(
    papers: Dict[int, PaperData],
    verifications: Dict[int, List[VerificationResult]],
) -> List[int]:
    """Sort PMIDs: verified first, then by evidence count."""
    def sort_key(pmid):
        vr_list = verifications.get(pmid, [])
        all_passed = all(v.passed for v in vr_list) if vr_list else False
        n_evidence = sum(
            len(sents) for sents in papers[pmid].evidence_sentences.values()
        )
        return (all_passed, n_evidence)

    return sorted(papers.keys(), key=sort_key, reverse=True)


def _overall_badge(vr_list: List[VerificationResult]) -> str:
    if not vr_list:
        return "[UNVERIFIED]"
    failed = [v for v in vr_list if not v.passed]
    if failed:
        return failed[0].badge
    return "[VERIFIED]"


def _synthesize(
    llm: LLMBackend,
    program_id: int,
    papers: Dict[int, PaperData],
    verifications: Dict[int, List[VerificationResult]],
    program_context: Dict[str, Any],
) -> str:
    """Generate LLM synthesis paragraph integrating all evidence."""
    system_prompt = _load_prompt("synthesis.txt")

    # Build user prompt with program context + paper evidence
    parts = []
    parts.append("PROGRAM CONTEXT:")
    for key, val in program_context.items():
        if val:
            parts.append(f"  {key}: {val}")
    parts.append("")
    parts.append("PAPER EVIDENCE:")

    sorted_pmids = _sort_papers(papers, verifications)
    for pmid in sorted_pmids[:15]:  # Limit to top 15 to control tokens
        paper = papers[pmid]
        badge = _overall_badge(verifications.get(pmid, []))
        parts.append(f"\n--- Paper PMID:{pmid} {badge} ---")
        parts.append(f"Title: {paper.title}")
        if paper.evidence_sentences:
            parts.append("Evidence sentences:")
            for gene, sents in paper.evidence_sentences.items():
                for sent in sents:
                    parts.append(f"  [{gene}]: {sent}")
        if paper.relations:
            parts.append("Gene-gene relations:")
            for rel in paper.relations[:3]:
                parts.append(f"  {rel['gene1']} {rel['type']} {rel['gene2']}")

    user_prompt = "\n".join(parts)

    try:
        return llm.complete(system_prompt, user_prompt, max_tokens=2048)
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        return "*Synthesis unavailable — LLM call failed.*"


# ------------------------------------------------------------------
# HTML output
# ------------------------------------------------------------------

_BADGE_COLORS = {
    "[VERIFIED]": ("#16a34a", "#dcfce7"),
    "[GENE_MISMATCH]": ("#dc2626", "#fee2e2"),
    "[WEAK_SUPPORT]": ("#ca8a04", "#fefce8"),
    "[UNVERIFIED]": ("#6b7280", "#f3f4f6"),
}


def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def _badge_html(badge: str) -> str:
    fg, bg = _BADGE_COLORS.get(badge, ("#6b7280", "#f3f4f6"))
    label = badge.strip("[]")
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:0.8em;font-weight:600;color:{fg};background:{bg}">'
        f'{_esc(label)}</span>'
    )


def write_program_html(
    output_dir: Path,
    program_id: int,
    papers: Dict[int, PaperData],
    verifications: Dict[int, List[VerificationResult]],
    program_context: Dict[str, Any],
    synthesis_text: str = "",
) -> Path:
    """Write a self-contained HTML report for one program."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"program_{program_id}.html"

    sorted_pmids = _sort_papers(papers, verifications)
    n_verified = sum(
        1 for pm in sorted_pmids
        if _overall_badge(verifications.get(pm, [])) == "[VERIFIED]"
    )

    parts: List[str] = []
    parts.append(_html_head(program_id))

    # -- Header --
    parts.append(f'<h1>Program {_esc(str(program_id))} &mdash; Literature Evidence</h1>')
    parts.append(
        f'<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} '
        f'&bull; {len(papers)} papers &bull; {n_verified} verified</p>'
    )

    # -- Context card --
    parts.append('<div class="card"><h2>Program Context</h2><table class="ctx">')
    ctx_labels = {
        "top_genes": "Top Genes",
        "regulators": "Regulators",
        "day_most_active": "Most Active Day",
        "cell_types": "Cell Types",
        "go_enrichment": "GO Enrichment",
        "other_enrichment": "Other Enrichment",
    }
    for key, label in ctx_labels.items():
        val = program_context.get(key, "")
        if val:
            # Turn gene lists into pills
            if key in ("top_genes", "regulators"):
                genes = [g.strip() for g in val.replace(";", ",").split(",") if g.strip()]
                val_html = " ".join(f'<span class="gene">{_esc(g)}</span>' for g in genes)
            else:
                val_html = _esc(val)
            parts.append(f'<tr><td class="lbl">{_esc(label)}</td><td>{val_html}</td></tr>')
    parts.append('</table></div>')

    # -- Synthesis --
    if synthesis_text and "unavailable" not in synthesis_text.lower():
        parts.append('<div class="card synthesis"><h2>Synthesis</h2>')
        parts.append(f'<p>{_esc(synthesis_text)}</p></div>')

    # -- Papers --
    parts.append('<h2>Paper Evidence</h2>')
    for rank, pmid in enumerate(sorted_pmids, 1):
        paper = papers[pmid]
        vr_list = verifications.get(pmid, [])
        badge = _overall_badge(vr_list)

        parts.append('<div class="card paper">')
        parts.append(
            f'<h3>{rank}. {_badge_html(badge)} '
            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">'
            f'{_esc(paper.title or "Untitled")}</a></h3>'
        )
        parts.append(
            f'<p class="meta">PMID:{pmid}'
            + (f' &bull; {_esc(paper.authors)}' if paper.authors else '')
            + (f' &bull; {_esc(paper.journal)} ({paper.year})' if paper.journal else '')
            + '</p>'
        )

        # LLM summary
        if paper.summary:
            parts.append(f'<p><strong>Summary:</strong> {_esc(paper.summary)}</p>')

        # Evidence sentences
        if paper.evidence_sentences:
            parts.append('<div class="evidence"><strong>Evidence sentences</strong><ul>')
            for gene, sents in paper.evidence_sentences.items():
                for sent in sents:
                    parts.append(
                        f'<li><span class="gene">{_esc(gene)}</span> '
                        f'&ldquo;{_esc(sent)}&rdquo;</li>'
                    )
            parts.append('</ul></div>')

        # Gene-gene relations
        if paper.relations:
            parts.append('<div class="relations"><strong>Gene-gene relations</strong><ul>')
            for rel in paper.relations[:5]:
                rtype = rel["type"].replace("_", " ")
                score = rel.get("score", "")
                score_str = f' <span class="score">({score:.2f})</span>' if score else ""
                parts.append(
                    f'<li><span class="gene">{_esc(rel["gene1"])}</span> '
                    f'&rarr; <em>{_esc(rtype)}</em> &rarr; '
                    f'<span class="gene">{_esc(rel["gene2"])}</span>{score_str}</li>'
                )
            parts.append('</ul></div>')

        # Verification
        if vr_list:
            parts.append('<details><summary>Verification details</summary><ul class="vr">')
            for vr in vr_list:
                icon = "&#10003;" if vr.passed else "&#10007;"
                cls = "pass" if vr.passed else "fail"
                parts.append(
                    f'<li class="{cls}">{icon} L{vr.level} ({_esc(vr.method)}): '
                    f'{_esc(vr.details)}</li>'
                )
            parts.append('</ul></details>')

        parts.append('</div>')  # .card.paper

    parts.append('</div></body></html>')  # close .container

    content = "\n".join(parts)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote HTML: %s", path)
    return path


def _html_head(program_id: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Program {program_id} — Literature Evidence</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#f8fafc;color:#1e293b;line-height:1.6}}
.container{{max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.6em;margin-bottom:4px}}
h2{{font-size:1.25em;margin:20px 0 10px;color:#334155}}
h3{{font-size:1.05em;margin-bottom:6px}}
h3 a{{color:#1d4ed8;text-decoration:none}}
h3 a:hover{{text-decoration:underline}}
.meta{{font-size:0.85em;color:#64748b;margin-bottom:8px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-bottom:12px}}
.card.synthesis{{background:#f0fdf4;border-color:#bbf7d0}}
.card.paper{{border-left:3px solid #3b82f6}}
table.ctx{{width:100%;border-collapse:collapse}}
table.ctx td{{padding:4px 8px;vertical-align:top;border-bottom:1px solid #f1f5f9}}
table.ctx td.lbl{{font-weight:600;white-space:nowrap;width:180px;color:#475569}}
.gene{{display:inline-block;padding:1px 6px;margin:1px;border-radius:3px;
  font-size:0.85em;font-family:monospace;background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe}}
.evidence ul,.relations ul{{padding-left:18px;margin:4px 0}}
.evidence li,.relations li{{margin-bottom:4px;font-size:0.92em}}
.score{{color:#94a3b8;font-size:0.85em}}
details{{margin-top:8px}}
summary{{cursor:pointer;font-size:0.88em;color:#64748b}}
ul.vr{{list-style:none;padding-left:8px;margin-top:4px}}
ul.vr li{{font-size:0.85em;margin:2px 0}}
ul.vr li.pass{{color:#16a34a}}
ul.vr li.fail{{color:#dc2626}}
</style>
</head><body><div class="container">"""
