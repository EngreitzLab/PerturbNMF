"""paper-qa wrapper: read a directory of PDFs, answer questions, classify gene pairs.

Two entry points, both operating over a local PDF directory (fully offline w.r.t. paper
discovery -- search/download happens upstream in 1.Search/search_literature):

  answer(question, pdf_dir, settings)        -> PQASession (ranked evidence + answer)
  classify_pair(a, b, pdf_dir, settings, ..) -> dict (controlled-vocab relationship)

The controlled vocabulary mirrors GeneProgramExplorer's literature_edges.EVIDENCE_CATEGORIES
so the GPE PaperQA provider can map a category straight onto a contract edge.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from paperqa import Docs, Settings
from paperqa.types import PQASession

# Mirror of GeneProgramExplorer literature_edges.EVIDENCE_CATEGORIES (category names only).
CATEGORIES = [
    "physical_interaction",
    "enzymatic_modification",
    "transcriptional_regulation",
    "same_pathway",
    "same_cellular_phenotype",
    "same_disease",
    "none",
]

_CATEGORY_RE = re.compile(r"CATEGORY\s*[:=]\s*([a-z_]+)", re.IGNORECASE)
_VERDICT_RE = re.compile(r"VERDICT\s*[:=]\s*(supported|contradicted|mixed|insufficient)", re.IGNORECASE)
_DIRECTED_RE = re.compile(r"DIRECTED\s*[:=]\s*(yes|no|true|false)", re.IGNORECASE)

# Edge excerpts must be short — the viewer renders evidence.excerpt unclamped.
SHORT_EXCERPT_CHARS = 200


def first_sentence(text: str, cap: int = SHORT_EXCERPT_CHARS) -> str:
    """First sentence of ``text``, collapsed to one line and capped (ellipsis if cut)."""
    flat = " ".join((text or "").split())
    if not flat:
        return ""
    match = re.search(r"(.+?[.!?])(\s|$)", flat)
    sentence = match.group(1) if match else flat
    if len(sentence) > cap:
        sentence = sentence[: cap - 1].rstrip() + "…"
    return sentence


def short_excerpt_from(answer_text: str, fallback: str = "") -> str:
    """A one-sentence edge excerpt: the justification after the DIRECTED line, else ``fallback``.

    The classification prompt asks for two header lines (CATEGORY/DIRECTED) then a one-sentence
    justification — that trailing sentence is the ideal short edge text.
    """
    tail = ""
    dir_match = _DIRECTED_RE.search(answer_text or "")
    if dir_match:
        tail = answer_text[dir_match.end():]
        # Drop any leftover header tokens / leading punctuation before the prose.
        tail = re.sub(r"^[\s:>\-]*", "", tail)
    candidate = first_sentence(tail) or first_sentence(fallback)
    return candidate


def build_settings(cfg: Optional[dict] = None) -> Settings:
    """Build paper-qa Settings from a config mapping (OpenAI backend by default).

    Throttled (concurrency 1, text-only) so it survives a low OpenAI rate limit; tune via
    cfg['qa'].{max_concurrent_requests,evidence_k}.
    """
    qa = (cfg or {}).get("qa", {}) if cfg else {}
    settings = Settings(
        llm=qa.get("llm", "gpt-4o"),
        summary_llm=qa.get("summary_llm", qa.get("llm", "gpt-4o")),
        embedding=qa.get("embedding", "text-embedding-3-small"),
        temperature=qa.get("temperature", 0.0),
    )
    settings.parsing.multimodal = False  # text only -> fewer LLM calls
    settings.answer.max_concurrent_requests = int(qa.get("max_concurrent_requests", 1))
    settings.answer.evidence_k = int(qa.get("evidence_k", 5))
    return settings


async def _build_docs(pdf_dir, settings: Settings) -> Docs:
    """Index every PDF in ``pdf_dir`` (a folder or a list of folders) into a fresh Docs collection."""
    docs = Docs()
    dirs = [pdf_dir] if isinstance(pdf_dir, (str, Path)) else list(pdf_dir)
    pdfs = sorted({p.resolve() for d in dirs for p in Path(d).glob("*.pdf")})
    for pdf in pdfs:
        try:
            await docs.aadd(str(pdf), settings=settings)
        except Exception as e:  # a single unparsable PDF shouldn't sink the whole run
            print(f"  [warn] paper-qa could not index {pdf.name}: {type(e).__name__}: {str(e)[:200]}")
            continue
    return docs


async def answer(question: str, pdf_dir: Path, settings: Settings) -> PQASession:
    """Answer a free-form question over the PDFs, returning ranked evidence + answer."""
    docs = await _build_docs(pdf_dir, settings)
    return await docs.aquery(question, settings=settings)


def classification_prompt(gene_a: str, gene_b: str, cell_type: str = "") -> str:
    """Prompt paper-qa to classify the A-B relationship into the controlled vocabulary."""
    context = f" in the context of {cell_type}" if cell_type else ""
    return (
        f"Based ONLY on the provided papers, classify how genes {gene_a} and {gene_b} "
        f"are related{context}. "
        "physical_interaction = they directly bind / form a complex; "
        "enzymatic_modification = one post-translationally modifies the other (e.g. phosphorylation); "
        "transcriptional_regulation = one controls the other's transcription; "
        "same_pathway = act in the same pathway/cascade but no direct molecular link is shown; "
        "same_cellular_phenotype = perturbing each causes a similar cellular phenotype, no direct link; "
        "same_disease = both implicated in the same disease; "
        "none = not meaningfully related in these papers. "
        "Prefer the most direct category the papers actually support; bare co-mention is NOT enough. "
        "Respond with exactly two header lines followed by a one-sentence justification:\n"
        "CATEGORY: <one of physical_interaction|enzymatic_modification|transcriptional_regulation|"
        "same_pathway|same_cellular_phenotype|same_disease|none>\n"
        "DIRECTED: <yes if the relationship has a clear source->target direction, else no>\n"
        "Then the justifying sentence."
    )


def _top_context(session: PQASession):
    """Return the highest-scoring context, or None."""
    if not session.contexts:
        return None
    return max(session.contexts, key=lambda c: (c.score if c.score is not None else -1))


def _citation_from_context(ctx, metas: Optional[list] = None) -> dict:
    """Pull title/year/doi/url (+ pmid/pmcid via metas) from a context's source doc."""
    doc = getattr(ctx.text, "doc", None) if ctx is not None else None
    title = getattr(doc, "title", None)
    year = getattr(doc, "year", None)
    doi = getattr(doc, "doi", None)
    url = getattr(doc, "url", None)
    pmids, titles = [], ([title] if title else [])
    if doi and metas:
        # metas = the agent's paper list from Literature_search/<pair>/search_log.json
        for m in metas:
            if (m.get("doi") or "").lower() == doi.lower():
                if m.get("pmid"):
                    pmids.append(str(m["pmid"]))
                if not url:
                    url = m.get("url")
                break
    if not url and doi:
        url = f"https://doi.org/{doi}"
    return {"title": title, "titles": titles, "year": year, "doi": doi, "url": url, "pmids": pmids}


async def classify_pair(gene_a: str, gene_b: str, pdf_dir: Path, settings: Settings,
                        cell_type: str = "", metas: Optional[list] = None) -> dict:
    """Classify the A-B relationship from the PDFs into the controlled vocabulary."""
    docs = await _build_docs(pdf_dir, settings)
    n_pdfs = len(docs.docs)
    result = {
        "source_symbol": gene_a, "target_symbol": gene_b, "cell_type": cell_type,
        "category": None, "directed": False, "excerpt": None, "short_excerpt": None,
        "answer": "", "title": None, "titles": [], "year": None, "doi": None, "url": None,
        "pmids": [], "source_db": "paper-qa", "n_pdfs": n_pdfs,
    }
    if n_pdfs == 0:
        return result

    session = await docs.aquery(classification_prompt(gene_a, gene_b, cell_type), settings=settings)
    answer_text = session.answer or session.formatted_answer or ""
    result["answer"] = answer_text

    cat_match = _CATEGORY_RE.search(answer_text)
    category = cat_match.group(1).lower() if cat_match else None
    if category not in CATEGORIES:
        category = None
    if category == "none":
        category = None
    result["category"] = category

    dir_match = _DIRECTED_RE.search(answer_text)
    result["directed"] = bool(dir_match and dir_match.group(1).lower() in ("yes", "true"))

    top = _top_context(session)
    if top is not None:
        result["excerpt"] = top.context
        result.update({k: v for k, v in _citation_from_context(top, metas).items()})
    # Short, one-sentence text for the contract edge (viewer renders excerpt unclamped).
    result["short_excerpt"] = short_excerpt_from(answer_text, fallback=result.get("excerpt") or "")
    return result


def question_prompt(question: str) -> str:
    """A curation question (llm_query_agent.py) plus a parseable verdict line."""
    return (
        f"{question}\n\nAnswer based ONLY on the provided papers. Start with exactly one header line:\n"
        "VERDICT: <supported|contradicted|mixed|insufficient>\n"
        "then a short answer that names the cell types / species / conditions of the cited evidence."
    )


async def answer_question(question: str, pdf_dirs, settings: Settings, metas: Optional[list] = None,
                          max_citations: int = 3) -> dict:
    """Answer one curation question over the PDFs of several folders; verdict + top citations."""
    docs = await _build_docs(pdf_dirs, settings)
    result = {"question": question, "verdict": None, "answer": "", "citations": [],
              "short_excerpt": None, "n_pdfs": len(docs.docs), "source_db": "paper-qa"}
    if not docs.docs:
        return result
    session = await docs.aquery(question_prompt(question), settings=settings)
    answer_text = session.answer or session.formatted_answer or ""
    result["answer"] = answer_text
    m = _VERDICT_RE.search(answer_text)
    result["verdict"] = m.group(1).lower() if m else None
    ranked = sorted(session.contexts, key=lambda c: (c.score if c.score is not None else -1), reverse=True)
    result["citations"] = [{**_citation_from_context(c, metas), "excerpt": first_sentence(c.context)}
                           for c in ranked[:max_citations]]
    result["short_excerpt"] = first_sentence(_VERDICT_RE.sub("", answer_text).strip())
    return result
