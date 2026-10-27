"""paper-qa wrapper: index a folder of PDFs once, then answer questions over it.

Fully offline w.r.t. paper discovery -- search / download happen first (gene-function-papers-skill, paper-fetch-skill).

  build_docs(pdf_dir, settings)                        -> Docs (every *.pdf directly in pdf_dir)
  build_docs_from_papers(papers, settings)             -> Docs (listed PDFs, citation from their metadata)
  answer_question(question, docs, settings, metas=..)  -> dict (verdict, answer, top citations)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from paperqa import Docs, Settings

_VERDICT_RE = re.compile(r"VERDICT\s*[:=]\s*(supported|contradicted|mixed|insufficient)", re.IGNORECASE)

# Excerpts must be short — the viewer renders them unclamped.
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


def list_pdfs(pdf_dir) -> list[Path]:
    """Every *.pdf directly in ``pdf_dir`` (not recursive), sorted."""
    return sorted(Path(pdf_dir).glob("*.pdf"))


async def build_docs(pdf_dir, settings: Settings) -> Docs:
    """Index every PDF in ``pdf_dir`` into a fresh Docs collection."""
    docs = Docs()
    for pdf in list_pdfs(pdf_dir):
        try:
            await docs.aadd(str(pdf), settings=settings)
        except Exception as e:  # a single unparsable PDF shouldn't sink the whole run
            print(f"  [warn] paper-qa could not index {pdf.name}: {type(e).__name__}: {str(e)[:200]}")
    return docs


async def build_docs_from_papers(papers: list[dict], settings: Settings, docs: Optional[Docs] = None) -> Docs:
    """Index PDFs from anywhere: papers = [{pdf_path, docname, title, year, journal, doi}].

    The citation, title and DOI come from the paper's metadata, which skips paper-qa's two
    citation-inference LLM calls per PDF. Pass docs to add the papers to an existing collection.
    """
    docs = docs if docs is not None else Docs()
    for p in papers:
        citation = ", ".join(str(x) for x in (p.get("title"), p.get("journal"), p.get("year")) if x)
        try:
            await docs.aadd(str(p["pdf_path"]), citation=citation or None, docname=p.get("docname"),
                            title=p.get("title"), doi=p.get("doi"), settings=settings)
        except Exception as e:  # a single unparsable PDF shouldn't sink the whole run
            print(f"  [warn] paper-qa could not index {p['pdf_path']}: {type(e).__name__}: {str(e)[:200]}")
    return docs


def _citation_from_context(ctx, metas: Optional[list] = None) -> dict:
    """Pull title/year/doi/url (+ pmid via metas) from a context's source doc."""
    doc = getattr(ctx.text, "doc", None) if ctx is not None else None
    title = getattr(doc, "title", None)
    year = getattr(doc, "year", None)
    doi = getattr(doc, "doi", None)
    url = getattr(doc, "url", None)
    pmids = []
    if doi and metas:
        # metas = the agent's paper list from Literature_search/<task>/search_log.json
        for m in metas:
            if (m.get("doi") or "").lower() == doi.lower():
                if m.get("pmid"):
                    pmids.append(str(m["pmid"]))
                if not url:
                    url = m.get("url")
                break
    if not url and doi:
        url = f"https://doi.org/{doi}"
    return {"title": title, "year": year, "doi": doi, "url": url, "pmids": pmids}


def question_prompt(question: str) -> str:
    """A question plus a parseable verdict line."""
    return (
        f"{question}\n\nAnswer based ONLY on the provided papers. Start with exactly one header line:\n"
        "VERDICT: <supported|contradicted|mixed|insufficient>\n"
        "then a short answer that names the cell types / species / conditions of the cited evidence."
    )


async def answer_question(question: str, docs: Docs, settings: Settings, metas: Optional[list] = None,
                          max_citations: int = 3, prompt: Optional[str] = None) -> dict:
    """Answer one question over an indexed Docs collection; verdict + top citations.

    prompt replaces question_prompt(question) as the text sent to paper-qa (the verdict is then
    None unless the prompt asks for a VERDICT line).
    """
    result = {"question": question, "verdict": None, "answer": "", "citations": [],
              "short_excerpt": None, "n_pdfs": len(docs.docs), "source_db": "paper-qa"}
    if not docs.docs:
        return result
    session = await docs.aquery(prompt or question_prompt(question), settings=settings)
    result["cost_usd"] = float(getattr(session, "cost", 0.0) or 0.0)   # paper-qa's LLM cost of this query
    answer_text = session.answer or session.formatted_answer or ""
    result["answer"] = answer_text
    m = _VERDICT_RE.search(answer_text)
    result["verdict"] = m.group(1).lower() if m else None
    ranked = sorted(session.contexts, key=lambda c: (c.score if c.score is not None else -1), reverse=True)
    result["citations"] = [{**_citation_from_context(c, metas), "excerpt": first_sentence(c.context)}
                           for c in ranked[:max_citations]]
    # every evidence summary paper-qa used, in score order (for a downstream judge)
    result["evidence"] = [{"title": _citation_from_context(c, metas)["title"], "score": c.score, "summary": c.context}
                          for c in ranked]
    result["short_excerpt"] = first_sentence(_VERDICT_RE.sub("", answer_text).strip())
    return result
