"""Fetch, parse, and cache paper metadata from PubTator3 BioC-JSON."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .literature_mining import (
    fetch_bioc_relations_with_text,
    extract_evidence_sentences,
    find_gene_mentions,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperData:
    """Structured data for one paper."""

    pmid: int
    title: str = ""
    abstract: str = ""
    authors: str = ""
    journal: str = ""
    year: int = 0
    gene_mentions: Dict[str, List[str]] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    evidence_sentences: Dict[str, List[str]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "gene_mentions": self.gene_mentions,
            "relations": self.relations,
            "evidence_sentences": self.evidence_sentences,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaperData:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PaperCache:
    """Simple JSON-file-per-PMID cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "papers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, pmid: int) -> Path:
        return self.cache_dir / f"{pmid}.json"

    def get(self, pmid: int) -> Optional[PaperData]:
        p = self._path(pmid)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return PaperData.from_dict(data)
            except Exception:
                return None
        return None

    def put(self, paper: PaperData) -> None:
        p = self._path(paper.pmid)
        # Atomic write: write to temp then rename
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(paper.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.rename(p)


class PaperFetcher:
    """Fetch and cache paper data from PubTator3."""

    def __init__(self, cache_dir: Path):
        self.cache = PaperCache(cache_dir)

    def fetch_papers(
        self,
        pmids: List[int],
        target_genes: List[str],
        context_genes: Optional[List[str]] = None,
        domain_keywords: Optional[List[str]] = None,
    ) -> Dict[int, PaperData]:
        """Fetch BioC data for PMIDs, extract evidence sentences.

        Args:
            pmids: List of PMIDs to fetch.
            target_genes: Genes to look for in evidence sentences (drivers).
            context_genes: Additional genes for sentence scoring (members).
            domain_keywords: Domain-specific keywords for evidence sentence scoring.

        Returns:
            Dict mapping PMID -> PaperData.
        """
        if context_genes is None:
            context_genes = target_genes

        results: Dict[int, PaperData] = {}
        to_fetch: List[int] = []

        # Check cache first
        for pmid in pmids:
            cached = self.cache.get(pmid)
            if cached is not None:
                results[pmid] = cached
            else:
                to_fetch.append(pmid)

        if to_fetch:
            logger.info("Fetching BioC data for %d uncached papers...", len(to_fetch))
            bioc_data = fetch_bioc_relations_with_text(to_fetch)

            for pmid in to_fetch:
                doc = bioc_data.get(pmid, {})
                title = doc.get("title", "")
                abstract = doc.get("abstract", "")
                gene_mentions = doc.get("gene_mentions", {})
                relations = doc.get("relations", [])

                # Extract evidence sentences
                target_set = set(target_genes)
                context_set = set(context_genes)
                evidence = {}
                if abstract or title:
                    evidence = extract_evidence_sentences(
                        abstract, title, target_set, context_set,
                        domain_keywords=domain_keywords,
                    )

                paper = PaperData(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    gene_mentions=gene_mentions,
                    relations=relations,
                    evidence_sentences=evidence,
                )
                self.cache.put(paper)
                results[pmid] = paper

        logger.info(
            "Papers ready: %d cached + %d fetched = %d total",
            len(pmids) - len(to_fetch),
            len(to_fetch),
            len(results),
        )
        return results


def summarize_paper(paper: PaperData, program_context: Dict[str, Any], llm) -> str:
    """Generate a 2-sentence LLM summary of a paper in the context of a program.

    Returns the summary string and also sets paper.summary.
    """
    if not paper.abstract and not paper.title:
        return ""

    system_prompt = (
        "You are a concise scientific summarizer. Given a paper abstract and "
        "the context of a gene expression program, write exactly 2 sentences:\n"
        "1. What the paper is about (main finding).\n"
        "2. How it relates to the gene program (which genes/regulators are mentioned "
        "and what interaction is described).\n\n"
        "If the paper doesn't clearly relate, say so in sentence 2.\n"
        "Return ONLY the 2 sentences, nothing else."
    )

    genes = program_context.get("top_genes", "")
    regulators = program_context.get("regulators", "")
    user_prompt = (
        f"PAPER TITLE: {paper.title}\n\n"
        f"ABSTRACT: {paper.abstract}\n\n"
        f"PROGRAM CONTEXT:\n"
        f"  Top genes: {genes}\n"
        f"  Regulators: {regulators}\n"
    )

    try:
        summary = llm.complete(system_prompt, user_prompt, max_tokens=256)
        paper.summary = summary.strip()
    except Exception as exc:
        logger.warning("Paper summary LLM call failed for PMID:%s: %s", paper.pmid, exc)
        paper.summary = ""

    return paper.summary
