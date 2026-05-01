"""PubTator3 search orchestration: baseline + LLM queries + dedup."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

from .literature_mining import search_pubtator, search_pubmed

logger = logging.getLogger(__name__)


class SearchEngine:
    """Orchestrates PubTator3 searches and merges results."""

    def __init__(self):
        pass

    def search_baseline(
        self, genes: List[str], max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Baseline search: (Gene1 OR Gene2 OR ... OR GeneN).

        Returns list of dicts with keys: pmid, title, score, text_hl.
        """
        if not genes:
            return []
        gene_or = " OR ".join(genes)
        query = f"({gene_or})"
        logger.info("Baseline query (%d genes): %s...", len(genes), query[:120])
        return search_pubtator(query, max_results=max_results)

    def search_query(
        self, query: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Execute a single PubTator3 query."""
        logger.debug("Executing query: %s", query[:120])
        return search_pubtator(query, max_results=max_results)

    def search_all_for_program(
        self,
        genes: List[str],
        llm_queries: List[Dict[str, str]],
        max_pubtator_results: int = 50,
        max_total: int = 30,
    ) -> List[Dict[str, Any]]:
        """Execute baseline + all LLM queries, dedup, rank by frequency.

        Papers found by multiple queries get higher priority.

        Returns list of dicts: {pmid, title, score, hit_count, sources}.
        """
        # Track which queries found each PMID
        pmid_hits: Counter = Counter()
        pmid_info: Dict[int, Dict[str, Any]] = {}

        # 1) Baseline search
        baseline_results = self.search_baseline(genes, max_results=max_pubtator_results)
        for r in baseline_results:
            pmid = r.get("pmid")
            if pmid is None:
                continue
            pmid = int(pmid)
            pmid_hits[pmid] += 1
            if pmid not in pmid_info:
                pmid_info[pmid] = {
                    "pmid": pmid,
                    "title": r.get("title", ""),
                    "score": r.get("score", 0),
                    "sources": ["baseline"],
                }
            else:
                pmid_info[pmid]["sources"].append("baseline")

        # 2) PubMed search (broader than PubTator3)
        pubmed_results = search_pubmed(f"({' OR '.join(genes[:10])})", max_results=max_pubtator_results)
        for r in pubmed_results:
            pmid = r.get("pmid")
            if pmid is None:
                continue
            pmid = int(pmid)
            pmid_hits[pmid] += 1
            if pmid not in pmid_info:
                pmid_info[pmid] = {
                    "pmid": pmid,
                    "title": r.get("title", ""),
                    "score": r.get("score", 0),
                    "sources": ["pubmed"],
                }
            else:
                pmid_info[pmid]["sources"].append("pubmed")

        # 3) LLM-generated queries
        for q in llm_queries:
            query_str = q.get("query", "")
            strategy = q.get("strategy", "unknown")
            if not query_str:
                continue
            results = self.search_query(query_str, max_results=max_pubtator_results)
            for r in results:
                pmid = r.get("pmid")
                if pmid is None:
                    continue
                pmid = int(pmid)
                pmid_hits[pmid] += 1
                if pmid not in pmid_info:
                    pmid_info[pmid] = {
                        "pmid": pmid,
                        "title": r.get("title", ""),
                        "score": r.get("score", 0),
                        "sources": [strategy],
                    }
                else:
                    pmid_info[pmid]["sources"].append(strategy)

        # 4) Rank by hit count (papers found by multiple queries are more relevant)
        ranked = sorted(
            pmid_info.values(),
            key=lambda x: (pmid_hits[x["pmid"]], x.get("score", 0)),
            reverse=True,
        )
        for item in ranked:
            item["hit_count"] = pmid_hits[item["pmid"]]

        logger.info(
            "Search complete: %d unique PMIDs from %d queries (keeping top %d)",
            len(ranked),
            2 + len(llm_queries),
            max_total,
        )
        return ranked[:max_total]
