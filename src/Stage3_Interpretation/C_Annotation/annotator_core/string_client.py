"""STRING v12 API (https://string-db.org/help/api/) through the shared disk cache.

Three calls, each for one gene set:
  network          edges among the set (combined score, and the physical-subnetwork score)
  enrichment       GO / KEGG / Reactome / ... terms, against a BACKGROUND set — for a group of
                   perturbed genes the background must be the screened targets, not the genome:
                   a CRISPRi library chosen for, say, chromatin regulators makes every subset
                   "enriched" for chromatin against the genome
  ppi_enrichment   whether the set has more STRING edges than random sets of the same size
                   from the background
STRING wants its own identifiers for a background, so symbols are mapped first
(`string_ids`). All requests are form POSTs, so large backgrounds fit.

STRING answers with ITS preferred names (MESDC1 comes back as TLNRD1), so every gene name in a
network or enrichment result is mapped back to the symbol it was queried with.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from http_cache import CachedHttp
from species import TAXON

STRING_API = "https://version-12-0.string-db.org/api/json"
CALLER = "PerturbNMF-annotator"


class StringClient:
    def __init__(self, http: CachedHttp, species: int = TAXON):
        self.http = http
        self.species = species

    def call(self, method: str, fields: dict):
        return self.http.post_form(f"{STRING_API}/{method}", {
            **fields, "species": self.species, "caller_identity": CALLER,
        })

    def string_ids(self, symbols: List[str]) -> Dict[str, str]:
        """symbol -> STRING id, best match only; unmapped symbols are absent."""
        return {q: row["stringId"] for q, row in self.id_records(symbols).items()}

    def id_records(self, symbols: List[str]) -> Dict[str, dict]:
        records: Dict[str, dict] = {}
        for start in range(0, len(symbols), 500):
            chunk = symbols[start:start + 500]
            rows = self.call("get_string_ids", {"identifiers": "\r".join(chunk), "limit": 1, "echo_query": 1}) or []
            for row in rows:
                records.setdefault(row.get("queryItem", ""), row)
        return records

    def to_query_symbol(self, symbols: List[str]) -> Dict[str, str]:
        """STRING preferred name -> the symbol it was queried as."""
        back = {s: s for s in symbols}
        for query, row in self.id_records(symbols).items():
            preferred = row.get("preferredName", query)
            if preferred not in symbols:  # a symbol that was itself queried keeps its own name
                back[preferred] = query
        return back

    def network(self, symbols: List[str], required_score: int = 400) -> List[dict]:
        """Edges among the symbols: {a, b, score, physical_score (0 when none)}."""
        if len(symbols) < 2:
            return []
        rows = self.call("network", {"identifiers": "\r".join(symbols), "required_score": required_score}) or []
        physical = self.call("network", {"identifiers": "\r".join(symbols), "required_score": required_score,
                                         "network_type": "physical"}) or []
        back = self.to_query_symbol(symbols)
        name = lambda value: back.get(value, value)  # noqa: E731
        physical_score = {tuple(sorted((name(r["preferredName_A"]), name(r["preferredName_B"])))): r["score"] for r in physical}
        edges = {}
        for row in rows:
            pair = tuple(sorted((name(row["preferredName_A"]), name(row["preferredName_B"]))))
            edges[pair] = {"a": pair[0], "b": pair[1], "score": round(float(row["score"]), 3),
                           "physical_score": round(float(physical_score.get(pair, 0.0)), 3)}
        return sorted(edges.values(), key=lambda e: -e["score"])

    def enrichment(self, symbols: List[str], background_ids: Optional[List[str]] = None) -> List[dict]:
        fields = {"identifiers": "\r".join(symbols)}
        if background_ids:
            fields["background_string_identifiers"] = "\r".join(background_ids)
        rows = self.call("enrichment", fields) or []
        back = self.to_query_symbol(symbols)
        return [{
            "category": r.get("category", ""), "term": r.get("term", ""), "description": r.get("description", ""),
            "fdr": float(r.get("fdr", 1.0)), "p_value": float(r.get("p_value", 1.0)),
            "genes": [back.get(g, g) for g in r.get("inputGenes", [])], "number_of_genes": int(r.get("number_of_genes", 0)),
            "number_of_genes_in_background": int(r.get("number_of_genes_in_background", 0)),
        } for r in rows if isinstance(r, dict)]

    def ppi_enrichment(self, symbols: List[str], background_ids: Optional[List[str]] = None,
                       required_score: int = 400) -> Optional[dict]:
        if len(symbols) < 2:
            return None
        fields = {"identifiers": "\r".join(symbols), "required_score": required_score}
        if background_ids:
            fields["background_string_identifiers"] = "\r".join(background_ids)
        rows = self.call("ppi_enrichment", fields) or []
        return rows[0] if rows else None
