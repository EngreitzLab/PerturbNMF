"""
@description 
Unified NCBI API wrapper for Pubtator 3 and E-Utilities.
Handles rate limiting, pagination, and data retrieval for gene-literaure integration.

It provides client methods for:
- Pubtator 3 Literature Search (find PMIDs for queries)
- Pubtator 3 BioC-JSON Export (fetch full annotations for PMIDs)
- E-Utilities Gene Summary (fetch official gene descriptions)
- E-Utilities ID Mapping (Symbol -> Entrez ID)

Key features:
- Respects NCBI Rate Limits (3/sec without key, 10/sec with key)
- Automatic Retries for transient failures
- Batch processing helpers

@dependencies
- requests
- time
- os (for env vars)

@examples
- search_literature("(GeneA OR GeneB) AND Context") -> [123, 456]
- fetch_bioc_annotations([123, 456]) -> [BioC_JSON_Objects]
"""

import os
import time
import requests
import logging
from typing import List, Dict, Any, Optional, Union

# URL Constants
PUBTATOR_API_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Rate Limits
RATE_LIMIT_NO_KEY = 0.34  # ~3 req/sec -> 333ms delay
RATE_LIMIT_WITH_KEY = 0.11  # ~9 req/sec -> 111ms delay

logger = logging.getLogger(__name__)

class NcbiClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NBCI Client.
        
        Args:
            api_key: NCBI API Key. If None, checks 'NCBI_API_KEY' env var.
        """
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.sleep_time = RATE_LIMIT_WITH_KEY if self.api_key else RATE_LIMIT_NO_KEY
        self.session = requests.Session()
        
        if self.api_key:
            logger.info("NcbiClient initialized with API Key (High throughput)")
        else:
            logger.warning("NcbiClient initialized WITHOUT API Key (Low throughput)")

    def _get(self, url: str, params: Dict[str, Any], retries: int = 3) -> Optional[requests.Response]:
        """Internal GET with retry and rate limiting."""
        if self.api_key:
            params["api_key"] = self.api_key
            
        for attempt in range(retries):
            try:
                time.sleep(self.sleep_time)
                # Increased timeout to 120s for complex gene queries
                resp = self.session.get(url, params=params, timeout=120)
                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 429:
                    # Too many requests, backoff
                    wait = (attempt + 1) * 2
                    logger.warning(f"Rate limit 429 hit. Sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"Request failed {url}: {resp.status_code} {resp.text[:100]}")
            except requests.RequestException as e:
                logger.warning(f"Request error (attempt {attempt+1}): {e}")
                time.sleep(1)
        return None

    def _post(self, url: str, json_data: Dict[str, Any], retries: int = 3) -> Optional[requests.Response]:
        """Internal POST with retry and rate limiting (API key passed in URL for POSTs sometimes needed, or params)."""
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(retries):
            try:
                time.sleep(self.sleep_time)
                # Note: For some NCBI endpoints, API key is better in params even for POST
                resp = self.session.post(url, params=params, json=json_data, timeout=60)
                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 429:
                    wait = (attempt + 1) * 2
                    logger.warning(f"Rate limit 429 hit. Sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"POST failed {url}: {resp.status_code} {resp.text[:100]}")
            except requests.RequestException as e:
                logger.warning(f"POST error (attempt {attempt+1}): {e}")
                time.sleep(1)
        return None

    # -------------------------------------------------------------------------
    # Pubtator 3
    # -------------------------------------------------------------------------

    def search_literature(self, query: str, page: int = 1, size: int = 20) -> List[int]:
        """
        Search Pubtator 3 for PMIDs.
        Wraps: GET https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/
        
        Args:
            query: Search query string
            page: Page number (1-indexed)
            size: Number of results per page (default 20, max ~100)
        """
        url = f"{PUBTATOR_API_BASE}/search/"
        params = {"text": query, "page": page, "size": min(size, 100)}
        resp = self._get(url, params)
        if not resp:
            return []
        
        try:
            data = resp.json()
            # Structure: {"results": [{"pmid": "123"}, ...], ...}
            # Or sometimes: {"results": [123, 456]} - Need to verify API response shape
            # Based on docs, it returns list of objects with _id or pmid field
            results = data.get("results", [])
            pmids = []
            for item in results:
                if isinstance(item, dict):
                    # Check for common id fields
                    pmid = item.get("_id") or item.get("pmid") or item.get("id")
                    if pmid:
                        pmids.append(int(pmid))
                elif isinstance(item, (int, str)):
                     pmids.append(int(item))
            return pmids
        except ValueError:
            logger.error("Failed to parse JSON from Pubtator search")
            return []

    def fetch_bioc_annotations(self, pmids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch BioC-JSON for a list of PMIDs.
        Wraps: POST https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson
        """
        url = f"{PUBTATOR_API_BASE}/publications/export/biocjson"
        # API expects pmids as list in body? Or comma string?
        # Pubtator3 API docs usually say POST with body `{"pmids": [1, 2]}`
        payload = {"pmids": pmids, "full": True}
        
        resp = self._post(url, payload)
        if not resp:
            return []
        
        # Responses is BioC-JSON (one JSON object usually, or lines?)
        # Typically it's a JSON object with "documents": [...] or just a list
        try:
            raw_text = resp.text.strip()
            # BioC-JSON export often valid JSON.
            data = resp.json()
            # If it's a wrapper, extract docs
            if isinstance(data, dict):
                 if "PubTator3" in data:
                     return data["PubTator3"]
                 return data.get("documents", []) if "documents" in data else [data]
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.error(f"Error parsing BioC JSON: {e}")
            return []

    # -------------------------------------------------------------------------
    # E-Utilities
    # -------------------------------------------------------------------------

    def normalize_genes(self, symbols: List[str], organism: str = "mouse") -> Dict[str, str]:
        """
        Map Gene Symbols to Entrez IDs using ESearch.
        Returns: {Symbol: EntrezID}
        """
        if not symbols:
            return {}
        
        # Build query: (Sym1[Gene] OR Sym2[Gene]) AND organism[Orgn]
        # Chunking needed if list is huge, but usually caller handles chunks.
        # But ESearch GET url length limit ~2k chars.
        
        mapping = {}
        # We can try to do this effectively by querying one by one or small batches
        # Or simpler: "Symbol[Gene] AND Organism[Orgn]"
        
        # A robust way is 'esearch' then 'esummary', but that is slow for many genes.
        # Alternative: Use 'elinky' or simply trust the symbol if unique.
        
        # For this implementation, let's assume we search them one by one or small groups 
        # is too slow.
        # Better approach: return "symbol" as ID if normalization is hard sans-DB.
        # BUT, for Pubtator, Symbol is often NOT enough for ID-based queries if we want precision.
        # However, Pubtator /search/ accepts text. So Symbol is fine for Part 1 (Search).
        
        # Wait, the PLAN says: "Step 1: Search ... Query: (GENE1 OR ...)". 
        # So we actually just need the symbols for the query text.
        
        # For "Step 2: Fetch & Cache", we rely on PMIDs.
        
        # The E-Utilities part is for "Gene Summaries" which NEEDS IDs.
        
        # Let's implement a bulk lookup:
        # esearch term="Sym1[Sym] OR Sym2[Sym] ..." -> this loses the mapping (who is who).
        
        # Correct approach for mapping:
        # Use E-Utils 'esearch' with 'term=Symbol[Gene] AND Organism[Orgn]' one-by-one? 
        # No, too slow.
        # Hack: The 'biotools' approach often downloads a gene_info.gz.
        # Given we are an agent, let's try a batched approach but parse carefully.
        # Actually, for < 1000 genes, single queries are okay-ish if parallel? No.
        
        # Let's use a simpler heuristic or just return the symbol for now if ID not needed for Context.
        # Implementation Plan says: "Fetch gene functional summaries (E-Utilities)".
        # This requires Entrez IDs.
        
        # Let's implement single lookup loop for now (safest).
        # Optimization: Only look up the *Driver* genes (Top 5-10/program).
        
        for sym in symbols:
            term = f"{sym}[Sym] AND {organism}[Orgn]"
            url = f"{EUTILS_BASE}/esearch.fcgi"
            params = {"db": "gene", "term": term, "retmode": "json", "retmax": 1}
            r = self._get(url, params)
            if r:
                try:
                    d = r.json()
                    ids = d.get("esearchresult", {}).get("idlist", [])
                    if ids:
                        mapping[sym] = ids[0]
                except:
                    pass
        return mapping

    def get_gene_summaries(self, gene_ids: List[str]) -> Dict[str, str]:
        """
        Fetch gene summaries from Entrez Gene.
        Args:
            gene_ids: List of Entrez IDs (strings).
        Returns:
            {GeneID: SummaryText}
        """
        if not gene_ids:
            return {}
        
        # ESummary allows batching comma-separated
        # Batch size 100 is safe
        results = {}
        
        # Chunking
        chunk_size = 50
        for i in range(0, len(gene_ids), chunk_size):
            chunk = gene_ids[i:i+chunk_size]
            url = f"{EUTILS_BASE}/esummary.fcgi"
            params = {"db": "gene", "id": ",".join(chunk), "retmode": "json"}
            
            resp = self._get(url, params)
            if not resp:
                continue
                
            try:
                data = resp.json()
                # struct: result: { "uids": [...], "123": {summary: ...}, "456": ... }
                base = data.get("result", {})
                uids = base.get("uids", [])
                for uid in uids:
                     item = base.get(uid, {})
                     summary = item.get("summary", "")
                     if summary:
                         results[str(uid)] = summary
            except Exception as e:
                logger.error(f"Error parsing ESummary JSON: {e}")
                
        return results

