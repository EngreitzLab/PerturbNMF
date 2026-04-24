"""
@description
Client utilities for retrieving gene metadata from the Harmonizome website.
It is responsible for fetching per-gene summaries suitable for LLM context assembly.

Key features:
- Default: Short NCBI-like descriptions from API (~400-600 chars).
- Optional: Full literature-based summaries from HTML pages (~2000+ chars) with PMID references.
- Falls back gracefully if requested source unavailable.

@dependencies
- requests: HTTP requests to Harmonizome
- html: HTML entity decoding
- json: Parse embedded JSON summary
- re: Regex for HTML parsing

@examples
- Default (short API summaries):
  client = HarmonizomeClient()
  summaries = client.get_gene_summaries(["NANOG", "SOX2"])

- Full HTML summaries:
  client = HarmonizomeClient(use_full_summaries=True)
  summaries = client.get_gene_summaries(["NANOG", "SOX2"])
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from typing import Dict, List, Optional, Iterable, Tuple

import requests

HARMONIZOME_BASE = "https://maayanlab.cloud/Harmonizome"
HARMONIZOME_API_BASE = f"{HARMONIZOME_BASE}/api/1.0"

logger = logging.getLogger(__name__)


def parse_summary_json(json_str: str) -> Tuple[str, List[Dict]]:
    """Parse Harmonizome summary JSON structure into plain text and references.
    
    The JSON has a tree structure with types: root, p (paragraph), t (text), 
    fg (footnote group), fg_f (footnote ref), rg (reference group), r (reference),
    b (bold), i (italic), a (link).
    
    Returns:
        Tuple of (plain_text_summary, list_of_references)
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse summary JSON: %s", e)
        return "", []
    
    paragraphs = []
    references = []
    
    def extract_text(node: dict) -> str:
        """Recursively extract text from a node."""
        if not isinstance(node, dict):
            return ""
        
        node_type = node.get("type", "")
        
        if node_type == "t":
            # Text node
            return node.get("text", "").strip()
        elif node_type in ("p", "root", "b", "i", "a"):
            # Container nodes - extract children
            children = node.get("children", [])
            return " ".join(extract_text(c) for c in children if isinstance(c, dict)).strip()
        elif node_type == "fg":
            # Footnote group - skip for plain text
            return ""
        elif node_type == "rg":
            # Reference group - process separately
            return ""
        else:
            # Unknown type - try to extract children
            children = node.get("children", [])
            return " ".join(extract_text(c) for c in children if isinstance(c, dict)).strip()
    
    def extract_references(node: dict) -> List[Dict]:
        """Extract reference information from rg (reference group) nodes."""
        refs = []
        if not isinstance(node, dict):
            return refs
        
        if node.get("type") == "rg":
            for child in node.get("children", []):
                if child.get("type") == "r":
                    ref_info = {"ref": child.get("ref")}
                    # Extract text parts
                    text_parts = []
                    pmid = None
                    for c in child.get("children", []):
                        if c.get("type") == "t":
                            text_parts.append(c.get("text", ""))
                        elif c.get("type") == "b":
                            # Title (bold)
                            title = extract_text(c)
                            ref_info["title"] = title
                        elif c.get("type") == "i":
                            # Journal (italic)
                            journal = extract_text(c)
                            ref_info["journal"] = journal
                        elif c.get("type") == "a":
                            # Link - check for PMID
                            href = c.get("href", "")
                            if "pubmed" in href:
                                pmid_match = re.search(r"/(\d+)$", href)
                                if pmid_match:
                                    pmid = pmid_match.group(1)
                    if pmid:
                        ref_info["pmid"] = pmid
                    refs.append(ref_info)
        else:
            # Recurse into children
            for child in node.get("children", []):
                refs.extend(extract_references(child))
        
        return refs
    
    # Process the root node
    if data.get("type") == "root":
        for child in data.get("children", []):
            if child.get("type") == "p":
                para_text = extract_text(child)
                if para_text:
                    # Clean up whitespace
                    para_text = re.sub(r'\s+', ' ', para_text).strip()
                    paragraphs.append(para_text)
            elif child.get("type") == "rg":
                references.extend(extract_references(child))
    
    summary_text = "\n\n".join(paragraphs)
    return summary_text, references


class HarmonizomeClient:
    """Client for Harmonizome gene summaries.
    
    By default, fetches short NCBI-like descriptions from the API.
    Optionally fetches full literature-based summaries from HTML pages
    when use_full_summaries=True.
    """

    def __init__(
        self,
        base_url: str = HARMONIZOME_BASE,
        session: Optional[requests.Session] = None,
        timeout: int = 30,
        sleep_seconds: float = 0.3,
        include_references: bool = False,
        use_full_summaries: bool = False,
    ) -> None:
        """Initialize the client.
        
        Args:
            base_url: Harmonizome base URL
            session: Optional requests session for connection pooling
            timeout: Request timeout in seconds
            sleep_seconds: Delay between requests (rate limiting)
            include_references: If True, append PMIDs to summary text (only with full summaries)
            use_full_summaries: If True, fetch full literature-based summaries from HTML pages.
                               If False (default), use short API descriptions (~400-600 chars).
        """
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/1.0"
        self.session = session or requests.Session()
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.include_references = include_references
        self.use_full_summaries = use_full_summaries

    def get_gene_summary_from_html(self, symbol: str) -> Optional[Tuple[str, List[Dict]]]:
        """Fetch full gene summary from HTML page.
        
        Returns:
            Tuple of (summary_text, references) or None if failed
        """
        url = f"{self.base_url}/gene/{symbol}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("Harmonizome HTML request failed for %s: %s", symbol, exc)
            return None
        
        if resp.status_code != 200:
            logger.warning("Harmonizome HTML returned %s for %s", resp.status_code, symbol)
            return None
        
        # Extract summary JSON from HTML
        # Look for: <div class="summary-content" type="application/json">...</div>
        match = re.search(
            r'<div class="summary-content"[^>]*>\s*(.*?)\s*</div>',
            resp.text,
            re.DOTALL
        )
        
        if not match:
            logger.debug("No summary-content div found for %s", symbol)
            return None
        
        # Decode HTML entities (&#034; -> ")
        json_str = html.unescape(match.group(1))
        
        summary_text, references = parse_summary_json(json_str)
        
        if not summary_text:
            return None
        
        return summary_text, references

    @staticmethod
    def _clean_source_annotations(text: str) -> str:
        """Remove source annotations like [provided by RefSeq, Jun 2016] to save tokens."""
        # Pattern matches: [provided by X, Month Year] or [provided by X]
        cleaned = re.sub(r'\s*\[provided by [^\]]+\]', '', text)
        # Clean up any double spaces
        cleaned = re.sub(r'  +', ' ', cleaned)
        return cleaned.strip()

    def get_gene_summary_from_api(self, symbol: str) -> Optional[str]:
        """Fetch gene description from API (fallback, shorter text)."""
        url = f"{self.api_base}/gene/{symbol}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("Harmonizome API request failed for %s: %s", symbol, exc)
            return None

        if resp.status_code != 200:
            logger.warning("Harmonizome API returned %s for %s", resp.status_code, symbol)
            return None

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("Invalid JSON for %s: %s", symbol, exc)
            return None

        summary = data.get("description") or data.get("name")
        if summary:
            return self._clean_source_annotations(str(summary))
        return None

    def get_gene_summary(self, symbol: str) -> Optional[str]:
        """Fetch gene summary.
        
        If use_full_summaries=True: tries HTML first for full literature-based summary,
        falls back to API if unavailable.
        
        If use_full_summaries=False (default): uses short API description only.
        """
        if self.use_full_summaries:
            # Try HTML first for full summary
            result = self.get_gene_summary_from_html(symbol)
            if result:
                summary_text, references = result
                if self.include_references and references:
                    # Append reference PMIDs
                    pmids = [r.get("pmid") for r in references if r.get("pmid")]
                    if pmids:
                        summary_text += f"\n\nReferences: {', '.join(f'PMID:{p}' for p in pmids)}"
                return summary_text
            
            # Fall back to API
            logger.debug("Falling back to API for %s", symbol)
        
        return self.get_gene_summary_from_api(symbol)

    def get_gene_summaries(self, symbols: Iterable[str]) -> Dict[str, str]:
        """Fetch summaries for multiple genes.
        
        Args:
            symbols: Iterable of gene symbols
            
        Returns:
            Dict mapping symbol to summary text
        """
        results: Dict[str, str] = {}
        symbols_list = list(symbols)
        
        for i, symbol in enumerate(symbols_list):
            summary = self.get_gene_summary(symbol)
            if summary:
                results[symbol] = summary
            else:
                logger.warning("No summary found for %s", symbol)
            
            # Rate limiting
            if self.sleep_seconds and i < len(symbols_list) - 1:
                time.sleep(self.sleep_seconds)
        
        return results
