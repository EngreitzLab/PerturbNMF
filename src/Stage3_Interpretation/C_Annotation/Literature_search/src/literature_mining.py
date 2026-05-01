"""
Literature mining utilities for PubTator3 search and evidence extraction.
Extracted from ProgramExplorer/src/literature_mining.py — only the functions
needed by the Literature Search pipeline.
"""
from __future__ import annotations

import re
import logging
import requests
from typing import List, Dict, Optional, Set, Any

logger = logging.getLogger(__name__)

PUBTATOR_API_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DEFAULT_DOMAIN_KEYWORDS = [
    "angiogenesis", "permeability", "barrier", "inflammation", "proliferation",
    "migration", "sprouting", "hypoxia", "metabolism", "junction", "adhesion",
    "leukocyte", "shear", "tip cell", "stalk cell", "arterial", "venous",
    "capillary", "blood-brain barrier", "bbb"
]

DEFAULT_INTERACTION_VERBS = [
    "regulates", "induces", "promotes", "inhibits", "suppresses", "activates",
    "binds", "phosphorylates", "modulates", "mediates", "targets", "controls",
    "decreases", "increases", "blocks", "triggers", "catalyzes"
]


def split_text_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    protected = text
    subs = {
        "et al.": "ET_AL_MARKER", "e.g.": "EG_MARKER", "i.e.": "IE_MARKER",
        "Fig.": "FIG_MARKER", "Ref.": "REF_MARKER", "vs.": "VS_MARKER"
    }
    for k, v in subs.items():
        protected = protected.replace(k, v)
    parts = re.split(r'(?<=[.?!])\s+', protected)
    sentences = []
    for p in parts:
        restored = p
        for k, v in subs.items():
            restored = restored.replace(v, k)
        s = restored.strip()
        if len(s) > 10:
            sentences.append(s)
    return sentences


def extract_evidence_sentences(
    abstract: str, title: str, target_genes: Set[str], context_genes: Set[str],
    domain_keywords: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if domain_keywords is None:
        domain_keywords = DEFAULT_DOMAIN_KEYWORDS
    full_text = f"{title}. {abstract}"
    sentences = split_text_into_sentences(full_text)
    gene_to_sentences: Dict[str, list] = {g: [] for g in target_genes}
    for sent in sentences:
        sent_lower = sent.lower()
        found_targets = []
        for g in target_genes:
            if re.search(rf"\b{re.escape(g)}\b", sent, re.IGNORECASE):
                found_targets.append(g)
        normalized_sent = sent.strip(" .")
        existing_sents = [s[1].strip(" .") for s in [item for sublist in gene_to_sentences.values() for item in sublist]]
        if normalized_sent in existing_sents:
            continue
        score = 0
        for m in context_genes:
            if m not in found_targets and re.search(rf"\b{re.escape(m)}\b", sent, re.IGNORECASE):
                score += 2
        if any(k in sent_lower for k in domain_keywords):
            score += 1
        if any(v in sent_lower for v in DEFAULT_INTERACTION_VERBS):
            score += 1
        for g in found_targets:
            if sent not in [s[1] for s in gene_to_sentences[g]]:
                gene_to_sentences[g].append((score, sent))
    final_map = {}
    for g, items in gene_to_sentences.items():
        if not items:
            continue
        items.sort(key=lambda x: x[0], reverse=True)
        final_map[g] = [x[1] for x in items[:2]]
    return final_map


def find_gene_mentions(doc: Dict[str, Any], member_genes: List[str]) -> List[str]:
    found = set()
    doc_genes = []
    for p in doc.get("passages", []):
        for ann in p.get("annotations", []):
            inf = ann.get("infons", {})
            if inf.get("type") == "Gene":
                txt = ann.get("text", "")
                if txt:
                    doc_genes.append(txt)
    for g in member_genes:
        if any(g.lower() == dg.lower() for dg in doc_genes):
            found.add(g)
    return list(found)


def search_pubtator(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    url = f"{PUBTATOR_API_BASE}/search/"
    params = {"text": query, "page": 1, "size": min(max_results, 100)}
    try:
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code != 200:
            logger.warning(f"PubTator search failed: {resp.status_code}")
            return []
        data = resp.json()
        results = data.get("results", [])[:max_results]
        return [{'pmid': r.get('pmid'), 'title': r.get('title', ''),
                 'score': r.get('score', 0), 'text_hl': r.get('text_hl', '')}
                for r in results]
    except Exception as e:
        logger.error(f"PubTator search error: {e}")
        return []


def fetch_bioc_relations_with_text(pmids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not pmids:
        return {}
    url = f"{PUBTATOR_API_BASE}/publications/export/biocjson"
    payload = {"pmids": pmids}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.warning(f"PubTator BioC fetch failed: {resp.status_code}")
            return {}
        data = resp.json()
        if isinstance(data, dict) and "PubTator3" in data:
            docs = data["PubTator3"]
        elif isinstance(data, list):
            docs = data
        else:
            docs = [data]
        result = {}
        for doc in docs:
            pmid = doc.get('pmid') or doc.get('id')
            if not pmid:
                continue
            title = ''
            abstract = ''
            gene_mentions: Dict[str, set] = {}
            for passage in doc.get('passages', []):
                ptype = passage.get('infons', {}).get('type', '')
                if ptype == 'title':
                    title = passage.get('text', '')
                elif ptype == 'abstract':
                    abstract = passage.get('text', '')
                for ann in passage.get('annotations', []):
                    if ann.get('infons', {}).get('type') == 'Gene':
                        gene_name = ann.get('infons', {}).get('name', '')
                        text_mention = ann.get('text', '')
                        if gene_name and text_mention:
                            if gene_name.upper() not in gene_mentions:
                                gene_mentions[gene_name.upper()] = set()
                            gene_mentions[gene_name.upper()].add(text_mention)
            relations = []
            for rel in doc.get('relations', []):
                infons = rel.get('infons', {})
                r1 = infons.get('role1', {})
                r2 = infons.get('role2', {})
                rel_type = infons.get('type', 'Unknown')
                score = float(infons.get('score', 0))
                if r1.get('type') == 'Gene' and r2.get('type') == 'Gene':
                    if rel_type in ('Positive_Correlation', 'Negative_Correlation'):
                        relations.append({
                            'gene1': r1.get('name', ''), 'gene2': r2.get('name', ''),
                            'type': rel_type, 'score': score
                        })
            gene_mentions_list = {k: list(v) for k, v in gene_mentions.items()}
            result[int(pmid)] = {
                'relations': relations, 'title': title,
                'abstract': abstract, 'gene_mentions': gene_mentions_list
            }
        return result
    except Exception as e:
        logger.error(f"PubTator BioC fetch error: {e}")
        return {}


def search_pubmed(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Search PubMed via E-Utilities (ESearch + ESummary).

    Broader than PubTator3 — finds any PubMed paper, not just gene-annotated ones.
    Returns list of dicts: {pmid, title, source, authors, pubdate}.
    """
    # Step 1: ESearch to get PMIDs
    search_url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(search_url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"PubMed ESearch failed: {resp.status_code}")
            return []
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
    except Exception as e:
        logger.error(f"PubMed ESearch error: {e}")
        return []

    # Step 2: ESummary to get metadata
    summary_url = f"{EUTILS_BASE}/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    try:
        resp = requests.get(summary_url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"PubMed ESummary failed: {resp.status_code}")
            return [{"pmid": int(p), "title": "", "source": "pubmed"} for p in pmids]
        data = resp.json().get("result", {})
        results = []
        for pmid in pmids:
            info = data.get(pmid, {})
            results.append({
                "pmid": int(pmid),
                "title": info.get("title", ""),
                "source": info.get("source", ""),
                "authors": ", ".join(
                    a.get("name", "") for a in info.get("authors", [])[:3]
                ),
                "pubdate": info.get("pubdate", ""),
            })
        return results
    except Exception as e:
        logger.error(f"PubMed ESummary error: {e}")
        return [{"pmid": int(p), "title": "", "source": "pubmed"} for p in pmids]
