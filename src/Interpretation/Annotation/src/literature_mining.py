"""
Program-level literature retrieval and evidence gathering.
Ported from ProgExplorer/pipeline/02_fetch_ncbi_data.py.

Library module — no CLI/argparse code.
"""
from __future__ import annotations

import re
import json
import logging
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from collections import Counter

from .ncbi_api import NcbiClient
from .harmonizome_api import HarmonizomeClient
from .string_api import (
    get_regulator_program_interactions,
    batch_validate_regulators,
)
from .column_mapper import standardize_regulator_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers (duplicated from gene_extraction to avoid circular imports)
# ---------------------------------------------------------------------------

def _ensure_program_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "program_id" in df.columns:
        return df
    if "RowID" not in df.columns:
        raise ValueError("CSV must have 'program_id' or 'RowID'")
    updated = df.copy()
    updated["program_id"] = updated["RowID"]
    return updated


def _add_global_uniqueness_scores(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Name", "Score", "program_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns for uniqueness: {sorted(missing)}")
    updated = df.copy()
    updated["Score"] = pd.to_numeric(updated["Score"], errors="coerce")
    updated["program_id"] = pd.to_numeric(updated["program_id"], errors="coerce")
    valid = updated.dropna(subset=["Name", "Score", "program_id"]).copy()
    if valid.empty:
        raise ValueError("No valid rows to compute uniqueness scores.")
    valid["program_id"] = valid["program_id"].astype(int)
    total_programs = valid["program_id"].nunique()
    gene_counts = valid.groupby("Name")["program_id"].nunique().astype(float)
    idf = np.log((total_programs + 1.0) / (gene_counts + 1.0))
    valid["UniquenessScore"] = valid["Score"] * valid["Name"].map(idf)
    updated["UniquenessScore"] = np.nan
    updated.loc[valid.index, "UniquenessScore"] = valid["UniquenessScore"]
    return updated


def _ensure_global_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    if "UniquenessScore" in df.columns and not df["UniquenessScore"].isna().all():
        return df
    logger.info("UniquenessScore missing; computing global uniqueness scores.")
    return _add_global_uniqueness_scores(df)


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = [
    "angiogenesis", "permeability", "barrier", "inflammation", "proliferation",
    "migration", "sprouting", "hypoxia", "metabolism", "junction", "adhesion",
    "leukocyte", "shear", "tip cell", "stalk cell", "arterial", "venous",
    "capillary", "blood-brain barrier", "bbb"
]

INTERACTION_VERBS = [
    "regulates", "induces", "promotes", "inhibits", "suppresses", "activates",
    "binds", "phosphorylates", "modulates", "mediates", "targets", "controls",
    "decreases", "increases", "blocks", "triggers", "catalyzes"
]

MECHANISTIC_VERBS = [
    "regulates", "induces", "promotes", "inhibits", "suppresses", "activates",
    "mediates", "controls", "stimulates", "blocks", "enhances", "reduces",
    "increases", "decreases", "triggers", "phosphorylates", "upregulates",
    "downregulates", "attenuates", "potentiates", "modulates", "drives"
]

CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "ncbi_bioc_cache.json"

PUBTATOR_API_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
PUBTATOR_RATE_LIMIT = 0.1


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

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


def extract_mechanistic_sentences(
    text: str, gene1: str, gene2: str, max_sentences: int = 2
) -> List[str]:
    if not text:
        return []
    sentences = split_text_into_sentences(text)
    mechanistic = []
    gene1_lower = gene1.lower()
    gene2_lower = gene2.lower()
    for sent in sentences:
        sent_lower = sent.lower()
        has_gene1 = gene1_lower in sent_lower
        has_gene2 = gene2_lower in sent_lower
        if has_gene1 and has_gene2:
            for verb in MECHANISTIC_VERBS:
                if verb in sent_lower:
                    mechanistic.append(sent.strip())
                    break
    return mechanistic[:max_sentences]


def extract_any_mechanistic_sentences(
    text: str, gene: str, max_sentences: int = 3
) -> List[str]:
    if not text:
        return []
    sentences = split_text_into_sentences(text)
    mechanistic = []
    gene_lower = gene.lower()
    for sent in sentences:
        sent_lower = sent.lower()
        if gene_lower not in sent_lower:
            continue
        for verb in MECHANISTIC_VERBS:
            if verb in sent_lower:
                mechanistic.append(sent.strip())
                break
    return mechanistic[:max_sentences]


def extract_evidence_sentences(
    abstract: str, title: str, target_genes: Set[str], context_genes: Set[str]
) -> Dict[str, List[str]]:
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
        if any(k in sent_lower for k in DOMAIN_KEYWORDS):
            score += 1
        if any(v in sent_lower for v in INTERACTION_VERBS):
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


# ---------------------------------------------------------------------------
# BioC parsing
# ---------------------------------------------------------------------------

def parse_bioc_abstract(doc: Dict[str, Any]) -> str:
    text_parts = []
    passages = doc.get("passages", [])
    for p in passages:
        inf = p.get("infons", {})
        if inf.get("type") in ("title", "abstract") or inf.get("section_type") in ("TITLE", "ABSTRACT"):
            text_parts.append(p.get("text", ""))
    return " ".join(text_parts)


def parse_bioc_relations(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    rels = doc.get("relations", [])
    for p in doc.get("passages", []):
        rels.extend(p.get("relations", []))
    extracted = []
    for r in rels:
        inf = r.get("infons", {})
        rel_type = inf.get("type")
        if rel_type in ("Gene-Disease", "Chemical-Gene", "Gene-Chemical"):
            extracted.append({"type": rel_type, "id": r.get("id")})
    return extracted


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


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Any]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


# ---------------------------------------------------------------------------
# Gene loading
# ---------------------------------------------------------------------------

def load_program_genes(
    csv_path: Path,
    top_n_loading: int = 20,
    top_n_unique: int = 10,
    top_n_member: int = 100
) -> Dict[int, Dict[str, List[str]]]:
    df = pd.read_csv(csv_path)
    df = _ensure_program_id_column(df)
    df = _ensure_global_uniqueness(df)
    programs = {}
    for pid, group in df.groupby("program_id"):
        sorted_loading = group.sort_values("Score", ascending=False)["Name"].astype(str).tolist()
        top_loading_genes = sorted_loading[:top_n_loading]
        top_loading_set = set(top_loading_genes)
        top_unique_genes = []
        if "UniquenessScore" in group.columns:
            sorted_unique = group.sort_values("UniquenessScore", ascending=False)["Name"].astype(str).tolist()
            for gene in sorted_unique:
                if gene not in top_loading_set:
                    top_unique_genes.append(gene)
                    if len(top_unique_genes) >= top_n_unique:
                        break
        drivers = top_loading_genes + top_unique_genes
        members = sorted_loading[:top_n_member]
        all_genes = sorted_loading
        programs[int(pid)] = {
            "drivers": drivers,
            "top_loading": top_loading_genes,
            "top_unique": top_unique_genes,
            "members": members,
            "all_genes": all_genes
        }
    return programs


# ---------------------------------------------------------------------------
# Gene summaries
# ---------------------------------------------------------------------------

def resolve_gene_summaries(
    source: str,
    programs: Dict[int, Dict[str, List[str]]],
    program_ids: List[int],
    ncbi_client: NcbiClient,
    harmonizome_client: Optional[HarmonizomeClient] = None,
    use_full_summaries: bool = False,
) -> Dict[int, Dict[str, str]]:
    program_gene_summaries: Dict[int, Dict[str, str]] = {}
    all_drivers: Set[str] = set()
    for pid in program_ids:
        all_drivers.update(programs[pid]["drivers"])
    if not all_drivers:
        return {pid: {} for pid in program_ids}
    if source == "ncbi":
        logger.info("Resolving Entrez IDs for %d unique driver genes...", len(all_drivers))
        symbol_to_id = ncbi_client.normalize_genes(list(all_drivers))
        valid_ids = sorted({gid for gid in symbol_to_id.values() if gid})
        logger.info("Fetching summaries for %d gene IDs...", len(valid_ids))
        id_to_summary = ncbi_client.get_gene_summaries(valid_ids)
        for pid in program_ids:
            drivers = programs[pid]["drivers"]
            summaries: Dict[str, str] = {}
            for sym in drivers:
                gid = symbol_to_id.get(sym)
                if gid:
                    text = id_to_summary.get(gid)
                    if text:
                        summaries[sym] = text
            program_gene_summaries[pid] = summaries
        return program_gene_summaries
    if source == "harmonizome":
        harmonizome_client = harmonizome_client or HarmonizomeClient(
            use_full_summaries=use_full_summaries
        )
        summary_type = "full HTML" if use_full_summaries else "short API"
        logger.info("Fetching Harmonizome %s summaries for %d unique driver genes...",
                     summary_type, len(all_drivers))
        symbol_to_summary = harmonizome_client.get_gene_summaries(sorted(all_drivers))
        for pid in program_ids:
            drivers = programs[pid]["drivers"]
            summaries = {sym: symbol_to_summary[sym] for sym in drivers if sym in symbol_to_summary}
            program_gene_summaries[pid] = summaries
        return program_gene_summaries
    raise ValueError(f"Unsupported gene summary source: {source}")


# ---------------------------------------------------------------------------
# Regulator loading
# ---------------------------------------------------------------------------

def load_regulator_data(
    csv_path: Path,
    significance_threshold: float = 0.05,
) -> Dict[int, pd.DataFrame]:
    if not csv_path or not csv_path.exists():
        logger.warning(f"Regulator file not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    df = standardize_regulator_results(df, significance_threshold=significance_threshold)
    df = df[df["significant"] == True].copy()
    result = {}
    for pid, group in df.groupby("program_id"):
        keep_cols = ["grna_target", "log_2_fold_change", "p_value", "significant"]
        if "adj_p_value" in group.columns:
            keep_cols.append("adj_p_value")
        result[int(pid)] = group[keep_cols].copy()
    logger.info(f"Loaded regulators for {len(result)} programs")
    return result


def get_top_regulators(
    regulator_data: Dict[int, pd.DataFrame],
    program_id: int,
    top_n: int = 3,
    top_n_positive: Optional[int] = None,
    top_n_negative: Optional[int] = None,
    use_all_significant: bool = False,
    max_regulators: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    reg_df = regulator_data.get(program_id)
    if reg_df is None or len(reg_df) == 0:
        return {'positive': [], 'negative': []}
    top_n_positive = top_n if top_n_positive is None else top_n_positive
    top_n_negative = top_n if top_n_negative is None else top_n_negative
    sig_df = reg_df[reg_df['significant'] == True].copy()
    sorted_df = sig_df.sort_values(by="log_2_fold_change")
    if use_all_significant:
        positive = sorted_df[sorted_df["log_2_fold_change"] < 0].head(max_regulators)
        negative = sorted_df[sorted_df["log_2_fold_change"] > 0].tail(max_regulators).iloc[::-1]
    else:
        positive = sorted_df[sorted_df["log_2_fold_change"] < 0].head(top_n_positive)
        negative = sorted_df[sorted_df["log_2_fold_change"] > 0].tail(top_n_negative).iloc[::-1]

    def extract_regulator(row):
        result = {'gene': row['grna_target'], 'log2fc': row['log_2_fold_change']}
        if 'p_value' in row:
            result['pvalue'] = row['p_value']
        return result

    return {
        'positive': [extract_regulator(row) for _, row in positive.iterrows()],
        'negative': [extract_regulator(row) for _, row in negative.iterrows()]
    }


# ---------------------------------------------------------------------------
# PubTator3 search and BioC fetch
# ---------------------------------------------------------------------------

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
            gene_mentions: Dict[str, Set[str]] = {}
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


# ---------------------------------------------------------------------------
# Regulator validation
# ---------------------------------------------------------------------------

def validate_regulator_with_string(
    regulator: str, program_genes: List[str],
    species: int = 10090, required_score: int = 400
) -> Dict[str, Any]:
    time.sleep(1.0)
    result = get_regulator_program_interactions(
        regulator=regulator, program_genes=program_genes,
        species=species, required_score=required_score, top_n=10
    )
    return {
        'regulator': regulator,
        'n_program_targets': result['n_interactions'],
        'string_interactions': result['interactions']
    }


def validate_regulator_program(
    regulator: str, program_genes: List[str],
    keyword: str = "endothelial OR vascular",
    max_pmids: int = 50, min_relation_score: float = 0.5
) -> Dict[str, Any]:
    genes_or = " OR ".join(program_genes[:10])
    query = f"({regulator}) AND ({genes_or}) AND ({keyword})"
    logger.info(f"  Validating {regulator}: {query[:80]}...")
    time.sleep(PUBTATOR_RATE_LIMIT)
    search_results = search_pubtator(query, max_results=60)
    if not search_results:
        return {'regulator': regulator, 'papers_found': 0,
                'papers_with_relations': 0, 'mechanistic_relations': [], 'top_papers': []}
    pmids = [r['pmid'] for r in search_results[:max_pmids] if r['pmid']]
    time.sleep(PUBTATOR_RATE_LIMIT)
    pmid_data = fetch_bioc_relations_with_text(pmids)
    mechanistic_relations = []
    regulator_upper = regulator.upper()
    program_genes_upper = {g.upper() for g in program_genes}
    seen_gene_pmid = set()
    pmids_with_relations = set()
    for pmid, data in pmid_data.items():
        rels = data.get('relations', [])
        abstract = data.get('abstract', '')
        title = data.get('title', '')
        for rel in rels:
            if rel['score'] < min_relation_score:
                continue
            g1 = rel['gene1'].upper()
            g2 = rel['gene2'].upper()
            rel_type = rel['type']
            if rel_type not in ('Positive_Correlation', 'Negative_Correlation'):
                continue
            if regulator_upper not in (g1, g2):
                continue
            pmids_with_relations.add(pmid)
            other_gene = rel['gene2'] if g1 == regulator_upper else rel['gene1']
            dedup_key = (other_gene.upper(), pmid)
            if dedup_key in seen_gene_pmid:
                continue
            seen_gene_pmid.add(dedup_key)
            sentences = extract_mechanistic_sentences(abstract, regulator, other_gene, max_sentences=2)
            if not sentences:
                sentences = extract_any_mechanistic_sentences(abstract, regulator, max_sentences=1)
            mechanistic_relations.append({
                'pmid': pmid, 'target_gene': other_gene, 'relation_type': rel_type,
                'score': rel['score'], 'title': title, 'sentences': sentences
            })
    if not mechanistic_relations:
        for pmid, data in pmid_data.items():
            abstract = data.get('abstract', '')
            title = data.get('title', '')
            gene_mentions = data.get('gene_mentions', {})
            if not abstract:
                continue
            regulator_aliases = gene_mentions.get(regulator.upper(), [regulator])
            if not regulator_aliases:
                regulator_aliases = [regulator]
            sentences = []
            for alias in regulator_aliases:
                sentences = extract_any_mechanistic_sentences(abstract, alias, max_sentences=2)
                if sentences:
                    break
            if not sentences:
                sentences = extract_any_mechanistic_sentences(abstract, regulator, max_sentences=2)
            if sentences:
                mechanistic_relations.append({
                    'pmid': pmid, 'target_gene': 'unknown', 'relation_type': 'Mentioned',
                    'score': 0.5, 'title': title, 'sentences': sentences
                })
    mechanistic_relations.sort(key=lambda x: -x['score'])
    mechanistic_relations = mechanistic_relations[:10]
    top_papers = [{'pmid': r['pmid'], 'title': r['title']} for r in search_results[:10]]
    papers_with_rels = len([p for p, d in pmid_data.items() if d.get('relations')])
    return {
        'regulator': regulator, 'papers_found': len(search_results),
        'papers_with_relations': papers_with_rels,
        'mechanistic_relations': mechanistic_relations, 'top_papers': top_papers
    }


def validate_program_regulators(
    program_id: int, regulator_data: Dict[int, pd.DataFrame],
    program_genes: List[str], keyword: str = "endothelial OR vascular",
    top_n_regulators: int = 3, top_n_positive_regulators: Optional[int] = None,
    top_n_negative_regulators: Optional[int] = None,
    max_pmids_per_regulator: int = 50, use_string: bool = True,
    use_all_significant: bool = False, max_regulators: int = 20,
    use_batch: bool = True, min_score: int = 400
) -> Dict[str, Any]:
    top_regs = get_top_regulators(
        regulator_data, program_id, top_n=top_n_regulators,
        top_n_positive=top_n_positive_regulators,
        top_n_negative=top_n_negative_regulators,
        use_all_significant=use_all_significant, max_regulators=max_regulators
    )
    result: Dict[str, Any] = {'positive_regulators': [], 'negative_regulators': []}
    all_regulator_genes = [r['gene'] for r in top_regs['positive']] + [r['gene'] for r in top_regs['negative']]

    if use_string and use_batch and all_regulator_genes:
        batch_results = batch_validate_regulators(
            regulator_genes=all_regulator_genes,
            program_genes=program_genes,
            required_score=min_score
        )
        for reg_info in top_regs['positive']:
            gene = reg_info['gene']
            interactions = batch_results.get(gene, [])
            result['positive_regulators'].append({
                'regulator': gene, 'log2fc': reg_info['log2fc'],
                'n_program_targets': len(interactions),
                'string_interactions': interactions
            })
        for reg_info in top_regs['negative']:
            gene = reg_info['gene']
            interactions = batch_results.get(gene, [])
            result['negative_regulators'].append({
                'regulator': gene, 'log2fc': reg_info['log2fc'],
                'n_program_targets': len(interactions),
                'string_interactions': interactions
            })
        return result

    for reg_info in top_regs['positive']:
        if use_string:
            logger.info(f"  Validating activator {reg_info['gene']} with STRING...")
            validation = validate_regulator_with_string(
                regulator=reg_info['gene'], program_genes=program_genes)
        else:
            validation = validate_regulator_program(
                regulator=reg_info['gene'], program_genes=program_genes,
                keyword=keyword, max_pmids=max_pmids_per_regulator)
        validation['log2fc'] = reg_info['log2fc']
        result['positive_regulators'].append(validation)

    for reg_info in top_regs['negative']:
        if use_string:
            logger.info(f"  Validating repressor {reg_info['gene']} with STRING...")
            validation = validate_regulator_with_string(
                regulator=reg_info['gene'], program_genes=program_genes)
        else:
            validation = validate_regulator_program(
                regulator=reg_info['gene'], program_genes=program_genes,
                keyword=keyword, max_pmids=max_pmids_per_regulator)
        validation['log2fc'] = reg_info['log2fc']
        result['negative_regulators'].append(validation)

    return result
