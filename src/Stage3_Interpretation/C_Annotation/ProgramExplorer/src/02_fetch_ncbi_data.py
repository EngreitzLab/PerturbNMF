"""
@description
Program-level literature retrieval and evidence gathering.
Implements "Search Narrow, Verify Broad" strategy to find high-confidence
papers connecting gene programs to specific diseases/functions.

Process:
1. INPUT: Loading Matrix (Genes/Scores per Program)
2. SEARCH: Query Pubtator 3 with "Driver Genes" (Top 5-10) + Keyword.
   Query = `(Driver1 OR ... OR DriverN) AND (endothelial OR ...)`
3. FETCH: Retrieve BioC-JSON annotations for found PMIDs (with caching).
4. VERIFY: Score papers by checking mentions of "Member Genes" (Top 50).
   Score = Unique Member Genes mentioned in Abstract.
   Filter: Keep papers with Score >= 2.
5. EXTRACT: specific Gene-Disease associations from high-scoring papers.
6. OUTPUT: JSON (for LLM context) + CSV (for user verification).

@dependencies
- ncbi_api
- pandas
- BioC-JSON parsing logic

@examples
python pipeline/02_fetch_ncbi_data.py \
    --input input/genes/FB_moi15_seq2_loading_gene_k100_top300.csv \
    --keyword "endothelial OR endothelium" \
    --csv-out results/output/ncbi_context.csv \
    --json-out results/output/ncbi_context.json \
    --api-key "$NCBI_API_KEY"
"""

import re
import argparse
import json
import logging
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from collections import Counter

# Add this directory to sys.path for sibling imports
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from ncbi_api import NcbiClient
from harmonizome_api import HarmonizomeClient
from string_api import (
    get_regulator_program_interactions,
    batch_validate_regulators,
)
from column_mapper import standardize_regulator_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

"""
@description
Configuration loader for the literature retrieval step.
It is responsible for reading JSON/YAML configs and applying per-step defaults
with CLI override precedence and optional test mode.

Key features:
- Supports JSON and YAML (if PyYAML is installed).
- Applies test mode topic filters when configured.

@dependencies
- json: Built-in JSON parser
- yaml (optional): YAML parser when available
- sys: CLI inspection for override detection
"""


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("PyYAML is required for YAML configs.") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise SystemExit("Config must be a mapping at the top level.")
    return data


def get_cli_overrides(argv: List[str]) -> Set[str]:
    overrides: Set[str] = set()
    for token in argv:
        if token.startswith("--"):
            name = token[2:]
            if "=" in name:
                name = name.split("=", 1)[0]
            overrides.add(name.replace("-", "_"))
    return overrides


def parse_topics_value(value: Optional[object]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(t.strip()) for t in value.split(",") if t.strip()]
    return None


def apply_test_mode(
    args: argparse.Namespace, config: Dict[str, Any], cli_overrides: Set[str]
) -> argparse.Namespace:
    test_cfg = config.get("test", {}) if isinstance(config.get("test", {}), dict) else {}
    enabled = bool(test_cfg.get("enabled") or config.get("test_mode"))
    if not enabled:
        return args

    topics = test_cfg.get("topics") or test_cfg.get("programs") or config.get("test_programs")
    if hasattr(args, "topics") and "topics" not in cli_overrides and not getattr(args, "topics", None):
        if topics is not None:
            args.topics = topics

    num_programs = test_cfg.get("num_programs") or config.get("test_num_programs")
    if hasattr(args, "num_programs") and "num_programs" not in cli_overrides and not getattr(args, "num_programs", None):
        if num_programs is not None:
            args.num_programs = num_programs
    return args


def apply_config_overrides(
    args: argparse.Namespace, config: Dict[str, Any], cli_overrides: Set[str]
) -> argparse.Namespace:
    steps_cfg = config.get("steps", {}) if isinstance(config.get("steps", {}), dict) else {}
    step_cfg = steps_cfg.get("literature_fetch", {})
    if not isinstance(step_cfg, dict):
        return args

    for key, value in step_cfg.items():
        dest = key.replace("-", "_")
        if dest in cli_overrides:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)
    return args

# Global uniqueness score helpers
"""
@description
This component ensures global gene uniqueness scores are available for pipeline
inputs. It normalizes program identifiers and computes TF-IDF-style uniqueness
when the column is missing.

Key features:
- Accepts RowID or program_id inputs.
- Computes UniquenessScore using a global IDF across all programs.

@dependencies
- numpy: IDF calculation
- pandas: DataFrame manipulation

@examples
- df = ensure_program_id_column(df)
- df = ensure_global_uniqueness(df, logger)
"""


def ensure_program_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "program_id" in df.columns:
        return df
    if "RowID" not in df.columns:
        raise ValueError("CSV must have 'program_id' or 'RowID'")
    updated = df.copy()
    updated["program_id"] = updated["RowID"]
    return updated


def add_global_uniqueness_scores(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Name", "Score", "program_id"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_sorted = sorted(missing)
        raise ValueError(
            f"CSV missing required columns for uniqueness: {missing_sorted}"
        )

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


def ensure_global_uniqueness(
    df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    if "UniquenessScore" in df.columns and not df["UniquenessScore"].isna().all():
        return df
    logger.info("UniquenessScore missing; computing global uniqueness scores.")
    return add_global_uniqueness_scores(df)

# Domain Constants for Programmatic Scoring
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

# Mechanistic verbs for sentence extraction (subset focused on causality)
MECHANISTIC_VERBS = [
    "regulates", "induces", "promotes", "inhibits", "suppresses", "activates",
    "mediates", "controls", "stimulates", "blocks", "enhances", "reduces",
    "increases", "decreases", "triggers", "phosphorylates", "upregulates",
    "downregulates", "attenuates", "potentiates", "modulates", "drives"
]


def extract_mechanistic_sentences(
    text: str,
    gene1: str,
    gene2: str,
    max_sentences: int = 2
) -> List[str]:
    """Extract sentences mentioning both genes with mechanistic verbs.
    
    Args:
        text: Abstract or full text to search
        gene1: First gene symbol (e.g., regulator)
        gene2: Second gene symbol (e.g., target)
        max_sentences: Maximum sentences to return
    
    Returns:
        List of full sentences containing both genes and a mechanistic verb
    """
    if not text:
        return []
    
    sentences = split_text_into_sentences(text)
    mechanistic = []
    
    gene1_lower = gene1.lower()
    gene2_lower = gene2.lower()
    
    for sent in sentences:
        sent_lower = sent.lower()
        # Check if both genes mentioned (or gene name variants)
        has_gene1 = gene1_lower in sent_lower
        has_gene2 = gene2_lower in sent_lower
        
        if has_gene1 and has_gene2:
            # Check for mechanistic verb
            for verb in MECHANISTIC_VERBS:
                if verb in sent_lower:
                    mechanistic.append(sent.strip())
                    break
    
    return mechanistic[:max_sentences]


def extract_any_mechanistic_sentences(
    text: str,
    gene: str,
    max_sentences: int = 3
) -> List[str]:
    """Extract sentences mentioning a gene with mechanistic verbs.
    
    Less strict than extract_mechanistic_sentences - finds any sentence
    where the gene appears with a mechanistic verb, regardless of second gene.
    
    Args:
        text: Abstract or full text to search
        gene: Gene symbol to search for
        max_sentences: Maximum sentences to return
    
    Returns:
        List of full sentences containing the gene and a mechanistic verb
    """
    if not text:
        return []
    
    sentences = split_text_into_sentences(text)
    mechanistic = []
    
    gene_lower = gene.lower()
    
    for sent in sentences:
        sent_lower = sent.lower()
        if gene_lower not in sent_lower:
            continue
        
        # Check for mechanistic verb
        for verb in MECHANISTIC_VERBS:
            if verb in sent_lower:
                mechanistic.append(sent.strip())
                break
    
    return mechanistic[:max_sentences]

def split_text_into_sentences(text: str) -> List[str]:
    """
    Robustly split abstract into sentences using Regex.
    Handles common abbreviations (e.g., 'Fig.', 'al.', 'e.g.') to avoid false splits.
    """
    if not text:
        return []
        
    # Simplified robust splitter for biomedical text:
    # 1. Protect common abbreviations by replacing temporarily
    protected = text
    subs = {
        "et al.": "ET_AL_MARKER",
        "e.g.": "EG_MARKER",
        "i.e.": "IE_MARKER",
        "Fig.": "FIG_MARKER",
        "Ref.": "REF_MARKER",
        "vs.": "VS_MARKER"
    }
    for k, v in subs.items():
        protected = protected.replace(k, v)
        
    # 2. Split by [.?!] followed by space or end of string
    parts = re.split(r'(?<=[.?!])\s+', protected)
    
    # 3. Restore and clean
    sentences = []
    for p in parts:
        restored = p
        for k, v in subs.items():
            restored = restored.replace(v, k)
        s = restored.strip()
        if len(s) > 10: # Filter noise
            sentences.append(s)
            
    return sentences

def extract_evidence_sentences(
    abstract: str, 
    title: str,
    target_genes: Set[str], 
    context_genes: Set[str]
) -> Dict[str, List[str]]:
    """
    Extracts high-quality sentences mentioning target genes.
    Returns: {gene: [sentence1, sentence2]}
    """
    full_text = f"{title}. {abstract}"
    sentences = split_text_into_sentences(full_text)
    
    gene_to_sentences = {g: [] for g in target_genes}
    
    for sent in sentences:
        sent_lower = sent.lower()
        
        # Check for gene mentions
        found_targets = []
        for g in target_genes:
            if re.search(rf"\b{re.escape(g)}\b", sent, re.IGNORECASE):
                found_targets.append(g)
        
            
        # Deduplication check (robust)
        normalized_sent = sent.strip(" .")
        existing_sents = [s[1].strip(" .") for s in [item for sublist in gene_to_sentences.values() for item in sublist]]
        if normalized_sent in existing_sents:
            # logger.info(f"DEBUG: Skipping duplicate: {normalized_sent[:20]}...")
            continue

        # Score Sentence
        score = 0
        
        # +2 for Co-occurrence with OTHER context genes (members)
        for m in context_genes:
            if m not in found_targets and re.search(rf"\b{re.escape(m)}\b", sent, re.IGNORECASE):
                score += 2
        
        # +1 for Domain Keywords
        if any(k in sent_lower for k in DOMAIN_KEYWORDS):
            score += 1
            
        # +1 for Interaction Verbs
        if any(v in sent_lower for v in INTERACTION_VERBS):
            score += 1
            
        # Assign sentence to all found targets
        for g in found_targets:
            if sent not in [s[1] for s in gene_to_sentences[g]]:
                gene_to_sentences[g].append((score, sent))
                
    # Sort and filter top K per gene
    final_map = {}
    for g, items in gene_to_sentences.items():
        if not items:
            continue
        items.sort(key=lambda x: x[0], reverse=True)
        # Keep top 2 unique sentences
        final_map[g] = [x[1] for x in items[:2]]
        
    return final_map

CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "ncbi_bioc_cache.json"

def load_program_genes(
    csv_path: Path, 
    top_n_loading: int = 20, 
    top_n_unique: int = 10,
    top_n_member: int = 100
) -> Dict[int, Dict[str, List[str]]]:
    """
    Load driver and member genes per program.
    Returns: {program_id: {"drivers": [...], "top_loading": [...], "top_unique": [...], "members": [...], "all_genes": [...]}}
    
    Drivers: Top N Loading + Top M Unique (non-overlapping) = exactly N + M genes
    Members (Verification): Top 100 Loading
    """
    df = pd.read_csv(csv_path)
    # Expect columns: program_id (or RowID), Name, Score
    df = ensure_program_id_column(df)
    df = ensure_global_uniqueness(df, logger)

    programs = {}
    for pid, group in df.groupby("program_id"):
        # 1. Top Loading (by Score)
        sorted_loading = group.sort_values("Score", ascending=False)["Name"].astype(str).tolist()
        top_loading_genes = sorted_loading[:top_n_loading]
        top_loading_set = set(top_loading_genes)
        
        # 2. Top Unique (by UniquenessScore) - NON-OVERLAPPING with top loading
        top_unique_genes = []
        if "UniquenessScore" in group.columns:
            sorted_unique = group.sort_values("UniquenessScore", ascending=False)["Name"].astype(str).tolist()
            # Filter out genes already in top_loading, then take top N
            for gene in sorted_unique:
                if gene not in top_loading_set:
                    top_unique_genes.append(gene)
                    if len(top_unique_genes) >= top_n_unique:
                        break
            
        # Drivers: exactly top_n_loading + top_n_unique genes (no overlap)
        drivers = top_loading_genes + top_unique_genes
        
        # Members: Top 100 Loading
        members = sorted_loading[:top_n_member]
        
        # All genes: Full list for STRING validation (typically 300)
        all_genes = sorted_loading
        
        programs[int(pid)] = {
            "drivers": drivers,
            "top_loading": top_loading_genes,
            "top_unique": top_unique_genes,
            "members": members,
            "all_genes": all_genes  # For STRING regulator validation
        }
    return programs


def resolve_gene_summaries(
    source: str,
    programs: Dict[int, Dict[str, List[str]]],
    program_ids: List[int],
    ncbi_client: NcbiClient,
    harmonizome_client: Optional[HarmonizomeClient] = None,
    use_full_summaries: bool = False,
) -> Dict[int, Dict[str, str]]:
    """
    @description
    This component handles retrieval of gene summaries for driver genes.
    It is responsible for selecting the configured source and mapping
    summaries back to per-program driver lists.

    Key features:
    - Supports NCBI Entrez summaries or Harmonizome gene descriptions.
    - Returns per-program summary dictionaries keyed by gene symbol.

    @dependencies
    - NcbiClient: Used for NCBI gene summary retrieval
    - HarmonizomeClient: Used for Harmonizome gene description retrieval

    @examples
    - resolve_gene_summaries("ncbi", programs, program_ids, ncbi_client)
    - resolve_gene_summaries("harmonizome", programs, program_ids, ncbi_client)
    """
    program_gene_summaries: Dict[int, Dict[str, str]] = {}
    all_drivers: Set[str] = set()
    for pid in program_ids:
        all_drivers.update(programs[pid]["drivers"])

    if not all_drivers:
        return {pid: {} for pid in program_ids}

    if source == "ncbi":
        logger.info(
            "Resolving Entrez IDs for %d unique driver genes...",
            len(all_drivers),
        )
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
        logger.info(
            "Fetching Harmonizome %s summaries for %d unique driver genes...",
            summary_type,
            len(all_drivers),
        )
        symbol_to_summary = harmonizome_client.get_gene_summaries(
            sorted(all_drivers)
        )
        for pid in program_ids:
            drivers = programs[pid]["drivers"]
            summaries = {
                sym: symbol_to_summary[sym]
                for sym in drivers
                if sym in symbol_to_summary
            }
            program_gene_summaries[pid] = summaries
        return program_gene_summaries

    raise ValueError(f"Unsupported gene summary source: {source}")


def load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_cache(cache: Dict[str, Any]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

def parse_bioc_abstract(doc: Dict[str, Any]) -> str:
    """Extract abstract text from BioC document."""
    text_parts = []
    passages = doc.get("passages", [])
    for p in passages:
        inf = p.get("infons", {})
        # Abstract or Title
        if inf.get("type") in ("title", "abstract") or inf.get("section_type") in ("TITLE", "ABSTRACT"):
             text_parts.append(p.get("text", ""))
    return " ".join(text_parts)

def parse_bioc_relations(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract relation annotations if available (Pubtator 3 exports them)."""
    # Pubtator 3 JSON usually has "annotations" and "relations" keys in passages or doc level
    # Actually Pubtator provides entities in 'annotations' and relations in 'relations'.
    # Checking doc level 'relations'
    rels = doc.get("relations", [])
    # Also check passages level
    for p in doc.get("passages", []):
        rels.extend(p.get("relations", []))
    
    extracted = []
    for r in rels:
        # Pubtator relations often: type="Chemical-Gene", etc.
        # infons: {type: ..., relationship: ...}
        # We need to map node IDs to text? The nodes are usually annotation IDs.
        # This is complex to parse perfectly without building an ID map.
        # Simplified: Just grab the 'type' and maybe 'infons'
        inf = r.get("infons", {})
        rel_type = inf.get("type")
        if rel_type in ("Gene-Disease", "Chemical-Gene", "Gene-Chemical"):
             extracted.append({"type": rel_type, "id": r.get("id")})
    return extracted

def find_gene_mentions(doc: Dict[str, Any], member_genes: List[str]) -> List[str]:
    """
    Check which member genes are mentioned in the doc.
    Uses 'annotations' from Pubtator if available, else naive text search.
    Pubtator 'annotations' have 'type': 'Gene' and 'text'.
    Retrieving Normalized Gene IDs is better, but here we work with Symbols in member_genes.
    We will try to match Annotation Text or Normalization ID?
    The user-provided `member_genes` are Symbols.
    Matching Symbols in text is risky (e.g. "MET").
    Trusting Pubtator 'annotations' is safer.
    """
    found = set()
    # Collect all gene annotations from doc
    doc_genes = [] 
    
    # 1. Collect annotated genes from Pubtator
    for p in doc.get("passages", []):
        for ann in p.get("annotations", []):
            inf = ann.get("infons", {})
            if inf.get("type") == "Gene":
                # Text mention
                txt = ann.get("text", "")
                if txt: doc_genes.append(txt)
                # Normalized ID
                # ident = inf.get("identifier") 
                
    # 2. Case-insensitive match against member_genes
    # Simple set intersection
    members_lower = {g.lower() for g in member_genes}
    for dg in doc_genes:
        if dg.lower() in members_lower:
            # Map back to original symbol casing? exact match logic tricky
            # Just add the doc_gene string or try to find which member it matched
            # Optimization: Pre-map lower -> original
            pass

    # Fallback to Text Search? 
    # If we requested BioC-XML/JSON from Pubtator, it SHOULD have annotations.
    # Text search is simpler for now given complex normalization.
    full_text = parse_bioc_abstract(doc).lower()
    for g in member_genes:
        # Naive boundary check
        # e.g " " + g + " " -> simplistic
        # Let's rely on string containment for MVP, but acknowledge risk.
        # Better: if g in doc_genes list.
        
        # Let's assume Pubtator annotations are the gold standard.
        # We check if any annotation matches our gene symbol (case-insensitive)
        if any(g.lower() == dg.lower() for dg in doc_genes):
            found.add(g)
    
    return list(found)


# =============================================================================
# Regulator Validation via PubTator3
# =============================================================================

import requests

PUBTATOR_API_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
PUBTATOR_RATE_LIMIT = 0.1  # 10 requests per second


def load_regulator_data(
    csv_path: Path,
    significance_threshold: float = 0.05,
) -> Dict[int, pd.DataFrame]:
    """Load significant regulators from SCEPTRE results CSV.
    
    Returns dict mapping program_id -> DataFrame with columns:
    [grna_target, log_2_fold_change, p_value]
    """
    if not csv_path or not csv_path.exists():
        logger.warning(f"Regulator file not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    df = standardize_regulator_results(
        df, significance_threshold=significance_threshold
    )
    df = df[df["significant"] == True].copy()  # type: ignore
    
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
    """Get top positive and negative regulators for a program.
    
    Args:
        regulator_data: Dict mapping program_id to DataFrame
        program_id: Program to get regulators for
        top_n: Backward-compatible default for both directions
        top_n_positive: Number of positive regulators to return
        top_n_negative: Number of negative regulators to return
        use_all_significant: If True, return all significant regulators (up to max_regulators)
        max_regulators: Maximum regulators per category when using all significant
    
    Returns: {
        'positive': [{'gene': 'Fzd4', 'log2fc': -0.62, 'pvalue': 1e-10}, ...],  # negative log2FC = activator
        'negative': [{'gene': 'Eng', 'log2fc': 0.49, 'pvalue': 1e-8}, ...]     # positive log2FC = repressor
    }
    """
    reg_df = regulator_data.get(program_id)
    if reg_df is None or len(reg_df) == 0:
        return {'positive': [], 'negative': []}
    
    top_n_positive = top_n if top_n_positive is None else top_n_positive
    top_n_negative = top_n if top_n_negative is None else top_n_negative
    sig_df = reg_df[reg_df['significant'] == True].copy()
    
    sorted_df = sig_df.sort_values(by="log_2_fold_change")
    
    # Negative log2FC = positive regulators (activators)
    if use_all_significant:
        positive = sorted_df[sorted_df["log_2_fold_change"] < 0].head(max_regulators)
        negative = sorted_df[sorted_df["log_2_fold_change"] > 0].tail(max_regulators).iloc[::-1]
    else:
        positive = sorted_df[sorted_df["log_2_fold_change"] < 0].head(top_n_positive)
        negative = sorted_df[sorted_df["log_2_fold_change"] > 0].tail(top_n_negative).iloc[::-1]
    
    def extract_regulator(row):
        result = {
            'gene': row['grna_target'], 
            'log2fc': row['log_2_fold_change']
        }
        if 'p_value' in row:
            result['pvalue'] = row['p_value']
        return result
    
    return {
        'positive': [extract_regulator(row) for _, row in positive.iterrows()],
        'negative': [extract_regulator(row) for _, row in negative.iterrows()]
    }


def validate_regulator_with_string(
    regulator: str,
    program_genes: List[str],
    species: int = 10090,  # mouse
    required_score: int = 400
) -> Dict[str, Any]:
    """Validate regulator-program relationship via STRING-DB.
    
    Queries STRING for direct protein-protein interactions between the
    regulator and program genes. Returns only interactions with genes
    that are actually in the program.
    
    Args:
        regulator: Gene symbol of the regulator
        program_genes: List of program's gene symbols
        species: NCBI taxon ID (10090 for mouse)
        required_score: Minimum STRING combined score (0-1000)
    
    Returns: {
        'regulator': 'Fzd4',
        'n_program_targets': 5,
        'string_interactions': [
            {'target_gene': 'Wnt7a', 'score': 989, 'experimental_score': 237, ...},
            ...
        ]
    }
    """
    import time
    time.sleep(1.0)  # Rate limiting
    
    result = get_regulator_program_interactions(
        regulator=regulator,
        program_genes=program_genes,
        species=species,
        required_score=required_score,
        top_n=10
    )
    
    return {
        'regulator': regulator,
        'n_program_targets': result['n_interactions'],
        'string_interactions': result['interactions']
    }


def search_pubtator(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Search PubTator3 for papers matching query.
    
    Returns list of {pmid, title, score, text_hl} dicts.
    
    Note: PubTator3 uses 'size' parameter for page size (max ~100 per request).
    """
    url = f"{PUBTATOR_API_BASE}/search/"
    # Use 'size' parameter for page size (not 'page_size' or 'limit')
    params = {"text": query, "page": 1, "size": min(max_results, 100)}
    
    try:
        # Increased timeout to 120s for complex gene queries
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code != 200:
            logger.warning(f"PubTator search failed: {resp.status_code}")
            return []
        
        data = resp.json()
        results = data.get("results", [])[:max_results]
        
        return [{
            'pmid': r.get('pmid'),
            'title': r.get('title', ''),
            'score': r.get('score', 0),
            'text_hl': r.get('text_hl', '')
        } for r in results]
        
    except Exception as e:
        logger.error(f"PubTator search error: {e}")
        return []


def fetch_bioc_relations_with_text(pmids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetch BioC-JSON and extract gene-gene relations plus abstract text.
    
    Returns: {pmid: {
        'relations': [{'gene1': 'FZD4', 'gene2': 'CTNNB1', 'type': 'Positive_Correlation', 'score': 0.99}, ...],
        'title': str,
        'abstract': str,
        'gene_mentions': {'NCOA1': ['SRC-1', 'NCOA1'], ...}  # gene name -> text mentions
    }}
    """
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
        
        # Handle different response formats
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
            
            # Extract title and abstract from passages, plus gene annotations
            title = ''
            abstract = ''
            gene_mentions: Dict[str, Set[str]] = {}  # normalized name -> set of text mentions
            
            for passage in doc.get('passages', []):
                ptype = passage.get('infons', {}).get('type', '')
                if ptype == 'title':
                    title = passage.get('text', '')
                elif ptype == 'abstract':
                    abstract = passage.get('text', '')
                
                # Extract gene annotations to build alias mapping
                for ann in passage.get('annotations', []):
                    if ann.get('infons', {}).get('type') == 'Gene':
                        gene_name = ann.get('infons', {}).get('name', '')
                        text_mention = ann.get('text', '')
                        if gene_name and text_mention:
                            if gene_name.upper() not in gene_mentions:
                                gene_mentions[gene_name.upper()] = set()
                            gene_mentions[gene_name.upper()].add(text_mention)
            
            # Extract relations
            relations = []
            for rel in doc.get('relations', []):
                infons = rel.get('infons', {})
                r1 = infons.get('role1', {})
                r2 = infons.get('role2', {})
                rel_type = infons.get('type', 'Unknown')
                score = float(infons.get('score', 0))
                
                # Only gene-gene relations with mechanistic types
                if r1.get('type') == 'Gene' and r2.get('type') == 'Gene':
                    if rel_type in ('Positive_Correlation', 'Negative_Correlation'):
                        relations.append({
                            'gene1': r1.get('name', ''),
                            'gene2': r2.get('name', ''),
                            'type': rel_type,
                            'score': score
                        })
            
            # Convert sets to lists for JSON serialization
            gene_mentions_list = {k: list(v) for k, v in gene_mentions.items()}
            
            result[int(pmid)] = {
                'relations': relations,
                'title': title,
                'abstract': abstract,
                'gene_mentions': gene_mentions_list
            }
        
        return result
        
    except Exception as e:
        logger.error(f"PubTator BioC fetch error: {e}")
        return {}


def validate_regulator_program(
    regulator: str,
    program_genes: List[str],
    keyword: str = "endothelial OR vascular",
    max_pmids: int = 50,
    min_relation_score: float = 0.5
) -> Dict[str, Any]:
    """Validate regulator-program relationship via PubTator3.
    
    Extracts mechanistic gene-gene relations (Positive/Negative_Correlation only)
    and the full sentences describing those relationships from abstracts.
    
    Args:
        regulator: Gene symbol of the regulator
        program_genes: List of program's top genes
        keyword: Keyword terms for search
        max_pmids: Number of PMIDs to fetch for relation extraction (default 50)
        min_relation_score: Minimum score for relations
    
    Returns: {
        'regulator': 'Fzd4',
        'papers_found': 20,
        'papers_with_relations': 8,
        'mechanistic_relations': [
            {
                'pmid': 37559903,
                'target_gene': 'Wnt7b',
                'relation_type': 'Positive_Correlation',
                'score': 0.998,
                'title': 'A Frizzled4-LRP5 agonist...',
                'sentences': ['Norrin and WNT7A/B induce blood-brain barrier...']
            },
            ...
        ],
        'top_papers': [{'pmid': 123, 'title': '...'}, ...]
    }
    """
    # Build query: regulator AND (gene1 OR gene2 OR ...) AND keyword
    # Search for regulator with program genes in endothelial keyword
    genes_or = " OR ".join(program_genes[:10])  # Use top 10 program genes
    query = f"({regulator}) AND ({genes_or}) AND ({keyword})"
    
    logger.info(f"  Validating {regulator}: {query[:80]}...")
    
    # Search
    time.sleep(PUBTATOR_RATE_LIMIT)
    search_results = search_pubtator(query, max_results=60)
    
    if not search_results:
        return {
            'regulator': regulator,
            'papers_found': 0,
            'papers_with_relations': 0,
            'mechanistic_relations': [],
            'top_papers': []
        }
    
    # Fetch BioC for top PMIDs (with abstract text)
    pmids = [r['pmid'] for r in search_results[:max_pmids] if r['pmid']]
    
    time.sleep(PUBTATOR_RATE_LIMIT)
    pmid_data = fetch_bioc_relations_with_text(pmids)
    
    # Extract mechanistic relations involving the regulator
    # Only Positive_Correlation and Negative_Correlation (skip Association)
    # Only keep relations where the target gene is in the program's gene list
    mechanistic_relations = []
    regulator_upper = regulator.upper()
    program_genes_upper = {g.upper() for g in program_genes}  # For filtering target genes
    seen_gene_pmid = set()  # Deduplicate by (gene, pmid) pair
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
            
            # Only mechanistic relation types
            if rel_type not in ('Positive_Correlation', 'Negative_Correlation'):
                continue
            
            # Check if regulator is involved
            if regulator_upper not in (g1, g2):
                continue
            
            pmids_with_relations.add(pmid)
            other_gene = rel['gene2'] if g1 == regulator_upper else rel['gene1']
            dedup_key = (other_gene.upper(), pmid)
            
            if dedup_key in seen_gene_pmid:
                continue
            seen_gene_pmid.add(dedup_key)
            
            # Extract mechanistic sentences mentioning both genes
            sentences = extract_mechanistic_sentences(
                abstract, regulator, other_gene, max_sentences=2
            )
            
            # If no sentence with both genes, try finding any mechanistic sentence about regulator
            if not sentences:
                sentences = extract_any_mechanistic_sentences(
                    abstract, regulator, max_sentences=1
                )
            
            mechanistic_relations.append({
                'pmid': pmid,
                'target_gene': other_gene,
                'relation_type': rel_type,
                'score': rel['score'],
                'title': title,
                'sentences': sentences
            })
    
    # Fallback: If no formal relations found, still extract mechanistic sentences from abstracts
    # This catches cases where the regulator is mentioned but not in PubTator relations
    if not mechanistic_relations:
        for pmid, data in pmid_data.items():
            abstract = data.get('abstract', '')
            title = data.get('title', '')
            gene_mentions = data.get('gene_mentions', {})
            
            if not abstract:
                continue
            
            # Get all text aliases for this regulator from PubTator annotations
            regulator_aliases = gene_mentions.get(regulator.upper(), [regulator])
            if not regulator_aliases:
                regulator_aliases = [regulator]
            
            # Try to find any mechanistic sentence using the regulator or its aliases
            sentences = []
            for alias in regulator_aliases:
                sentences = extract_any_mechanistic_sentences(abstract, alias, max_sentences=2)
                if sentences:
                    break
            
            # If no alias worked, try the original name
            if not sentences:
                sentences = extract_any_mechanistic_sentences(abstract, regulator, max_sentences=2)
            
            if sentences:
                mechanistic_relations.append({
                    'pmid': pmid,
                    'target_gene': 'unknown',
                    'relation_type': 'Mentioned',
                    'score': 0.5,  # Lower confidence score
                    'title': title,
                    'sentences': sentences
                })
    
    # Sort by score descending, keep top 10
    mechanistic_relations.sort(key=lambda x: -x['score'])
    mechanistic_relations = mechanistic_relations[:10]
    
    # Build top papers list (for context even without relations)
    top_papers = []
    for r in search_results[:10]:
        top_papers.append({
            'pmid': r['pmid'],
            'title': r['title']
        })
    
    papers_with_rels = len([p for p, d in pmid_data.items() if d.get('relations')])
    
    return {
        'regulator': regulator,
        'papers_found': len(search_results),
        'papers_with_relations': papers_with_rels,
        'mechanistic_relations': mechanistic_relations,
        'top_papers': top_papers
    }


def validate_program_regulators(
    program_id: int,
    regulator_data: Dict[int, pd.DataFrame],
    program_genes: List[str],
    keyword: str = "endothelial OR vascular",
    top_n_regulators: int = 3,
    top_n_positive_regulators: Optional[int] = None,
    top_n_negative_regulators: Optional[int] = None,
    max_pmids_per_regulator: int = 50,
    use_string: bool = True,
    use_all_significant: bool = False,
    max_regulators: int = 20,
    use_batch: bool = True,
    min_score: int = 400
) -> Dict[str, Any]:
    """Validate all top regulators for a program.
    
    Uses STRING-DB for interaction validation (default) or PubTator3 for literature.
    With use_batch=True, makes ONE STRING query per program (much faster).
    
    Args:
        use_all_significant: If True, validate all significant regulators (not just top N)
        max_regulators: Maximum regulators per category when using all significant
        use_batch: If True, use single batch STRING query (faster)
        min_score: Minimum STRING combined score (0-1000, 400=medium confidence)
    
    Returns: {
        'positive_regulators': [validation_result, ...],
        'negative_regulators': [validation_result, ...]
    }
    """
    top_regs = get_top_regulators(
        regulator_data, 
        program_id, 
        top_n=top_n_regulators,
        top_n_positive=top_n_positive_regulators,
        top_n_negative=top_n_negative_regulators,
        use_all_significant=use_all_significant,
        max_regulators=max_regulators
    )
    
    result = {
        'positive_regulators': [],
        'negative_regulators': []
    }
    
    all_regulator_genes = [r['gene'] for r in top_regs['positive']] + [r['gene'] for r in top_regs['negative']]
    
    # Use batch STRING query (ONE call for all regulators)
    if use_string and use_batch and all_regulator_genes:
        batch_results = batch_validate_regulators(
            regulator_genes=all_regulator_genes,
            program_genes=program_genes,
            required_score=min_score
        )
        
        # Build results for positive regulators (activators)
        for reg_info in top_regs['positive']:
            gene = reg_info['gene']
            interactions = batch_results.get(gene, [])
            result['positive_regulators'].append({
                'regulator': gene,
                'log2fc': reg_info['log2fc'],
                'n_program_targets': len(interactions),
                'string_interactions': interactions  # List of {target, score}
            })
        
        # Build results for negative regulators (repressors)
        for reg_info in top_regs['negative']:
            gene = reg_info['gene']
            interactions = batch_results.get(gene, [])
            result['negative_regulators'].append({
                'regulator': gene,
                'log2fc': reg_info['log2fc'],
                'n_program_targets': len(interactions),
                'string_interactions': interactions
            })
        
        return result
    
    # Fallback: individual queries (slower)
    for reg_info in top_regs['positive']:
        if use_string:
            logger.info(f"  Validating activator {reg_info['gene']} with STRING...")
            validation = validate_regulator_with_string(
                regulator=reg_info['gene'],
                program_genes=program_genes
            )
        else:
            validation = validate_regulator_program(
                regulator=reg_info['gene'],
                program_genes=program_genes,
                keyword=keyword,
                max_pmids=max_pmids_per_regulator
            )
        validation['log2fc'] = reg_info['log2fc']
        result['positive_regulators'].append(validation)
    
    for reg_info in top_regs['negative']:
        if use_string:
            logger.info(f"  Validating repressor {reg_info['gene']} with STRING...")
            validation = validate_regulator_with_string(
                regulator=reg_info['gene'],
                program_genes=program_genes
            )
        else:
            validation = validate_regulator_program(
                regulator=reg_info['gene'],
                program_genes=program_genes,
                keyword=keyword,
                max_pmids=max_pmids_per_regulator
            )
        validation['log2fc'] = reg_info['log2fc']
        result['negative_regulators'].append(validation)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch NCBI literature evidence for gene programs")
    parser.add_argument("--config", help="Path to config file (YAML or JSON)")
    parser.add_argument(
        "--input",
        help="Loading CSV (Name, Score, program_id or RowID; UniquenessScore optional)",
    )
    parser.add_argument("--csv-out", help="Summary CSV output")
    parser.add_argument("--json-out", help="Full JSON context output")
    parser.add_argument(
        "--keyword",
        default='(endothelial OR endothelium OR "vascular endothelial")',
        help="Keyword query string for PubMed search",
    )
    parser.add_argument("--api-key", help="NCBI API Key")
    parser.add_argument(
        "--gene-summary-source",
        choices=["ncbi", "harmonizome"],
        default="harmonizome",
        help="Source for gene summaries: harmonizome (default) or ncbi (Entrez)",
    )
    parser.add_argument(
        "--full-summaries",
        action="store_true",
        help="Fetch full literature-based summaries from Harmonizome HTML pages (~3x longer). "
             "Default is short API descriptions.",
    )
    parser.add_argument("--num-programs", type=int, help="Limit number of programs (for testing)")
    parser.add_argument("--topics", type=str, help="Comma-separated list of topic IDs to process (e.g. '6,7,8')")
    parser.add_argument("--top-loading", type=int, default=20, help="Number of top loading genes (default 20)")
    parser.add_argument("--top-unique", type=int, default=10, help="Number of top unique genes, non-overlapping with loading (default 10)")
    parser.add_argument("--top-member", type=int, default=100, help="Number of member genes for context")
    parser.add_argument("--regulator-file", type=str, help="CSV with regulator perturbation results")
    parser.add_argument("--top-regulators", type=int, default=3, help="Backward-compatible default for positive/negative regulator counts")
    parser.add_argument("--top-positive-regulators", type=int, help="Number of top positive regulators to validate")
    parser.add_argument("--top-negative-regulators", type=int, help="Number of top negative regulators to validate")
    parser.add_argument("--regulator-significance-threshold", type=float, default=0.05, help="Adjusted p-value threshold used when the regulator file has no 'significant' column (default 0.05)")
    parser.add_argument("--regulator-pmids", type=int, default=50, help="Number of PMIDs to fetch per regulator (default 50)")
    parser.add_argument("--all-significant", action="store_true", help="Validate ALL significant regulators (not just top N). Fast with STRING.")
    parser.add_argument("--max-regulators", type=int, default=20, help="Maximum regulators per category when using --all-significant (default 20)")
    
    args = parser.parse_args()
    config = load_config(args.config)
    cli_overrides = get_cli_overrides(sys.argv)
    args = apply_config_overrides(args, config, cli_overrides)
    args = apply_test_mode(args, config, cli_overrides)

    if not args.input or not args.csv_out or not args.json_out:
        raise SystemExit("--input, --csv-out, and --json-out are required (via CLI or config).")
    
    # Init Client
    client = NcbiClient(api_key=args.api_key)
    
    # Load Programs
    logger.info(f"Loading programs from {args.input}...")
    programs = load_program_genes(
        Path(args.input), 
        top_n_loading=args.top_loading, 
        top_n_unique=args.top_unique,
        top_n_member=args.top_member
    )
    program_ids = sorted(programs.keys())
    
    # Filter by specific topics if requested
    selected_topics = parse_topics_value(args.topics)
    if selected_topics:
        program_ids = [pid for pid in program_ids if pid in selected_topics]
        logger.info(f"Limiting to specific topics: {program_ids}")
    # Fallback to num-programs limit
    elif args.num_programs:
        program_ids = program_ids[:args.num_programs]
        logger.info(f"Limiting to first {args.num_programs} programs: {program_ids}")

    # =========================================================================
    # Step 1: Search (Gather PMIDs)
    # =========================================================================
    
    all_pmids = set()
    program_pmid_map = {} # pid -> [pmids]
    program_meta = {} # pid -> {query: str, total_hits: int}
    
    for pid in program_ids:
        drivers = programs[pid]["drivers"]
        if not drivers:
            continue
            
        # Construct Query
        # (Gene1 OR Gene2 ...) AND Keyword
        genes_or = " OR ".join(drivers)
        query = f"({genes_or}) AND {args.keyword}"
        
        logger.info(f"[Program {pid}] Searching: {query}")
        pmids = client.search_literature(query, size=25)
        logger.info(f"[Program {pid}] Found {len(pmids)} articles")
        
        program_pmid_map[pid] = pmids
        program_meta[pid] = {
            "query": query,
            "total_hits": len(pmids)
        }
        all_pmids.update(pmids)

    # =========================================================================
    # Step 1.5: Fetch Gene Summaries (NCBI or Harmonizome)
    # =========================================================================

    gene_summary_source = args.gene_summary_source
    use_full_summaries = getattr(args, "full_summaries", False)
    program_gene_summaries = resolve_gene_summaries(
        source=gene_summary_source,
        programs=programs,
        program_ids=program_ids,
        ncbi_client=client,
        use_full_summaries=use_full_summaries,
    )

    # =========================================================================
    # Step 2: Fetch BioC Attributes (Fetch & Cache)
    # =========================================================================
    
    logger.info(f"Fetching annotations for {len(all_pmids)} total PMIDs...")
    cache = load_cache()
    
    # Determine missing PMIDs
    missing_pmids = [p for p in all_pmids if str(p) not in cache]
    
    if missing_pmids:
        # Batch fetch
        BATCH_SIZE = 100
        for i in range(0, len(missing_pmids), BATCH_SIZE):
            batch = missing_pmids[i:i+BATCH_SIZE]
            logger.info(f"Fetching batch {i}-{i+len(batch)}/{len(missing_pmids)}")
            
            docs = client.fetch_bioc_annotations(batch)
            
            # Update cache with docs
            for doc in docs:
                # Find ID
                # BioC ID is in doc["id"] usually
                doc_id = doc.get("id")
                if doc_id:
                    cache[str(doc_id)] = doc
            
            # Save incrementally
            save_cache(cache)
            
    logger.info(f"Cache updated. Total docs in cache: {len(cache)}")

    
    final_context = {}  # pid -> {literautre: ...}
    summary_rows = []
    
    for pid in program_ids:
        pmids = program_pmid_map.get(pid, [])
        members = set(programs[pid]["members"]) # All top 50 (Context)
        drivers = set(programs[pid]["drivers"]) # Top 10 (Targets)
        
        scored_papers = []
        
        # Aggregation of Evidence Sentences
        # gene -> list of "sentence (PMID:123)"
        aggregated_snippets = {g: [] for g in drivers}
        
        for pmid in pmids:
            doc = cache.get(str(pmid))
            if not doc:
                continue
                
            # 1. Parsing
            abstract = parse_bioc_abstract(doc)
            title = ""
            for p in doc.get("passages", []):
                    if p.get("infons", {}).get("type") == "title":
                        title = p.get("text", "")
                        break
            if not title: title = abstract[:50] + "..."
            
            # 2. Identify Genes mentioned in this paper (for scoring the paper itself)
            mentioned_genes = find_gene_mentions(doc, list(members))
            
            # 3. High-Quality Papers (Gold Standard) - Score >= 2
            score = len(mentioned_genes)
            if score >= 2:
                scored_papers.append({
                    "pmid": pmid,
                    "title": title,
                    "score": score,
                    "mentions": mentioned_genes,
                    "abstract": abstract
                })
                
            # 4. Sentence-Level Evidence Extraction
            # We extract sentences for ALL papers, even low scoring ones, 
            # if they contain a Driver Gene + useful context.
            
            evidence_map = extract_evidence_sentences(abstract, title, drivers, members)
            
            for g, sentences in evidence_map.items():
                for s in sentences:
                    snippet = f'"{s}" (PMID:{pmid})'
                    aggregated_snippets[g].append(snippet)
        
        # Sort Papers by Score
        scored_papers.sort(key=lambda x: x["score"], reverse=True)
        top_papers = scored_papers[:5]  # Keep top 5
        
        # Build Context JSON Data
        paper_contexts = []
        for p in top_papers:
            paper_contexts.append({
                "pmid": p["pmid"],
                "title": p["title"],
                "score": p["score"],
                "genes": p["mentions"]
            })
            
            # Add to Summary CSV
            summary_rows.append({
                "program_id": pid,
                "pmid": p["pmid"],
                "score": p["score"],
                "title": p["title"],
                "genes_found": "|".join(p["mentions"])
            })
            
        final_context[pid] = {
            "meta": program_meta.get(pid, {}),
            "top_papers": paper_contexts,
            "gene_summaries": program_gene_summaries.get(pid, {}),
            "gene_summaries_source": gene_summary_source,
            "evidence_snippets": aggregated_snippets
        }

    # =========================================================================
    # Step 4: Regulator Validation (if regulator file provided)
    # =========================================================================
    
    if args.regulator_file:
        logger.info("=" * 60)
        logger.info("Step 4: Validating regulator-program relationships via PubTator3")
        logger.info("=" * 60)
        
        regulator_data = load_regulator_data(
            Path(args.regulator_file),
            significance_threshold=args.regulator_significance_threshold,
        )
        
        for pid in program_ids:
            # Use all 300 genes for STRING validation (not just drivers)
            program_genes = programs[pid].get("all_genes", programs[pid]["drivers"])
            
            if args.all_significant:
                logger.info(f"[Program {pid}] Validating ALL significant regulators against {len(program_genes)} program genes...")
            else:
                top_pos = args.top_positive_regulators or args.top_regulators
                top_neg = args.top_negative_regulators or args.top_regulators
                logger.info(
                    f"[Program {pid}] Validating top {top_pos} positive + "
                    f"{top_neg} negative regulators against {len(program_genes)} program genes..."
                )
            
            validation_result = validate_program_regulators(
                program_id=pid,
                regulator_data=regulator_data,
                program_genes=program_genes,
                keyword=args.keyword,
                top_n_regulators=args.top_regulators,
                top_n_positive_regulators=args.top_positive_regulators,
                top_n_negative_regulators=args.top_negative_regulators,
                max_pmids_per_regulator=args.regulator_pmids,
                use_all_significant=args.all_significant,
                max_regulators=args.max_regulators
            )
            
            # Add to final context
            final_context[pid]["regulator_validation"] = validation_result
            
            # Log summary
            pos_count = len(validation_result['positive_regulators'])
            neg_count = len(validation_result['negative_regulators'])
            # Handle both STRING (n_program_targets) and PubTator (papers_found) formats
            total_targets = sum(r.get('n_program_targets', r.get('papers_found', 0)) or 0 
                               for r in validation_result['positive_regulators'])
            total_targets += sum(r.get('n_program_targets', r.get('papers_found', 0)) or 0 
                                for r in validation_result['negative_regulators'])
            logger.info(f"[Program {pid}] Validated {pos_count} positive + {neg_count} negative regulators, {total_targets} total STRING interactions")

    # =========================================================================
    # Output
    # =========================================================================
    
    # JSON
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(final_context, f, indent=2)
    logger.info(f"Written JSON context to {args.json_out}")
    
    # CSV
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        # Reorder
        df_sum = df_sum[["program_id", "score", "genes_found", "pmid", "title"]]
        df_sum.to_csv(args.csv_out, index=False)
        logger.info(f"Written Summary CSV to {args.csv_out}")
    else:
        logger.warning("No high-quality papers found. Summary CSV is empty.")
        Path(args.csv_out).touch()

if __name__ == "__main__":
    main()
