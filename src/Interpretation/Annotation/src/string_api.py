"""
@description
STRING-DB API wrapper for querying protein-protein interactions.
Provides direct mechanistic relationship queries between regulators and gene programs.

Key features:
- Query interactions between a regulator and a list of program genes
- Get action types (activation, inhibition, binding) for mechanistic context
- Batch queries with rate limiting

@dependencies
- requests

@examples
>>> from tools.string_api import get_regulator_program_interactions
>>> interactions = get_regulator_program_interactions(
...     regulator="Fzd4",
...     program_genes=["Ctnnb1", "Lef1", "Wnt7a", "Wnt7b"],
...     species=10090  # mouse
... )
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# STRING API configuration
STRING_API_URL = "https://version-12-0.string-db.org/api"
RATE_LIMIT_SECONDS = 1.0
CALLER_IDENTITY = "topic_annotation_pipeline"

# Mouse taxon ID
MOUSE_TAXON = 10090


def get_string_ids(
    identifiers: List[str],
    species: int = MOUSE_TAXON,
    limit: int = 1
) -> Dict[str, str]:
    """Map gene names to STRING identifiers.
    
    Args:
        identifiers: List of gene names/symbols
        species: NCBI taxon ID (10090 for mouse)
        limit: Max matches per identifier
        
    Returns:
        Dict mapping input name -> STRING ID (e.g., "Fzd4" -> "10090.ENSMUSP00000029156")
    """
    if not identifiers:
        return {}
    
    request_url = f"{STRING_API_URL}/json/get_string_ids"
    
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "limit": limit,
        "caller_identity": CALLER_IDENTITY
    }
    
    try:
        response = requests.post(request_url, data=params, timeout=30)
        response.raise_for_status()
        results = response.json()
        
        # Map input name to STRING ID
        mapping = {}
        for r in results:
            # Use preferredName as key for reliable matching
            pref_name = r.get("preferredName", "").lower()
            string_id = r.get("stringId", "")
            if pref_name and string_id:
                mapping[pref_name] = string_id
                
        return mapping
        
    except Exception as e:
        logger.warning(f"STRING ID mapping failed: {e}")
        return {}


def get_network_interactions(
    identifiers: List[str],
    species: int = MOUSE_TAXON,
    required_score: int = 400,
    network_type: str = "functional"
) -> List[Dict[str, Any]]:
    """Get interactions between a set of proteins.
    
    Args:
        identifiers: List of gene names
        species: NCBI taxon ID
        required_score: Minimum combined score (0-1000, default 400 = medium confidence)
        network_type: "functional" or "physical"
        
    Returns:
        List of interaction dicts with scores for each evidence channel
    """
    if len(identifiers) < 2:
        return []
    
    request_url = f"{STRING_API_URL}/json/network"
    
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "required_score": required_score,
        "network_type": network_type,
        "caller_identity": CALLER_IDENTITY
    }
    
    try:
        response = requests.post(request_url, data=params, timeout=60)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.warning(f"STRING network query failed: {e}")
        return []


def get_interaction_partners(
    identifiers: List[str],
    species: int = MOUSE_TAXON,
    limit: int = 50,
    required_score: int = 400
) -> List[Dict[str, Any]]:
    """Get all STRING interaction partners of proteins.
    
    Args:
        identifiers: List of gene names
        species: NCBI taxon ID  
        limit: Max partners per protein
        required_score: Minimum combined score
        
    Returns:
        List of interaction dicts
    """
    request_url = f"{STRING_API_URL}/json/interaction_partners"
    
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "limit": limit,
        "required_score": required_score,
        "caller_identity": CALLER_IDENTITY
    }
    
    try:
        response = requests.post(request_url, data=params, timeout=60)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.warning(f"STRING interaction_partners query failed: {e}")
        return []


def get_regulator_program_interactions(
    regulator: str,
    program_genes: List[str],
    species: int = MOUSE_TAXON,
    required_score: int = 400,
    top_n: int = 10,
    partner_limit: int = 500
) -> Dict[str, Any]:
    """Query STRING for interactions between a regulator and program genes.
    
    This is the main function for validating regulator-program relationships.
    It queries STRING for the regulator and finds which program genes it 
    directly interacts with according to STRING evidence.
    
    Args:
        regulator: Gene symbol of the regulator (e.g., "Fzd4")
        program_genes: List of program's gene symbols (can be up to 300)
        species: NCBI taxon ID (10090 for mouse)
        required_score: Minimum STRING combined score (0-1000)
        top_n: Max interactions to return in output
        partner_limit: Max partners to fetch from STRING (default 500)
        
    Returns:
        Dict with:
        - regulator: str
        - interactions: List of {target_gene, score, evidence_scores}
        - n_interactions: int (total found, before top_n limit)
        - program_genes_with_evidence: List of gene names that interact
    """
    result = {
        "regulator": regulator,
        "interactions": [],
        "n_interactions": 0,
        "program_genes_with_evidence": []
    }
    
    # Query STRING for regulator's interaction partners
    # Use higher limit to search against larger program gene lists
    partners = get_interaction_partners(
        identifiers=[regulator],
        species=species,
        limit=partner_limit,
        required_score=required_score
    )
    
    if not partners:
        logger.debug(f"No STRING partners found for {regulator}")
        return result
    
    # Create lowercase set of program genes for matching
    program_set = {g.lower() for g in program_genes}
    
    # Filter to only interactions with program genes
    program_interactions = []
    for p in partners:
        # STRING returns preferredName_A (query) and preferredName_B (partner)
        partner_name = p.get("preferredName_B", "")
        if partner_name.lower() in program_set:
            interaction = {
                "target_gene": partner_name,
                "score": int(p.get("score", 0) * 1000) if p.get("score", 0) <= 1 else int(p.get("score", 0)),
                "experimental_score": int(float(p.get("escore", 0)) * 1000),
                "database_score": int(float(p.get("dscore", 0)) * 1000),
                "textmining_score": int(float(p.get("tscore", 0)) * 1000),
                "coexpression_score": int(float(p.get("ascore", 0)) * 1000),
            }
            program_interactions.append(interaction)
    
    # Sort by score and take top N
    program_interactions.sort(key=lambda x: -x["score"])
    program_interactions = program_interactions[:top_n]
    
    result["interactions"] = program_interactions
    result["n_interactions"] = len(program_interactions)
    result["program_genes_with_evidence"] = [i["target_gene"] for i in program_interactions]
    
    return result


def batch_validate_regulators(
    regulator_genes: List[str],
    program_genes: List[str],
    species: int = MOUSE_TAXON,
    required_score: int = 400
) -> Dict[str, List[Dict[str, Any]]]:
    """Batch query STRING for all regulator-program interactions in ONE call.
    
    Much faster than individual queries - one API call per program.
    
    Args:
        regulator_genes: List of regulator gene symbols
        program_genes: List of program gene symbols
        species: NCBI taxon ID
        required_score: Minimum combined score (0-1000, 400 = medium confidence)
        
    Returns:
        Dict mapping regulator -> list of interactions with program genes
        {
            'Fzd4': [{'target': 'Znrf3', 'score': 943}, ...],
            'Tek': [{'target': 'Aplnr', 'score': 521}, ...],
            ...
        }
    """
    result = {reg: [] for reg in regulator_genes}
    
    if not regulator_genes or not program_genes:
        return result
    
    # Combine all genes for single query
    all_genes = list(set(regulator_genes + program_genes))
    
    logger.info(f"  Batch STRING query: {len(regulator_genes)} regulators x {len(program_genes)} program genes...")
    
    # Query STRING network for all interactions
    interactions = get_network_interactions(
        identifiers=all_genes,
        species=species,
        required_score=required_score
    )
    
    if not interactions:
        logger.warning("  No STRING interactions returned")
        return result
    
    # Create sets for fast lookup
    regulator_set = {g.lower() for g in regulator_genes}
    program_set = {g.lower() for g in program_genes}
    
    # Parse interactions: keep only regulator <-> program gene pairs
    for i in interactions:
        gene_a = i.get("preferredName_A", "")
        gene_b = i.get("preferredName_B", "")
        score = int(i.get("score", 0) * 1000) if i.get("score", 0) <= 1 else int(i.get("score", 0))
        
        # Check if this is a regulator-program interaction
        a_lower, b_lower = gene_a.lower(), gene_b.lower()
        
        if a_lower in regulator_set and b_lower in program_set:
            # gene_a is regulator, gene_b is program gene
            for reg in regulator_genes:
                if reg.lower() == a_lower:
                    result[reg].append({"target": gene_b, "score": score})
                    break
        elif b_lower in regulator_set and a_lower in program_set:
            # gene_b is regulator, gene_a is program gene
            for reg in regulator_genes:
                if reg.lower() == b_lower:
                    result[reg].append({"target": gene_a, "score": score})
                    break
    
    # Sort each regulator's interactions by score
    for reg in result:
        result[reg].sort(key=lambda x: -x["score"])
    
    n_total = sum(len(v) for v in result.values())
    n_with_int = sum(1 for v in result.values() if v)
    logger.info(f"  Found {n_total} interactions for {n_with_int}/{len(regulator_genes)} regulators")
    
    return result


def validate_regulators_with_string(
    regulators: List[Dict[str, Any]],
    program_genes: List[str],
    species: int = MOUSE_TAXON,
    top_regulators: int = 3,
    required_score: int = 400
) -> Dict[str, Any]:
    """Validate multiple regulators against program genes using STRING.
    
    Args:
        regulators: List of regulator dicts with 'gene', 'log2FC', etc.
        program_genes: List of program gene symbols
        species: NCBI taxon ID
        top_regulators: Number of top regulators per category to validate
        required_score: Minimum STRING score
        
    Returns:
        Dict with 'activators' and 'repressors', each containing validated
        regulator info with STRING interactions.
    """
    result = {
        "activators": [],
        "repressors": []
    }
    
    # Split into activators (negative log2FC) and repressors (positive log2FC)
    activators = [r for r in regulators if r.get("log2FC", 0) < 0]
    repressors = [r for r in regulators if r.get("log2FC", 0) > 0]
    
    # Sort by absolute effect size
    activators.sort(key=lambda x: x.get("log2FC", 0))  # Most negative first
    repressors.sort(key=lambda x: -x.get("log2FC", 0))  # Most positive first
    
    # Validate top activators
    for i, reg in enumerate(activators[:top_regulators]):
        gene = reg.get("gene", "")
        logger.info(f"  Validating activator {gene} with STRING...")
        
        time.sleep(RATE_LIMIT_SECONDS)
        interactions = get_regulator_program_interactions(
            regulator=gene,
            program_genes=program_genes,
            species=species,
            required_score=required_score
        )
        
        result["activators"].append({
            "gene": gene,
            "log2FC": reg.get("log2FC", 0),
            "pvalue": reg.get("pvalue"),
            "string_interactions": interactions["interactions"],
            "n_program_targets": interactions["n_interactions"]
        })
    
    # Add remaining activators without validation
    for reg in activators[top_regulators:top_regulators + 5]:
        result["activators"].append({
            "gene": reg.get("gene", ""),
            "log2FC": reg.get("log2FC", 0),
            "pvalue": reg.get("pvalue"),
            "string_interactions": None,  # Not validated
            "n_program_targets": None
        })
    
    # Validate top repressors
    for i, reg in enumerate(repressors[:top_regulators]):
        gene = reg.get("gene", "")
        logger.info(f"  Validating repressor {gene} with STRING...")
        
        time.sleep(RATE_LIMIT_SECONDS)
        interactions = get_regulator_program_interactions(
            regulator=gene,
            program_genes=program_genes,
            species=species,
            required_score=required_score
        )
        
        result["repressors"].append({
            "gene": gene,
            "log2FC": reg.get("log2FC", 0),
            "pvalue": reg.get("pvalue"),
            "string_interactions": interactions["interactions"],
            "n_program_targets": interactions["n_interactions"]
        })
    
    # Add remaining repressors without validation  
    for reg in repressors[top_regulators:top_regulators + 5]:
        result["repressors"].append({
            "gene": reg.get("gene", ""),
            "log2FC": reg.get("log2FC", 0),
            "pvalue": reg.get("pvalue"),
            "string_interactions": None,
            "n_program_targets": None
        })
    
    return result


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with Fzd4 and some known Wnt pathway genes
    test_genes = ["Ctnnb1", "Lef1", "Wnt7a", "Wnt7b", "Axin2", "Vegfa", "Kdr"]
    
    print("Testing STRING API...")
    print(f"Query: Fzd4 interactions with {test_genes}")
    
    result = get_regulator_program_interactions(
        regulator="Fzd4",
        program_genes=test_genes,
        species=10090
    )
    
    print(f"\nFound {result['n_interactions']} interactions:")
    for i in result["interactions"]:
        print(f"  Fzd4 -- {i['target_gene']} (score={i['score']}, exp={i['experimental_score']}, db={i['database_score']})")
