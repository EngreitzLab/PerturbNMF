"""Gene coordinates for the deterministic screens.

Input: the `gene_coordinates.tsv` every annotator takes — no header, one row per gene record:
`name chrom start end strand gene_type` (1-based, from the GTF used for alignment). A symbol can
have several records (rRNA repeats, PAR genes).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_gene_coordinates(path: Path) -> Dict[str, dict]:
    """Map gene symbol -> {chrom, start, end, gene_type}, keeping the first record per symbol.

    A handful of symbols (rRNA repeats, PAR genes) appear on several contigs. Taking the first
    sorted record is arbitrary but harmless here: the positional screen asks whether the top
    genes concentrate somewhere, and a repeat family is not what drives that signal.
    """
    coordinates: Dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            name, chrom, start, end, _strand, gene_type = line.rstrip("\n").split("\t")
            if name in coordinates:
                continue
            coordinates[name] = {
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "gene_type": gene_type,
            }
    return coordinates


def load_gene_tss(path: Path) -> Dict[str, List[dict]]:
    """Map gene symbol -> every {chrom, tss, strand, gene_type} record (TSS = start on +, end on -).

    Unlike load_gene_coordinates this keeps every record: a promoter screen must not miss a
    neighbour because its symbol also sits on another contig.
    """
    records: Dict[str, List[dict]] = {}
    with path.open() as handle:
        for line in handle:
            name, chrom, start, end, strand, gene_type = line.rstrip("\n").split("\t")
            tss = int(start) if strand == "+" else int(end)
            records.setdefault(name, []).append(
                {"chrom": chrom, "tss": tss, "strand": strand, "gene_type": gene_type}
            )
    return records


# ---- CRISPRi promoter neighbours ----------------------------------------------------------
# hCRISPRi-v2-style guide names carry the protospacer position: GENE_<strand>_<position>.<len>-<promoter>
# (e.g. "ACAA1_+_38178488.23-P1P2"). The chromosome comes from the target gene's record.
GUIDE_NAME = re.compile(r"^(?P<gene>.+?)_(?P<strand>[+-])_(?P<position>\d+)\.\d+(?:-(?P<promoter>\S+))?$")
DEFAULT_NEIGHBOUR_TYPES = ("protein_coding", "lncRNA")


def parse_guide_position(name: str) -> Optional[Tuple[str, int]]:
    """(target symbol, position) from a hCRISPRi-v2-style guide name, or None."""
    match = GUIDE_NAME.match(str(name))
    return (match.group("gene"), int(match.group("position"))) if match else None


def target_sites(target: str, tss: Dict[str, List[dict]], guide_positions: Dict[str, List[int]]) -> List[dict]:
    """Where the CRISPRi guides of a target act: its guide positions when known (on the target's
    chromosome), else every TSS record of the target. Each site: {chrom, position, source}."""
    records = tss.get(target, [])
    if not records:
        return []
    if guide_positions.get(target):
        chrom = records[0]["chrom"]
        return [{"chrom": chrom, "position": p, "source": "guide"} for p in sorted(set(guide_positions[target]))]
    return [{"chrom": r["chrom"], "position": r["tss"], "source": "tss"} for r in records]


def orientation(target_record: Optional[dict], neighbour_record: dict) -> str:
    """divergent (head-to-head, bidirectional promoter), convergent / other, or same strand."""
    if not target_record:
        return "unknown"
    if target_record["strand"] == neighbour_record["strand"]:
        return "same strand"
    plus, minus = (target_record, neighbour_record) if target_record["strand"] == "+" else (neighbour_record, target_record)
    return "divergent" if minus["tss"] <= plus["tss"] + 100 else "opposite strand"


def is_readthrough_of(neighbour: str, target: str) -> bool:
    """Readthrough loci (TIMM23B-AGAP6) share the target's TSS by construction; not a neighbour."""
    return neighbour != target and target in neighbour.split("-")


def promoter_neighbours(target: str, tss: Dict[str, List[dict]], guide_positions: Dict[str, List[int]],
                        window: int, gene_types=DEFAULT_NEIGHBOUR_TYPES) -> List[dict]:
    """Genes with a TSS within `window` bp of any site where the target's guides act."""
    sites = target_sites(target, tss, guide_positions)
    if not sites:
        return []
    by_chrom: Dict[str, list] = {}
    for site in sites:
        by_chrom.setdefault(site["chrom"], []).append(site)
    target_record = min(tss.get(target, []), key=lambda r: min(abs(r["tss"] - s["position"]) for s in sites), default=None)
    found: Dict[str, dict] = {}
    for gene, records in tss.items():
        if gene == target or is_readthrough_of(gene, target):
            continue
        for record in records:
            if record["gene_type"] not in gene_types or record["chrom"] not in by_chrom:
                continue
            distance = min(abs(record["tss"] - s["position"]) for s in by_chrom[record["chrom"]])
            if distance <= window and (gene not in found or distance < found[gene]["distance"]):
                found[gene] = {"gene": gene, "distance": int(distance), "gene_type": record["gene_type"],
                               "orientation": orientation(target_record, record),
                               "site_source": sites[0]["source"]}
    return sorted(found.values(), key=lambda n: n["distance"])


def guide_positions_on_this_assembly(guide_positions: Dict[str, List[int]], tss: Dict[str, List[dict]],
                                     max_offset: int = 2000) -> Tuple[Dict[str, List[int]], List[str]]:
    """Keep a target's guide positions only when they sit at its promoter in THIS coordinate file.

    Guide libraries are often named on an older assembly (hCRISPRi-v2 names carry hg19
    positions) while the coordinates come from the alignment GTF (hg38); a position that is
    hundreds of kb from every TSS of its target is on another assembly and would place the
    "promoter" at a random locus. Such targets fall back to their TSSs. Returns (kept, dropped).
    """
    kept, dropped = {}, []
    for target, positions in guide_positions.items():
        records = tss.get(target, [])
        near = [p for p in positions if records and min(abs(p - r["tss"]) for r in records) <= max_offset]
        if near and len(near) >= len(positions) / 2:
            kept[target] = near
        else:
            dropped.append(target)
    return kept, sorted(dropped)
