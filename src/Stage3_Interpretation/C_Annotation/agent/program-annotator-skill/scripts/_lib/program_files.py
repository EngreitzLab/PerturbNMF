"""Readers for one program folder (Programs/P<k>/); stdlib only, so this skill needs nothing outside itself."""

import csv
import json
import re
from pathlib import Path

FAMILY_NAMES = {  # build_confounder_screens.py family / marker-set keys
    "cytoplasmic_ribosome": "Cytoplasmic ribosome",
    "mitochondrial_ribosome": "Mitochondrial ribosome",
    "mitochondrially_encoded": "Mitochondrially encoded (MT-)",
    "histone": "Histone",
    "small_nucleolar_or_nuclear_rna": "snoRNA / snRNA",
    "microrna": "microRNA",
    "long_noncoding_or_antisense": "lncRNA / antisense",
    "cell_cycle_s_phase": "Cell cycle, S phase",
    "cell_cycle_g2m_phase": "Cell cycle, G2/M phase",
    "heat_shock": "Heat shock",
    "integrated_stress_response": "Integrated stress response",
    "unfolded_protein_response": "Unfolded protein response",
    "interferon_response": "Interferon response",
}


def family_name(key):
    return FAMILY_NAMES.get(key, key.replace("_", " "))


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def sibling_programs(program_dir):
    """Every program folder of the run (Programs/P<k>/), numeric order."""
    root = Path(program_dir).resolve().parent
    dirs = [d for d in root.glob("P*") if d.is_dir() and re.fullmatch(r"P\d+", d.name)]
    return sorted(dirs, key=lambda d: int(d.name[1:]))


def regulators(program_dir):
    """{regulator: best row} of regulators_sig.tsv (lowest adj_p over conditions), in table order."""
    best = {}
    for r in read_tsv(Path(program_dir) / "regulators_sig.tsv"):
        if r["regulator"] not in best or float(r["adj_p"]) < float(
            best[r["regulator"]]["adj_p"]
        ):
            best[r["regulator"]] = r
    return best


def short(text, n=320):
    """text without UniProt (PubMed:...) citations, cut to about n characters at a sentence end."""
    text = " ".join(
        re.sub(r"\s*\((?:PubMed:\d+(?:,\s*)?)+[^)]*\)", "", text or "").split()
    )
    if len(text) <= n:
        return text
    cut = text[:n]
    return cut[: cut.rfind(". ") + 1] if ". " in cut else cut.rstrip() + "..."


def gene_card(row):
    """name, function / summary text and location of one gene_info.jsonl row."""
    mg, nc, up = (row.get(s) or {} for s in ("MyGene", "NCBI", "UniProt"))
    return {
        "name": mg.get("name") or nc.get("name") or up.get("protein_name"),
        "function": up.get("function") or mg.get("summary") or nc.get("summary"),
        "location": up.get("subcellular_location") or [],
    }
