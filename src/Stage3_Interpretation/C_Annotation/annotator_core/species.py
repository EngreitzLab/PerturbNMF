"""The species of the screen, for every network lookup that needs one (STRING, mygene.info,
UniProt, QuickGO) and for matching human-curated gene lists (OmniPath complexes, the confounder
marker sets) against the screen's symbols.

Set with the ANNOTATOR_SPECIES environment variable (NCBI taxon id): 9606 human (default) or
10090 mouse. It is an environment variable, like PYTHON and ANNOTATOR_MODEL, so every step of a
run picks it up without a flag on each script.
"""
from __future__ import annotations

import os

SUPPORTED = {9606: "human", 10090: "mouse"}

TAXON = int(os.environ.get("ANNOTATOR_SPECIES", "9606"))
if TAXON not in SUPPORTED:
    raise SystemExit(f"ANNOTATOR_SPECIES={TAXON} is not supported; use one of {sorted(SUPPORTED)}")
MYGENE_SPECIES = SUPPORTED[TAXON]


def from_human_symbol(symbol: str) -> str:
    """A human-curated symbol in this species' convention (mouse: first letter upper, rest lower).

    For mouse this is the ortholog name for the large majority of one-to-one orthologs
    (SMARCA4 -> Smarca4); the exceptions (renamed or many-to-one orthologs) simply fail to match,
    which loses a complex member rather than inventing one.
    """
    if TAXON == 9606:
        return symbol
    return symbol[:1].upper() + symbol[1:].lower()
