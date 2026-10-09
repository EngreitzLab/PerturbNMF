"""Mouse to human ortholog mapping.

Human gene-set libraries (Enrichr, MSigDB human) and human gene summaries are matched
to mouse genes through curated orthologs rather than by upper-casing mouse symbols.
Upper-casing only works when the two species share a symbol, misses orthologs with
different names (e.g. Zfp451 -> ZNF451) and can pick an unrelated human gene.

The ortholog table is the union of two sources, since each has gaps the other fills:
- MGI HOM_MouseHumanSequence.rpt (mouse-human homology classes)
- Ensembl BioMart mouse -> human homologs

The combined table is downloaded on first use and cached as a TSV in
src/Stage2_Evaluation/Resources/, so later runs are offline and reproducible. A
pre-built table can also be passed explicitly (columns: mouse_symbol, mouse_ensembl,
human_symbol, source).
"""

from __future__ import annotations

import io
import logging
import os
import urllib.parse
import urllib.request
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Resources",
)
DEFAULT_TABLE_PATH = os.path.join(RESOURCES_DIR, "orthologs_mouse_human.tsv")
TABLE_COLUMNS = ["mouse_symbol", "mouse_ensembl", "human_symbol", "source"]

MGI_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
# www.ensembl.org redirects BioMart requests; the mirrors answer directly.
ENSEMBL_MARTS = [
    "https://useast.ensembl.org/biomart/martservice",
    "https://asia.ensembl.org/biomart/martservice",
    "https://www.ensembl.org/biomart/martservice",
]
ENSEMBL_QUERY = (
    '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1">'
    '<Dataset name="mmusculus_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="external_gene_name"/>'
    '<Attribute name="hsapiens_homolog_associated_gene_name"/>'
    "</Dataset></Query>"
)
DOWNLOAD_TIMEOUT_S = 600
MOUSE_TAXON = "10090"
HUMAN_TAXON = "9606"


def _download(url: str) -> str:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        return response.read().decode("utf-8")


def _fetch_mgi() -> pd.DataFrame:
    hom = pd.read_csv(io.StringIO(_download(MGI_URL)), sep="\t", dtype=str)
    key, taxon = "DB Class Key", "NCBI Taxon ID"
    mouse = hom.loc[hom[taxon] == MOUSE_TAXON, [key, "Symbol"]]
    human = hom.loc[hom[taxon] == HUMAN_TAXON, [key, "Symbol"]]
    pairs = mouse.merge(human, on=key, suffixes=("_mouse", "_human"))
    if pairs.empty:
        raise RuntimeError(f"No mouse-human pairs parsed from {MGI_URL}")
    return pd.DataFrame(
        {
            "mouse_symbol": pairs["Symbol_mouse"],
            "mouse_ensembl": pd.NA,
            "human_symbol": pairs["Symbol_human"],
            "source": "MGI",
        }
    )


def _fetch_ensembl() -> pd.DataFrame:
    query = urllib.parse.urlencode({"query": ENSEMBL_QUERY})
    errors = []
    for mart in ENSEMBL_MARTS:
        try:
            text = _download(f"{mart}?{query}")
            df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
            if df.shape[1] != 3 or df.empty:
                raise RuntimeError(f"unexpected response: {text[:200]!r}")
            break
        except (OSError, ValueError, RuntimeError) as err:  # try the next mirror
            errors.append(f"{mart}: {err}")
    else:
        raise RuntimeError("Ensembl BioMart query failed:\n" + "\n".join(errors))
    df.columns = ["mouse_ensembl", "mouse_symbol", "human_symbol"]
    # Rows without a human ortholog are kept: they map Ensembl IDs to mouse symbols.
    df["source"] = "Ensembl"
    return df[TABLE_COLUMNS]


def build_ortholog_table() -> pd.DataFrame:
    """Download MGI and Ensembl orthologs and return their union."""
    table = pd.concat([_fetch_mgi(), _fetch_ensembl()], ignore_index=True)
    table = table.dropna(subset=["mouse_symbol"]).drop_duplicates()
    return table[TABLE_COLUMNS].reset_index(drop=True)


def load_ortholog_table(path: str | None = None) -> pd.DataFrame:
    """Load the ortholog table from `path`, the cache, or download and cache it."""
    if path is not None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Ortholog table not found: {path}")
        table = pd.read_csv(path, sep="\t", dtype=str)
    elif os.path.isfile(DEFAULT_TABLE_PATH):
        logger.info(f"Loading cached ortholog table from {DEFAULT_TABLE_PATH}")
        table = pd.read_csv(DEFAULT_TABLE_PATH, sep="\t", dtype=str)
    else:
        logger.info("Downloading mouse-human ortholog table (MGI + Ensembl)")
        table = build_ortholog_table()
        os.makedirs(RESOURCES_DIR, exist_ok=True)
        table.to_csv(DEFAULT_TABLE_PATH, sep="\t", index=False)
        logger.info(f"Saved ortholog table to {DEFAULT_TABLE_PATH}")

    missing = set(TABLE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Ortholog table is missing columns: {sorted(missing)}")
    return table


def map_mouse_to_human(
    genes: Iterable[str], table: pd.DataFrame | None = None
) -> dict[str, list[str]]:
    """Map mouse gene symbols or Ensembl IDs (ENSMUSG...) to human ortholog symbols.

    Returns a dict from each input gene to a sorted list of human symbols; genes with
    no human ortholog map to an empty list. A gene with several human orthologs keeps
    all of them.
    """
    if table is None:
        table = load_ortholog_table()
    pairs = table.dropna(subset=["human_symbol"])
    symbol_to_human = pairs.groupby("mouse_symbol")["human_symbol"].apply(set)
    id_to_symbol = (
        table.dropna(subset=["mouse_ensembl"])
        .groupby("mouse_ensembl")["mouse_symbol"]
        .apply(set)
    )

    mapping = {}
    for gene in genes:
        gene = str(gene)
        if gene.upper().startswith("ENSMUSG"):
            symbols = id_to_symbol.get(gene.split(".", 1)[0], set())  # drop version
        else:
            symbols = {gene}
        human = set()
        for symbol in symbols:
            human |= symbol_to_human.get(symbol, set())
        mapping[gene] = sorted(human)
    return mapping


def collapse_to_human(
    df: pd.DataFrame, table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Re-index a (mouse genes x columns) DataFrame by human ortholog symbols.

    Each human ortholog takes the values of its mouse gene; genes without a human
    ortholog are dropped, and a human gene with several mouse orthologs keeps the
    per-column maximum.
    """
    df = df.groupby(level=0).max()  # one row per mouse gene
    mapping = map_mouse_to_human(df.index, table)
    rows = [(mouse, human) for mouse, hs in mapping.items() for human in hs]
    n_unmapped = sum(1 for hs in mapping.values() if not hs)
    logger.info(
        f"Ortholog mapping: {len(mapping) - n_unmapped}/{len(mapping)} mouse genes "
        f"have a human ortholog; the rest are excluded."
    )
    if not rows:
        raise ValueError(
            "No mouse genes could be mapped to human orthologs. Check that gene "
            "names are mouse symbols or Ensembl IDs (e.g. set gene_names_key)."
        )
    mouse_idx, human_idx = zip(*rows)
    expanded = df.loc[list(mouse_idx)].copy()
    expanded.index = pd.Index(human_idx, name=df.index.name)
    return expanded.groupby(level=0).max()
