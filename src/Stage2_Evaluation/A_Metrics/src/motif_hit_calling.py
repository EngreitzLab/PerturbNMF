"""Motif hit calling for program TF-motif enrichment: regions -> sequences -> FIMO-format hit tables.

Upstream of ``motif_enrichment.py`` (the statistics). Every hit table this module writes uses the
MEME FIMO tsv columns (:data:`FIMO_COLUMNS`), so ``motif_enrichment.read_fimo_hits`` reads them.

Regions
-------
* Promoters (``build_promoter_regions``): one per gene symbol; ``sequence_name`` = gene symbol.
    - ``strand_aware`` (default): TSS-``upstream`` .. TSS+``downstream`` in the direction of
      transcription, TSS base included (default 250 / 50 -> 301 bp).
    - ``schnitzler2024``: genome coordinates [TSS-250, TSS+51) ignoring strand. Reproduces the
      Schnitzler et al. 2024 region: they built TSS +/- 250 and then kept [start, start+301),
      which is upstream-heavy for + genes and downstream-heavy for - genes. TSS here is the BED
      ``start`` for + genes and the BED ``end`` for - genes, exactly as in their TSS500bp file.
      (The paper then lifted hg19 -> hg38 and wrote - gene sequences reverse-complemented; with the
      hg19 genome this mode gives the same sequences, so identical gene x TF counts, but hit
      start/stop/strand of - genes are mirrored relative to the paper's fimo.tsv.)
  TSS per gene: from a GTF, the TSS of the transcript tagged ``Ensembl_canonical`` if the gene has
  one, else the gene record start (+) / end (-). From a BED6 of gene bounds (e.g. ABC's
  ``RefSeqCurated.170308.bed.CollapsedGeneBounds.bed``): start (+) / end - 1 (-), the last base of
  the gene; in ``schnitzler2024`` mode the - gene TSS is the BED ``end`` (one past the last base), as
  in the paper's TSS500bp file.
  Genes sharing a symbol are collapsed to one promoter: prefer canonical-tagged, then
  protein_coding, then a primary chromosome (chr1-22, X, Y, M), then first in the file.
* Enhancers (``read_enhancer_gene_links`` + ``build_enhancer_regions``): one region per
  element-gene link; ``sequence_name`` = ``chrom:start-end|class|element_name|TargetGene``
  (4 ``|``-separated fields, same as the 2024 paper's ABC FASTA names, where the ABC ``name``
  column is itself ``class|element``). Parse with :func:`parse_enhancer_sequence_name` /
  :func:`target_gene_from_sequence_name`.

Hit calling
-----------
* ``scan_regions_with_fimo``: pyfaidx sequence extraction + FIMO. Backend ``meme`` (MEME suite
  ``fimo`` binary, used by the paper: MEME 5.3.3, ``--thresh 1e-4``, background from the motif
  file) or ``memelite`` (``memelite.fimo``; uniform background, no q-values). Unique genomic
  intervals are scanned once, in parallel chunks, and hits are expanded to every sequence_name that
  shares the interval. Hit ``start``/``stop`` are 1-based within the region, which is always in
  genome + orientation (FIMO scans both strands, so counts do not depend on orientation).
  ``meme_text_mode=True`` (default) runs ``fimo --text``: no ``--max-stored-scores`` pruning and an
  empty q-value column. ``False`` runs FIMO's default mode (q-values, per-motif store capped at
  ``--max-stored-scores`` 100,000 -- the paper's run used this mode).
* ``call_hits_from_finemo``: Fi-NeMo hit calls (``hits.tsv``) contained in the regions -> same
  table. p-value and q-value are NA (Fi-NeMo has no p-values); ``score`` = ``hit_coefficient`` by
  default. Downstream p-value filters must be disabled for these tables
  (``motif_enrichment.read_fimo_hits(path, pvalue_threshold=None)``).

Fi-NeMo motif names (ENCODE ChromBPNet / BPNet "sequence motifs" files)
-----------------------------------------------------------------------
ENCODE ships two tars per model (annotation set, e.g. ENCSR313RDW):
  * "sequence motifs instances": ``{counts,profile}/seq_motifs_instances.{head}.lambda_0p{6..9}/
    seq_motifs_instances.{head}.lambda_0p7.<ENCSR>.tsv`` (BPNet) or ``{head}/lambda_0.7/
    finemo.motif_hits.{head}.0.7.<ENCSR>.tsv.gz`` (ChromBPNet) -- Fi-NeMo hits; ``motif_name`` is the
    TF-MoDISco pattern id (``pos_patterns.<ENCSR>_<assay>_<target>_<biosample>_<model>_counts_pattern_3``).
    The same hit is repeated once per overlapping peak (different ``peak_id``); :func:`read_finemo_hits`
    drops the repeats.
  * "sequence motifs report": ``{head}/seq_motifs_report.{head}.fold_mean.<ENCSR>.html`` or
    ``{head}/tfmodisco.report.{head}.<ENCSR>.html`` -- the TF-MoDISco report; each pattern has a TOMTOM
    table (top 2-3 matches, Q-value) against a MotifCompendium-Database-Human release (names
    ``{TF family}_{n}``, e.g. ``KLF-SP_0``, ``GATA_2``; the ChromBPNet reports use the 2025-09 release,
    whose family stems are partly HOCOMOCO mnemonics such as ``NF2L-NFE_0``, ``ANDR_0``).
    The report writes spaces in the biosample as ``-``; :func:`normalize_finemo_pattern_id` makes the
    two files agree.
:func:`read_finemo_motif_annotation` parses the report; :func:`name_finemo_patterns` names each
pattern by its top TOMTOM match -- the database motif cluster (``KLF-SP_0``), whatever its q-value
(an optional q-value threshold keeps weaker patterns as a label such as ``pos-counts-pattern-3``);
``motif_family`` is the cluster name without ``_<n>`` (``KLF-SP``). :func:`build_finemo_motif_name_map`
turns that into the hit table's ``motif_id``; the enrichment tests one row per cluster (patterns with
the same top match are pooled, different clusters of one family are not), so counts must be read with
``motif_enrichment.read_fimo_hits(..., collapse_motif_ids=False)``.
Defaults (``FINEMO_DEFAULTS``): counts head, lambda 0.7, positive patterns only.

TF lists (MotifCompendium database metadata)
--------------------------------------------
Each database motif lists the TFs whose source motifs it merges (``KLF-SP_0`` -> KLF1..KLF17, MAZ,
PATZ1, SALL1, SALL4, SP1..SP9, VEZF1). Motif indices are not stable across database releases
(``KLF_3`` = KLF15 in the 2025-09 release, KLF10/KLF11 from 2026-01), so the metadata must be the
release the motifs were matched against. Two releases are bundled in ``motif_databases/`` (MIT
license, kundajelab/MotifCompendium): 2026-05-02 (commit 2ad26dc; the release of the default PFM file and
the default FIMO database) and 2025-09-24 (commit 5b20d47; the release the ENCODE ChromBPNet TF-MoDISco reports
were matched against). :func:`select_motifcompendium_metadata` picks the one that contains the most
of the given motif names (ties: newest). :func:`add_database_tfs_to_pattern_names` adds
the TF list of each pattern's top match.

Local Fi-NeMo tables (e.g. ``Slurm_Version/export_motif_hits_for_perturbnmf.py``)
---------------------------------------------------------------------------------
``motif_annotation.tsv`` (one row per motif: ``motif_id``, ``database_motif``, ``candidate_tfs``,
``posneg``) plus ``motif_hits_<dataset>.tsv.gz`` (``#chrom start end strand motif_id ... score``).
:func:`name_finemo_patterns_from_annotation_table` gives the same pattern-name table as the report path
(tf = ``database_motif``, family = it without ``_<n>``; TF list = ``candidate_tfs``).
"""

import glob
import gzip
import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIMO_COLUMNS = ["motif_id", "motif_alt_id", "sequence_name", "start", "stop", "strand",
                "score", "p-value", "q-value", "matched_sequence"]
REGION_COLUMNS = ["chrom", "start", "end", "strand", "sequence_name", "gene"]
PRIMARY_CHROMOSOMES = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]}
PROMOTER_WINDOW_MODES = ("strand_aware", "schnitzler2024")


def open_text(path: str):
    """Open plain or gzip text for reading."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


# ---------------------------------------------------------------------------
# Promoters
# ---------------------------------------------------------------------------

GTF_ATTRIBUTE_RE = re.compile(r'(\S+) "([^"]*)"')


def read_gtf_gene_tss(gtf_path: str, gene_types: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """One TSS per gene symbol from a GENCODE-style GTF (gzip ok).

    Returns chrom, tss, strand, gene, gene_id, gene_type, is_canonical_tss.
    ``tss`` is a 0-based genome coordinate of the first transcribed base.
    """
    genes, canonical_tss = [], {}
    with open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in ("gene", "transcript"):
                continue
            chrom, feature, start, end, strand, attributes = (
                fields[0], fields[2], int(fields[3]), int(fields[4]), fields[6], fields[8])
            if feature == "transcript" and 'tag "Ensembl_canonical"' not in attributes:
                continue
            attribute_values = dict(GTF_ATTRIBUTE_RE.findall(attributes))
            tss = start - 1 if strand == "+" else end - 1  # GTF is 1-based closed
            gene_id = attribute_values.get("gene_id")
            if feature == "transcript":
                canonical_tss.setdefault(gene_id, tss)
                continue
            genes.append((chrom, tss, strand, attribute_values.get("gene_name", gene_id), gene_id,
                          attribute_values.get("gene_type", attribute_values.get("gene_biotype", ""))))
    table = pd.DataFrame(genes, columns=["chrom", "tss", "strand", "gene", "gene_id", "gene_type"])
    table["is_canonical_tss"] = table["gene_id"].isin(canonical_tss)
    table.loc[table["is_canonical_tss"], "tss"] = table.loc[table["is_canonical_tss"], "gene_id"].map(canonical_tss)
    if gene_types is not None:
        table = table[table["gene_type"].isin(set(gene_types))]
    return collapse_genes_by_symbol(table)


def read_bed_gene_tss(bed_path: str, window_mode: str = "strand_aware") -> pd.DataFrame:
    """One TSS per gene symbol from a BED6 of gene bounds (chrom, start, end, name, score, strand).

    TSS = ``start`` for + genes. For - genes: ``end - 1`` (the first transcribed base, 0-based, same
    as :func:`read_gtf_gene_tss`) in ``strand_aware`` mode; ``end`` (one past the last base -- the
    convention of ABC's TSS files and the 2024 paper) in ``schnitzler2024`` mode, so that mode
    reproduces the paper's promoter windows exactly.
    """
    if window_mode not in PROMOTER_WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {PROMOTER_WINDOW_MODES}, got {window_mode!r}")
    bed = pd.read_csv(bed_path, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3, 5],
                      names=["chrom", "start", "end", "gene", "strand"], dtype={"chrom": str, "gene": str})
    minus_strand_tss = bed["end"] if window_mode == "schnitzler2024" else bed["end"] - 1
    table = pd.DataFrame({
        "chrom": bed["chrom"], "tss": np.where(bed["strand"] == "-", minus_strand_tss, bed["start"]),
        "strand": bed["strand"], "gene": bed["gene"], "gene_id": bed["gene"], "gene_type": "",
        "is_canonical_tss": False,
    })
    return collapse_genes_by_symbol(table)


def collapse_genes_by_symbol(table: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per gene symbol (canonical > protein_coding > primary chromosome > file order)."""
    rank = pd.DataFrame({
        "not_canonical": ~table["is_canonical_tss"].to_numpy(),
        "not_protein_coding": (table["gene_type"] != "protein_coding").to_numpy(),
        "not_primary": ~table["chrom"].isin(PRIMARY_CHROMOSOMES).to_numpy(),
        "file_order": np.arange(len(table)),
    }, index=table.index)
    order = rank.sort_values(list(rank.columns)).index
    collapsed = table.loc[order].drop_duplicates("gene", keep="first")
    return collapsed.sort_values("tss", kind="stable").sort_values("chrom", kind="stable").reset_index(drop=True)


def build_promoter_regions(gene_tss: pd.DataFrame, upstream: int = 250, downstream: int = 50,
                           window_mode: str = "strand_aware") -> pd.DataFrame:
    """Promoter window per gene (columns :data:`REGION_COLUMNS`, 0-based half-open, sequence_name = gene).

    ``strand_aware``: + genes [tss-upstream, tss+downstream+1); - genes [tss-downstream, tss+upstream+1).
    ``schnitzler2024``: [tss-250, tss+51) for every gene (upstream/downstream ignored).
    """
    if window_mode not in PROMOTER_WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {PROMOTER_WINDOW_MODES}, got {window_mode!r}")
    tss = gene_tss["tss"].to_numpy()
    if window_mode == "schnitzler2024":
        start, end = tss - 250, tss + 51
    else:
        minus = (gene_tss["strand"] == "-").to_numpy()
        start = np.where(minus, tss - downstream, tss - upstream)
        end = np.where(minus, tss + upstream + 1, tss + downstream + 1)
    return pd.DataFrame({
        "chrom": gene_tss["chrom"].to_numpy(), "start": np.maximum(start, 0), "end": end,
        "strand": gene_tss["strand"].to_numpy(), "sequence_name": gene_tss["gene"].to_numpy(),
        "gene": gene_tss["gene"].to_numpy(),
    })


# ---------------------------------------------------------------------------
# Enhancer-gene links
# ---------------------------------------------------------------------------

ABC_HEADERLESS_COLUMNS = [
    "chr", "start", "end", "name", "class", "activity_base", "TargetGene", "TargetGeneTSS",
    "TargetGeneExpression", "TargetGenePromoterActivityQuantile", "TargetGeneIsExpressed", "distance",
    "isSelfPromoter", "powerlaw_contact", "powerlaw_contact_reference", "hic_contact",
    "hic_contact_pl_scaled", "hic_pseudocount", "hic_contact_pl_scaled_adj", "ABC.Score.Numerator",
    "ABC.Score", "powerlaw.Score.Numerator", "powerlaw.Score", "CellType",
]
LINK_COLUMN_ALIASES = {
    "chrom": ["chr", "chrom", "chromosome", "ElementChr", "ChrElement", "chrElement"],
    "start": ["start", "ElementStart", "StartElement", "startElement"],
    "end": ["end", "ElementEnd", "EndElement", "endElement"],
    "name": ["name", "ElementName", "element_name"],
    "element_class": ["class", "ElementClass", "element_class"],
    "gene": ["TargetGene", "GeneSymbol", "TargetGeneSymbol", "gene_symbol", "gene"],
    "score": ["Score", "E2G.Score", "ENCODE-rE2G.Score", "ABC.Score", "score"],
}
LINK_FORMATS = ("auto", "tsv", "abc_headerless", "bedpe")
BEDPE_SELF_PROMOTER_WINDOW = 500
UNSAFE_NAME_RE = re.compile(r"[|\s]+")


def detect_link_format(path: str) -> str:
    """``bedpe`` if the file name says so; else ``tsv`` if the first line is a header, else ``abc_headerless``."""
    if ".bedpe" in os.path.basename(path):
        return "bedpe"
    with open_text(path) as handle:
        first = handle.readline().rstrip("\n").split("\t")
    try:
        int(first[1])
        return "abc_headerless"
    except (IndexError, ValueError):
        return "tsv"


def find_column(columns: Iterable[str], aliases: List[str]) -> Optional[str]:
    columns = list(columns)
    return next((alias for alias in aliases if alias in columns), None)


def read_enhancer_gene_links(path: str, link_format: str = "auto", score_threshold: Optional[float] = None,
                             score_column: Optional[str] = None, drop_promoters: bool = True) -> pd.DataFrame:
    """Read element-gene links into chrom, start, end, element_class, element_name, gene, score.

    Formats:
      * ``tsv`` -- headered table, columns found by alias (:data:`LINK_COLUMN_ALIASES`): ABC
        ``Predictions`` (score ``ABC.Score``), ENCODE-rE2G / scE2G (``Score``), IGVF element-gene tsv.
        An ABC ``name`` of the form ``class|element`` is split into class and element name.
      * ``abc_headerless`` -- ABC predictions grepped from ``CombinedPredictions`` (24 columns, no header).
      * ``bedpe`` -- IGVF / E2G bedpe: element chrom/start/end, TSS chrom/start/end,
        ``chrom:start-end_Gene``, score. No element class in the file: an element overlapping its target
        gene's TSS +/- ``BEDPE_SELF_PROMOTER_WINDOW`` (500 bp, ABC's promoter definition) gets class
        ``promoter``, every other element ``.``.
    ``score_threshold`` keeps score >= threshold. ``drop_promoters`` drops class ``promoter`` elements.
    """
    if link_format not in LINK_FORMATS:
        raise ValueError(f"link_format must be one of {LINK_FORMATS}, got {link_format!r}")
    if link_format == "auto":
        link_format = detect_link_format(path)

    if link_format == "bedpe":
        raw = pd.read_csv(path, sep="\t", header=None, comment="#", dtype={0: str, 6: str})
        element_and_gene = raw[6].str.split("_", n=1, expand=True)
        tss = raw[4].astype(np.int64)
        self_promoter = ((raw[2] > tss - BEDPE_SELF_PROMOTER_WINDOW) & (raw[1] < tss + BEDPE_SELF_PROMOTER_WINDOW)
                         & (raw[0] == raw[3]))
        links = pd.DataFrame({
            "chrom": raw[0], "start": raw[1], "end": raw[2],
            "element_class": np.where(self_promoter, "promoter", "."),
            "element_name": element_and_gene[0], "gene": element_and_gene[1],
            "score": pd.to_numeric(raw[7], errors="coerce"),
        })
    else:
        if link_format == "abc_headerless":
            raw = pd.read_csv(path, sep="\t", header=None, names=ABC_HEADERLESS_COLUMNS, dtype={"chr": str})
        else:
            raw = pd.read_csv(path, sep="\t", comment=None, dtype=str)
        column = {key: find_column(raw.columns, aliases) for key, aliases in LINK_COLUMN_ALIASES.items()}
        if score_column is not None:
            if score_column not in raw.columns:
                raise ValueError(f"{path}: score column {score_column!r} not found; columns are {list(raw.columns)}")
            column["score"] = score_column
        missing = [key for key in ("chrom", "start", "end", "gene") if column[key] is None]
        if missing:
            raise ValueError(f"{path}: could not find columns for {missing}; columns are {list(raw.columns)}")
        names = raw[column["name"]].astype(str) if column["name"] else None
        classes = raw[column["element_class"]].astype(str) if column["element_class"] else pd.Series(".", index=raw.index)
        if names is not None and names.str.contains("|", regex=False).all():
            split = names.str.split("|", n=1, expand=True)       # ABC: "genic|chr1:1-500"
            classes, names = split[0], split[1]
        links = pd.DataFrame({
            "chrom": raw[column["chrom"]].astype(str), "start": pd.to_numeric(raw[column["start"]]),
            "end": pd.to_numeric(raw[column["end"]]), "element_class": classes,
            "element_name": names if names is not None else None, "gene": raw[column["gene"]].astype(str),
            "score": pd.to_numeric(raw[column["score"]], errors="coerce") if column["score"] else np.nan,
        })
    links["start"] = links["start"].astype(np.int64)
    links["end"] = links["end"].astype(np.int64)
    coordinates = links["chrom"] + ":" + links["start"].astype(str) + "-" + links["end"].astype(str)
    links["element_name"] = links["element_name"].fillna(coordinates)
    if score_threshold is not None and links["score"].isna().all():
        raise ValueError(f"{path}: score_threshold={score_threshold} given but no score column was found "
                         f"(or every score is non-numeric); pass link_score_column / --link_score_column")
    if drop_promoters:
        links = links[links["element_class"] != "promoter"]
    if score_threshold is not None:
        links = links[links["score"] >= score_threshold]
    return links.reset_index(drop=True)



def build_enhancer_regions(links: pd.DataFrame) -> pd.DataFrame:
    """One region per link; sequence_name = ``chrom:start-end|class|element_name|TargetGene``."""
    def safe(values: pd.Series) -> pd.Series:
        return values.astype(str).str.replace(UNSAFE_NAME_RE, "_", regex=True)
    coordinates = links["chrom"] + ":" + links["start"].astype(str) + "-" + links["end"].astype(str)
    sequence_name = (coordinates + "|" + safe(links["element_class"]) + "|" + safe(links["element_name"])
                     + "|" + safe(links["gene"]))
    return pd.DataFrame({
        "chrom": links["chrom"].to_numpy(), "start": links["start"].to_numpy(), "end": links["end"].to_numpy(),
        "strand": ".", "sequence_name": sequence_name.to_numpy(), "gene": links["gene"].to_numpy(),
    })


def parse_enhancer_sequence_name(sequence_names: pd.Series) -> pd.DataFrame:
    """Split ``chrom:start-end|class|element_name|TargetGene`` into region, element_class, element_name, gene."""
    parts = pd.Series(sequence_names).astype(str).str.split("|", n=3, expand=True)
    if parts.shape[1] != 4:
        raise ValueError(f"expected 4 '|'-separated fields in enhancer sequence names, got {parts.shape[1]}")
    return pd.DataFrame({"region": parts[0].to_numpy(), "element_class": parts[1].to_numpy(),
                         "element_name": parts[2].to_numpy(), "gene": parts[3].to_numpy()})


def target_gene_from_sequence_name(sequence_names: pd.Series) -> pd.Series:
    """Target gene of each hit: last ``|`` field (enhancer names) or the whole name (promoter names)."""
    return pd.Series(sequence_names).astype(str).str.rsplit("|", n=1).str[-1]


# ---------------------------------------------------------------------------
# Sequences + FIMO
# ---------------------------------------------------------------------------

def region_scan_ids(regions: pd.DataFrame) -> pd.Series:
    return regions["chrom"].astype(str) + ":" + regions["start"].astype(str) + "-" + regions["end"].astype(str)


MAX_MISSING_INTERVAL_FRACTION = 0.05


def read_fasta_chromosome_lengths(genome_fasta: str) -> Dict[str, int]:
    """Chromosome -> length from the FASTA index (``.fai``, built by pyfaidx if absent)."""
    fai = genome_fasta + ".fai"
    if not os.path.exists(fai):
        import pyfaidx
        pyfaidx.Faidx(genome_fasta)
    index = pd.read_csv(fai, sep="\t", header=None, usecols=[0, 1], names=["chrom", "length"], dtype={"chrom": str})
    return dict(zip(index["chrom"], index["length"].astype(int)))


def match_chromosome_names(region_chromosomes: Iterable[str], fasta_chromosomes: Iterable[str]) -> Dict[str, str]:
    """Region chromosome -> FASTA chromosome, adding or stripping a ``chr`` prefix when that is what
    makes a region chromosome present in the FASTA (``1`` -> ``chr1``, ``chrM`` -> ``MT`` too).
    Chromosomes with no match are left out."""
    fasta_chromosomes = set(fasta_chromosomes)
    mapping = {}
    for chrom in set(region_chromosomes):
        stripped = chrom[3:] if chrom.startswith("chr") else chrom
        candidates = [chrom, stripped, "chr" + stripped]
        if stripped in ("M", "MT"):
            candidates += ["chrM", "MT", "M"]
        found = next((candidate for candidate in candidates if candidate in fasta_chromosomes), None)
        if found is not None:
            mapping[chrom] = found
    return mapping


def check_region_chromosomes(regions: pd.DataFrame, genome_fasta: str,
                             max_missing_fraction: float = MAX_MISSING_INTERVAL_FRACTION) -> Dict[str, str]:
    """Region chromosome -> FASTA chromosome (:func:`match_chromosome_names`); raises ValueError if more
    than ``max_missing_fraction`` of the unique intervals are on chromosomes absent from the FASTA
    (wrong genome or naming), else logs the skipped ones."""
    intervals = regions[["chrom", "start", "end"]].drop_duplicates()
    mapping = match_chromosome_names(intervals["chrom"], read_fasta_chromosome_lengths(genome_fasta))
    renamed = sorted(chrom for chrom, fasta_chrom in mapping.items() if chrom != fasta_chrom)
    if renamed:
        logger.info("region chromosomes renamed to match %s: %s", genome_fasta,
                    {chrom: mapping[chrom] for chrom in renamed[:10]})
    missing = intervals["chrom"][~intervals["chrom"].isin(mapping)]
    if len(intervals) and len(missing) / len(intervals) > max_missing_fraction:
        raise ValueError(
            f"{len(missing)} of {len(intervals)} region intervals ({len(missing) / len(intervals):.1%}) are on "
            f"chromosomes absent from {genome_fasta} (e.g. {sorted(set(missing))[:5]}); the genome FASTA "
            f"does not match the region coordinates (build or chromosome naming)")
    if len(missing):
        logger.warning("skipping %d intervals on %d chromosomes missing from %s: %s", len(missing),
                       missing.nunique(), genome_fasta, sorted(set(missing))[:10])
    return mapping


def write_region_fasta(regions: pd.DataFrame, genome_fasta: str, out_fasta: str,
                       chromosome_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Write one FASTA record per unique interval (id ``chrom:start-end``); returns the intervals written.

    ``chromosome_names`` maps region chromosomes to FASTA chromosomes (:func:`check_region_chromosomes`);
    record ids keep the region chromosome. Intervals on chromosomes absent from the genome are skipped
    (logged); ends are clipped to the chromosome length. Case (soft-masking) is kept, as bedtools
    getfasta does.
    """
    import pyfaidx
    genome = pyfaidx.Fasta(genome_fasta, as_raw=True, sequence_always_upper=False)
    intervals = regions[["chrom", "start", "end"]].drop_duplicates()
    if chromosome_names is None:
        chromosome_names = {chrom: chrom for chrom in set(intervals["chrom"]) if chrom in genome.keys()}
    missing = sorted(set(intervals["chrom"]) - set(chromosome_names))
    if missing:
        logger.warning("skipping %d intervals on %d chromosomes missing from %s: %s",
                       int(intervals["chrom"].isin(missing).sum()), len(missing), genome_fasta, missing[:10])
        intervals = intervals[~intervals["chrom"].isin(missing)]
    written = []
    with open(out_fasta, "w") as handle:
        for chrom, start, end in intervals.itertuples(index=False):
            sequence = genome[chromosome_names[chrom]][int(start):int(end)]
            if len(sequence) == 0:
                continue
            handle.write(f">{chrom}:{start}-{end}\n{sequence}\n")
            written.append((chrom, start, end))
    return pd.DataFrame(written, columns=["chrom", "start", "end"])


# ---------------------------------------------------------------------------
# Genome builds
# ---------------------------------------------------------------------------

GENOME_BUILD_ALIASES = {"hg19": "hg19", "grch37": "hg19", "hg38": "hg38", "grch38": "hg38"}
CHR1_LENGTH_TO_GENOME_BUILD = {249250621: "hg19", 248956422: "hg38"}
GENOME_BUILD_IN_NAME_RE = re.compile(r"(?<![a-z0-9])(hg19|hg38|grch37|grch38)(?![0-9])", re.I)


def normalize_genome_build(build: Optional[str]) -> Optional[str]:
    """``GRCh38`` / ``hg38`` -> ``hg38``, ``GRCh37`` / ``hg19`` -> ``hg19``; other names lowercased; empty -> None."""
    if build is None or str(build).strip() in ("", "nan", "None"):
        return None
    build = str(build).strip().lower()
    return GENOME_BUILD_ALIASES.get(build, build)


def infer_genome_build_from_fasta(genome_fasta: str) -> Optional[str]:
    """hg19 / hg38 from the length of chr1 (or ``1``) in the FASTA index; None if unknown or unreadable."""
    try:
        lengths = read_fasta_chromosome_lengths(genome_fasta)
    except (OSError, ValueError) as error:
        logger.warning("could not index %s to infer its genome build: %s", genome_fasta, error)
        return None
    chr1_length = lengths.get("chr1", lengths.get("1"))
    return CHR1_LENGTH_TO_GENOME_BUILD.get(chr1_length)


def infer_genome_build_from_path(path: Optional[str]) -> Optional[str]:
    """Build named in a file name (``...hg19...``, ``GRCh38``); None if absent or ambiguous."""
    if not path:
        return None
    builds = {normalize_genome_build(m) for m in GENOME_BUILD_IN_NAME_RE.findall(os.path.basename(str(path)))}
    return builds.pop() if len(builds) == 1 else None


def check_genome_builds(declared_build: str, observed_builds: Dict[str, Optional[str]]) -> None:
    """Raise ValueError if any known observed build (source -> build) differs from ``declared_build``."""
    declared = normalize_genome_build(declared_build)
    mismatched = {source: build for source, build in observed_builds.items()
                  if normalize_genome_build(build) not in (None, declared)}
    if mismatched:
        raise ValueError(f"genome build mismatch: declared {declared} but {mismatched}; region coordinates, "
                         f"genome FASTA and hit calls must all be on one build (set --genome_build / inputs)")


def run_meme_fimo(fasta: str, motif_file: str, threshold: float = 1e-4, fimo_binary: str = "fimo",
                  text_mode: bool = True, max_stored_scores: Optional[int] = None,
                  background_file: Optional[str] = None) -> pd.DataFrame:
    """Run MEME ``fimo`` on a FASTA; returns hits with :data:`FIMO_COLUMNS` (q-value empty in text mode)."""
    command = [fimo_binary, "--verbosity", "1", "--thresh", str(threshold)]
    if max_stored_scores is not None:
        command += ["--max-stored-scores", str(max_stored_scores)]
    if background_file is not None:
        command += ["--bgfile", background_file]
    with tempfile.TemporaryDirectory() as output_dir:
        if text_mode:
            tsv = os.path.join(output_dir, "fimo.tsv")
            with open(tsv, "w") as handle:
                subprocess.run(command + ["--text", motif_file, fasta], stdout=handle, check=True)
        else:
            subprocess.run(command + ["--oc", output_dir, motif_file, fasta], check=True)
            tsv = os.path.join(output_dir, "fimo.tsv")
        hits = pd.read_csv(tsv, sep="\t", comment="#", dtype={"motif_id": str, "motif_alt_id": str,
                                                             "sequence_name": str, "q-value": object},
                           keep_default_na=False, na_values={"p-value": [""], "score": [""]})
    return hits[FIMO_COLUMNS]


def run_memelite_fimo(fasta: str, motif_file: str, threshold: float = 1e-4) -> pd.DataFrame:
    """Run ``memelite.fimo`` on a FASTA; returns hits with :data:`FIMO_COLUMNS` (q-value empty)."""
    from memelite import fimo
    from memelite.io import read_meme
    import pyfaidx
    # memelite's read_meme drops the last motif unless a line follows its matrix: parse a padded copy
    with open(motif_file) as handle, tempfile.NamedTemporaryFile("w", suffix=".meme", delete=False) as padded:
        padded.write(handle.read() + "\n\n")
    try:
        motifs = read_meme(padded.name)
    finally:
        os.remove(padded.name)
    hits = pd.concat(fimo(motifs, fasta, threshold=threshold), ignore_index=True)
    if hits.empty:
        return pd.DataFrame(columns=FIMO_COLUMNS)
    sequences = pyfaidx.Fasta(fasta, as_raw=True, sequence_always_upper=False)
    starts = hits["start"].to_numpy(dtype=np.int64)    # memelite: 0-based start, exclusive end
    ends = hits["end"].to_numpy(dtype=np.int64)
    matched = [sequences[name][int(s):int(e)] for name, s, e in zip(hits["sequence_name"], starts, ends)]
    motif_fields = hits["motif_name"].astype(str).str.split(" ", n=1)
    return pd.DataFrame({
        "motif_id": motif_fields.str[0], "motif_alt_id": motif_fields.str[1].fillna(""),
        "sequence_name": hits["sequence_name"].astype(str),
        "start": starts + 1, "stop": ends, "strand": hits["strand"], "score": hits["score"],
        "p-value": hits["p-value"], "q-value": "", "matched_sequence": matched,
    })


def meme_fimo_available(fimo_binary: str = "fimo") -> bool:
    return shutil.which(fimo_binary) is not None


def resolve_fimo_backend(backend: str, fimo_binary: str = "fimo") -> str:
    """``auto`` -> ``meme`` if the fimo binary is found, else ``memelite``; validates the name."""
    if backend == "auto":
        backend = "meme" if meme_fimo_available(fimo_binary) else "memelite"
    if backend not in ("meme", "memelite"):
        raise ValueError(f"backend must be meme, memelite or auto, got {backend!r}")
    return backend


def describe_fimo_backend(backend: str, fimo_binary: str = "fimo") -> dict:
    """Resolved backend + tool identity for cache keys: fimo binary path and ``fimo --version`` output
    (meme), or the memelite package version."""
    backend = resolve_fimo_backend(backend, fimo_binary)
    if backend == "meme":
        binary_path = shutil.which(fimo_binary)
        version = subprocess.run([binary_path, "--version"], capture_output=True, text=True, check=False)
        return {"fimo_backend": backend, "fimo_binary_path": os.path.realpath(binary_path),
                "fimo_version": (version.stdout or version.stderr).strip()}
    from importlib.metadata import PackageNotFoundError, version as package_version
    try:
        memelite_version = package_version("memelite")
    except PackageNotFoundError:
        memelite_version = None
    return {"fimo_backend": backend, "memelite_version": memelite_version}


def scan_regions_with_fimo(regions: pd.DataFrame, genome_fasta: str, motif_file: str, out_tsv: str,
                           backend: str = "auto", threshold: float = 1e-4, fimo_binary: str = "fimo",
                           meme_text_mode: bool = True, max_stored_scores: Optional[int] = None,
                           background_file: Optional[str] = None, n_chunks: int = 1, n_jobs: int = 1,
                           work_dir: Optional[str] = None) -> int:
    """Scan regions with FIMO and write a FIMO-format tsv keyed by ``sequence_name``; returns #hit rows.

    ``backend``: ``meme`` (MEME fimo binary), ``memelite``, or ``auto`` (:func:`resolve_fimo_backend`).
    Unique intervals are split into ``n_chunks`` chunks scanned by ``n_jobs`` parallel workers (MEME
    only; memelite parallelises over motifs internally, so chunks run serially).
    Raises ValueError if > 5% of intervals are on chromosomes missing from the FASTA
    (:func:`check_region_chromosomes`; ``chr`` prefix differences are resolved), RuntimeError on 0 hits.
    """
    backend = resolve_fimo_backend(backend, fimo_binary)
    logger.info("FIMO backend: %s", backend)
    chromosome_names = check_region_chromosomes(regions, genome_fasta)
    names_by_scan_id = pd.DataFrame({"scan_id": region_scan_ids(regions).to_numpy(),
                                     "region_sequence_name": regions["sequence_name"].to_numpy()})
    own_work_dir = work_dir is None
    work_dir = tempfile.mkdtemp(prefix="motif_hits_") if own_work_dir else work_dir
    os.makedirs(work_dir, exist_ok=True)
    try:
        intervals = regions[["chrom", "start", "end"]].drop_duplicates().reset_index(drop=True)
        chunk_fastas = []
        for index, chunk in enumerate(np.array_split(np.arange(len(intervals)), max(1, min(n_chunks, len(intervals))))):
            chunk_fasta = os.path.join(work_dir, f"chunk_{index:04d}.fa")
            write_region_fasta(intervals.iloc[chunk], genome_fasta, chunk_fasta, chromosome_names)
            chunk_fastas.append(chunk_fasta)

        def scan(chunk_fasta: str) -> pd.DataFrame:
            if backend == "meme":
                return run_meme_fimo(chunk_fasta, motif_file, threshold, fimo_binary, meme_text_mode,
                                     max_stored_scores, background_file)
            return run_memelite_fimo(chunk_fasta, motif_file, threshold)

        n_hits = 0
        with open(out_tsv, "w") as handle:
            handle.write("\t".join(FIMO_COLUMNS) + "\n")
            workers = n_jobs if backend == "meme" else 1
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for hits in pool.map(scan, chunk_fastas):
                    expanded = hits.rename(columns={"sequence_name": "scan_id"}).merge(names_by_scan_id, on="scan_id")
                    expanded = expanded.rename(columns={"region_sequence_name": "sequence_name"})[FIMO_COLUMNS]
                    expanded.to_csv(handle, sep="\t", header=False, index=False)
                    n_hits += len(expanded)
                    logger.info("chunk done: %d hits (%d total)", len(expanded), n_hits)
    finally:
        if own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
    if n_hits == 0 and len(intervals):
        raise RuntimeError(f"FIMO found 0 hits in {len(intervals)} intervals of {genome_fasta} with {motif_file}: "
                           f"check the genome FASTA, motif file and threshold ({threshold})")
    return n_hits


# ---------------------------------------------------------------------------
# Fi-NeMo hit calls
# ---------------------------------------------------------------------------

FINEMO_COLUMN_ALIASES = {
    "chrom": ["chr", "chrom"], "start": ["start"], "end": ["end"],
    "motif_name": ["motif_name", "motif", "name", "motif_id"], "strand": ["strand"],
}


FINEMO_PATTERN_PREFIXES = ("pos_patterns", "neg_patterns")
FINEMO_DEFAULTS = {"head": "counts", "lambda_value": 0.7, "pattern_prefixes": ("pos_patterns",),
                   "qvalue_threshold": None}


def read_finemo_hits(path: str, score_column: str = "hit_coefficient",
                     pattern_prefixes: Optional[Iterable[str]] = None, deduplicate: bool = True) -> pd.DataFrame:
    """Read Fi-NeMo ``hits.tsv`` (headered, gzip ok) or a headerless BED6 (chrom, start, end, motif, score, strand).

    Parameters
    ----------
    score_column : column copied into ``score`` (Fi-NeMo ``hit_coefficient`` by default).
    pattern_prefixes : keep only motif names starting with one of these (e.g. ``("pos_patterns",)``);
        None keeps all.
    deduplicate : drop repeated (chrom, start, end, motif_name, strand) rows -- ENCODE's instance
        tables repeat a hit once per overlapping peak; the first row (file order) is kept.

    A headered table may start with ``#`` (``#chrom``, tabix style, as written by
    ``export_motif_hits_for_perturbnmf.py``); its ``motif_id`` column is used when there is no
    ``motif_name`` and its ``score`` column when ``score_column`` is absent.

    Returns chrom, start, end, motif_name, strand, score (0-based half-open coordinates).
    """
    with open_text(path) as handle:
        first = handle.readline().rstrip("\n").split("\t")
    first[0] = first[0].lstrip("#")
    if first[0] in FINEMO_COLUMN_ALIASES["chrom"]:
        raw = pd.read_csv(path, sep="\t", header=0, names=first, dtype={first[0]: str})
        column = {key: find_column(raw.columns, aliases) for key, aliases in FINEMO_COLUMN_ALIASES.items()}
        missing = [key for key, value in column.items() if value is None]
        if missing:
            raise ValueError(f"{path}: missing Fi-NeMo columns {missing}; columns are {list(raw.columns)}")
        score_source = score_column if score_column in raw.columns else ("score" if "score" in raw.columns else None)
        score = raw[score_source] if score_source else np.nan
        hits = pd.DataFrame({key: raw[value] for key, value in column.items()})
        hits["score"] = score
    else:
        hits = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5],
                           names=["chrom", "start", "end", "motif_name", "score", "strand"], dtype={"chrom": str})
    hits = hits[["chrom", "start", "end", "motif_name", "strand", "score"]]
    if pattern_prefixes is not None:
        hits = hits[hits["motif_name"].astype(str).str.startswith(tuple(pattern_prefixes))]
    if deduplicate:
        hits = hits.drop_duplicates(["chrom", "start", "end", "motif_name", "strand"], keep="first")
    return hits.reset_index(drop=True)


def normalize_finemo_pattern_id(pattern_id: str) -> str:
    """Pattern id with whitespace runs as ``-`` (the TF-MoDISco report writes ``smooth muscle cell`` as ``smooth-muscle-cell``)."""
    return re.sub(r"\s+", "-", str(pattern_id).strip())


def find_finemo_instances_file(root: str, head: str = "counts", lambda_value: float = 0.7) -> str:
    """Path of the Fi-NeMo instance tsv for ``head`` and ``lambda_value`` inside an extracted ENCODE
    "sequence motifs instances" tar. Two layouts exist:
    BPNet (ChIP): ``{head}/seq_motifs_instances.{head}.lambda_0p7/seq_motifs_instances.{head}.lambda_0p7.<ENCSR>.tsv``;
    ChromBPNet (DNase/ATAC): ``{head}/lambda_0.7/finemo.motif_hits.{head}.0.7.<ENCSR>.tsv.gz``."""
    tags = {f"{lambda_value:g}".replace(".", "p"), f"{lambda_value:g}"}
    patterns = []
    for tag in tags:
        for extension in ("tsv", "tsv.gz"):
            patterns.append(os.path.join(root, "**", f"seq_motifs_instances.{head}.lambda_{tag}", f"*.{extension}"))
            patterns.append(os.path.join(root, "**", head, f"lambda_{tag}", f"*.{extension}"))
    matches = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {head} lambda {lambda_value:g} instance tsv under {root}, found {matches}")
    return matches[0]


def find_finemo_report_file(root: str, head: str = "counts") -> str:
    """Path of the TF-MoDISco report html for ``head`` inside an extracted "sequence motifs report" tar
    (``seq_motifs_report.{head}.*.html`` or ``tfmodisco.report.{head}.*.html``)."""
    patterns = [os.path.join(root, "**", f"seq_motifs_report.{head}.*.html"),
                os.path.join(root, "**", f"tfmodisco.report.{head}.*.html")]
    matches = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {head} report html under {root}, found {matches}")
    return matches[0]


PATTERN_SECTION_RE = re.compile(r'<div class="pattern-section"[^>]*>(.*?)(?=<div class="pattern-section"|</body>)', re.S)
PATTERN_ID_RE = re.compile(r'<div class="pattern-title">.*?<small>\((.*?)\)</small>', re.S)
TOMTOM_TABLE_RE = re.compile(r'<table class="tomtom-table">(.*?)</table>', re.S)
TOMTOM_ROW_RE = re.compile(
    r'<tr>\s*<td class="num[_-]col">\s*(\d+)\s*</td>\s*<td>\s*(?:<code>)?(.*?)(?:</code>)?\s*</td>.*?'
    r'<td class="num[_-]col">\s*([^<\s]+)\s*</td>\s*</tr>', re.S)
MOTIF_ANNOTATION_COLUMNS = ["pattern_id", "match_rank", "match", "qvalue"]


def read_finemo_motif_annotation(report_html: str) -> pd.DataFrame:
    """TOMTOM matches of every TF-MoDISco pattern in an ENCODE "sequence motifs report" html.

    Returns
    -------
    DataFrame with columns pattern_id (normalized, see :func:`normalize_finemo_pattern_id`),
    match_rank (1 = best), match (reference motif name, e.g. ``KLF-SP_0``), qvalue. Patterns with no
    TOMTOM table get one row with match_rank 0, match NaN and qvalue NaN.
    """
    with open(report_html) as handle:
        text = re.sub(r'src="data:[^"]*"', 'src=""', handle.read())   # drop embedded logos
    rows = []
    for section in PATTERN_SECTION_RE.finditer(text):
        body = section.group(1)
        pattern_match = PATTERN_ID_RE.search(body)
        if pattern_match is None:
            continue
        pattern_id = normalize_finemo_pattern_id(html.unescape(pattern_match.group(1)))
        table = TOMTOM_TABLE_RE.search(body)
        matches = TOMTOM_ROW_RE.findall(table.group(1)) if table else []
        if not matches:
            rows.append((pattern_id, 0, np.nan, np.nan))
        for rank, name, qvalue in matches:
            rows.append((pattern_id, int(rank), html.unescape(name).strip(), float(qvalue)))
    if not rows:
        raise ValueError(f"{report_html}: no TF-MoDISco pattern sections found")
    return pd.DataFrame(rows, columns=MOTIF_ANNOTATION_COLUMNS)


def collapse_motifcompendium_name(name: str) -> str:
    """MotifCompendium motif name -> TF family: drop a trailing ``_<number>`` (``KLF-SP_0`` -> ``KLF-SP``)."""
    return re.sub(r"_\d+$", "", str(name))


def finemo_pattern_label(pattern_id: str) -> str:
    """Short label without ``_`` for an unnamed pattern: ``pos_patterns...._counts_pattern_3`` -> ``pos-counts-pattern-3``."""
    polarity = "neg" if str(pattern_id).startswith("neg") else "pos"
    tail = re.search(r"(counts|profile)?_?pattern_(\d+)$", str(pattern_id))
    if tail is None:
        return f"{polarity}-" + re.sub(r"[_\s]+", "-", str(pattern_id))
    return f"{polarity}-{tail.group(1) + '-' if tail.group(1) else ''}pattern-{tail.group(2)}"


PATTERN_NAME_COLUMNS = ["pattern_id", "tf", "motif_family", "top_match", "top_match_qvalue", "is_named",
                        "all_matches"]


def name_finemo_patterns(annotation: pd.DataFrame, qvalue_threshold: Optional[float] = None) -> pd.DataFrame:
    """One name per pattern: its top TOMTOM match (database motif cluster, e.g. ``KLF-SP_0``), whatever
    its q-value (``qvalue_threshold`` None, the default). With a threshold, only matches with
    q < ``qvalue_threshold`` name a pattern; the others, and patterns without any match, keep the
    pattern label (:func:`finemo_pattern_label`).

    Returns :data:`PATTERN_NAME_COLUMNS`: pattern_id, tf (the test unit in the enrichment tables: the
    cluster name, or the pattern label), motif_family (cluster name without ``_<n>``: ``KLF-SP``; the
    label for unnamed patterns), top_match, top_match_qvalue, is_named (bool), all_matches
    (``;``-joined ``match(q)``).
    """
    rows = []
    for pattern_id, group in annotation.sort_values(["pattern_id", "match_rank"]).groupby("pattern_id", sort=False):
        matched = group[group["match"].notna()]
        top = matched.iloc[0] if len(matched) else None
        is_named = top is not None and (qvalue_threshold is None or top["qvalue"] < qvalue_threshold)
        name = str(top["match"]) if is_named else finemo_pattern_label(pattern_id)
        rows.append({
            "pattern_id": pattern_id, "tf": name,
            "motif_family": collapse_motifcompendium_name(name) if is_named else name,
            "top_match": top["match"] if top is not None else np.nan,
            "top_match_qvalue": top["qvalue"] if top is not None else np.nan,
            "is_named": bool(is_named),
            "all_matches": ";".join(f"{m}({q:.2g})" for m, q in zip(matched["match"], matched["qvalue"])),
        })
    return pd.DataFrame(rows, columns=PATTERN_NAME_COLUMNS)


# ---------------------------------------------------------------------------
# MotifCompendium database metadata: database motif -> TF list
# ---------------------------------------------------------------------------

MOTIF_DATABASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motif_databases")
MOTIFCOMPENDIUM_METADATA_GLOB = "MotifCompendium-Database-Human.metadata.*.tsv"
TF_LIST_SEPARATOR_RE = re.compile(r"[,;@]")


def bundled_motifcompendium_metadata_paths() -> List[str]:
    """Bundled MotifCompendium metadata releases, oldest first (file names carry the release date)."""
    return sorted(glob.glob(os.path.join(MOTIF_DATABASES_DIR, MOTIFCOMPENDIUM_METADATA_GLOB)))


def split_tf_list(tfs) -> List[str]:
    """``"KLF1,KLF2"`` -> ``["KLF1", "KLF2"]``; NaN / empty -> []."""
    if not isinstance(tfs, str):
        return []
    return [tf.strip() for tf in TF_LIST_SEPARATOR_RE.split(tfs) if tf.strip()]


def read_motifcompendium_metadata(path: str) -> pd.DataFrame:
    """MotifCompendium-Database-Human metadata -> name, motif_family (name without ``_<n>``), database_tfs
    (comma-joined TF names as in the database: gene symbols, some HOCOMOCO mnemonics), readable_name."""
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = {"name", "TF"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing MotifCompendium metadata columns {sorted(missing)}")
    return pd.DataFrame({
        "name": table["name"],
        "motif_family": table["name"].map(collapse_motifcompendium_name),
        "database_tfs": table["TF"].map(lambda tfs: ",".join(split_tf_list(tfs))),
        "readable_name": table["readable_name"] if "readable_name" in table.columns else "",
    })


def select_motifcompendium_metadata(match_names: Iterable[str], paths: Optional[List[str]] = None) -> str:
    """The metadata release (bundled by default) containing the most of ``match_names`` (ties: newest).

    Database motif indices change between releases, so the TF list of ``KLF_3`` is only right in
    the release the TOMTOM matches were made against."""
    paths = bundled_motifcompendium_metadata_paths() if paths is None else list(paths)
    if not paths:
        raise FileNotFoundError(f"no MotifCompendium metadata in {MOTIF_DATABASES_DIR}")
    names = {str(name) for name in match_names if isinstance(name, str)}
    coverage = [(len(names & set(pd.read_csv(path, sep="\t", usecols=["name"], dtype=str)["name"])), index, path)
                for index, path in enumerate(paths)]
    n_found, _, best = max(coverage)
    if names and n_found < len(names):
        logger.warning("MotifCompendium metadata %s has %d of %d motif names; missing ones get no TF list",
                       os.path.basename(best), n_found, len(names))
    return best


def add_database_tfs_to_pattern_names(pattern_names: pd.DataFrame, metadata_path: Optional[str] = None) -> pd.DataFrame:
    """Add ``database_tfs`` (TF list of the pattern's top match, comma-joined; empty for unnamed patterns or
    names missing from the metadata) and ``motifcompendium_metadata`` (file name of the release used).
    ``metadata_path`` None picks the bundled release by :func:`select_motifcompendium_metadata`."""
    if metadata_path is None:
        metadata_path = select_motifcompendium_metadata(pattern_names["top_match"])
    tf_lists = read_motifcompendium_metadata(metadata_path).set_index("name")["database_tfs"]
    named = pattern_names["is_named"].astype(bool)
    return pattern_names.assign(
        database_tfs=pattern_names["top_match"].map(tf_lists).where(named, "").fillna(""),
        motifcompendium_metadata=os.path.basename(metadata_path))


LOCAL_ANNOTATION_COLUMNS = {"motif_id", "database_motif", "candidate_tfs"}


def name_finemo_patterns_from_annotation_table(path: str, pattern_prefixes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Pattern names from a local ``motif_annotation.tsv`` (``export_motif_hits_for_perturbnmf.py``).

    tf = ``database_motif`` (cluster, e.g. ``KLF-SP_0``), motif_family = it without ``_<n>`` (``KLF-SP``),
    TF list = ``candidate_tfs`` (MotifCompendium database TFs of that match). Motifs without a database match keep ``motif_id``
    with ``_`` replaced by ``-`` (``cluster_12`` -> ``cluster-12``). ``pattern_prefixes``
    (``pos_patterns`` / ``neg_patterns``) keep motifs whose ``posneg`` column is ``pos`` / ``neg``.

    Returns :data:`PATTERN_NAME_COLUMNS` + database_tfs, motifcompendium_metadata (``""``); pattern_id =
    ``motif_id``, top_match_qvalue NaN (the table has a match score, kept as top_match_score).
    """
    table = pd.read_csv(path, sep="\t", dtype={"motif_id": str, "database_motif": str, "candidate_tfs": str})
    missing = LOCAL_ANNOTATION_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing motif annotation columns {sorted(missing)}")
    if pattern_prefixes is not None and "posneg" in table.columns:
        wanted = {prefix.split("_")[0] for prefix in pattern_prefixes}
        table = table[table["posneg"].astype(str).isin(wanted)]
    is_named = table["database_motif"].notna() & (table["database_motif"].astype(str).str.strip() != "")
    names = np.where(is_named, table["database_motif"].astype(str),
                     table["motif_id"].str.replace("_", "-", regex=False))
    families = np.where(is_named, table["database_motif"].map(collapse_motifcompendium_name), names)
    score = table["database_match_score"] if "database_match_score" in table.columns else np.nan
    return pd.DataFrame({
        "pattern_id": table["motif_id"].to_numpy(), "tf": names, "motif_family": families,
        "top_match": table["database_motif"].where(is_named).to_numpy(), "top_match_qvalue": np.nan,
        "is_named": is_named.to_numpy(),
        "all_matches": table["database_motif"].where(is_named, "").fillna("").to_numpy(),
        "database_tfs": table["candidate_tfs"].map(lambda tfs: ",".join(split_tf_list(tfs))).to_numpy(),
        "motifcompendium_metadata": "",
        "top_match_score": score if np.isscalar(score) else score.to_numpy(),
    }).reset_index(drop=True)


def build_finemo_motif_name_map(pattern_names: pd.DataFrame) -> Dict[str, str]:
    """Pattern id -> motif_id of the hit table = the pattern's ``tf`` (database cluster such as
    ``KLF-SP_0``, or the pattern label). Read the table with ``collapse_motif_ids=False`` so the
    cluster id stays the test unit."""
    return dict(zip(pattern_names["pattern_id"], pattern_names["tf"]))


def intersect_hits_with_regions(hits: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Pairs (hit, region) with the hit fully inside the region. Returns hit columns + region_index."""
    pairs = []
    for chrom, region_group in regions.groupby("chrom", sort=False):
        hit_group = hits[hits["chrom"] == chrom]
        if hit_group.empty:
            continue
        order = np.argsort(region_group["start"].to_numpy(), kind="stable")
        region_starts = region_group["start"].to_numpy()[order]
        region_ends = region_group["end"].to_numpy()[order]
        region_index = region_group.index.to_numpy()[order]
        max_length = int((region_ends - region_starts).max())
        hit_starts, hit_ends = hit_group["start"].to_numpy(), hit_group["end"].to_numpy()
        # candidate regions start in [hit_end - max_length, hit_start]
        low = np.searchsorted(region_starts, hit_ends - max_length, side="left")
        high = np.searchsorted(region_starts, hit_starts, side="right")
        counts = np.maximum(high - low, 0)
        hit_positions = np.repeat(np.arange(len(hit_group)), counts)
        offset_within_hit = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        candidate = np.repeat(low, counts) + offset_within_hit
        inside = region_ends[candidate] >= hit_ends[hit_positions]
        pairs.append(pd.DataFrame({"hit_row": hit_group.index.to_numpy()[hit_positions[inside]],
                                   "region_index": region_index[candidate[inside]]}))
    if not pairs:
        return hits.iloc[0:0].assign(region_index=pd.Series(dtype=int))
    pairs = pd.concat(pairs, ignore_index=True)
    return hits.loc[pairs["hit_row"]].reset_index(drop=True).assign(region_index=pairs["region_index"].to_numpy())


def call_hits_from_finemo(finemo_hits: pd.DataFrame, regions: pd.DataFrame,
                          motif_name_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Fi-NeMo hits inside regions -> FIMO-format table (p-value / q-value NA, matched_sequence empty).

    ``start``/``stop`` are 1-based within the region (genome + orientation). ``motif_name_map``
    renames Fi-NeMo motif names (e.g. MoDISco patterns -> ``TF_...`` ids, see
    :func:`build_finemo_motif_name_map`) before they become motif_id; keys and names are compared after
    :func:`normalize_finemo_pattern_id`. TF-MoDISco pattern ids (``pos_patterns.`` / ``neg_patterns.``)
    missing from the map become :func:`finemo_pattern_label` (no ``_``), so
    ``motif_enrichment.collapse_motif_to_tf`` keeps them apart instead of collapsing all to ``pos``;
    other names are kept as is. The original pattern id goes to ``motif_alt_id``.
    """
    regions = regions.reset_index(drop=True)
    inside = intersect_hits_with_regions(finemo_hits, regions)
    region = regions.loc[inside["region_index"]].reset_index(drop=True)
    normalized_map = {normalize_finemo_pattern_id(k): v for k, v in (motif_name_map or {}).items()}
    unmapped_patterns = set()

    def motif_id_of(name: str) -> str:
        mapped = normalized_map.get(normalize_finemo_pattern_id(name))
        if mapped is not None:
            return mapped
        if name.startswith(FINEMO_PATTERN_PREFIXES):
            unmapped_patterns.add(name)
            return finemo_pattern_label(name)
        return name

    names = inside["motif_name"].astype(str)
    motif_id = names.map({name: motif_id_of(name) for name in names.unique()})
    if unmapped_patterns:
        logger.warning("%d Fi-NeMo patterns not in the motif name map; named by pattern label (e.g. %s -> %s)",
                       len(unmapped_patterns), sorted(unmapped_patterns)[0],
                       finemo_pattern_label(sorted(unmapped_patterns)[0]))
    return pd.DataFrame({
        "motif_id": motif_id.to_numpy(), "motif_alt_id": inside["motif_name"].astype(str).to_numpy(),
        "sequence_name": region["sequence_name"].to_numpy(),
        "start": (inside["start"] - region["start"] + 1).to_numpy(), "stop": (inside["end"] - region["start"]).to_numpy(),
        "strand": inside["strand"].to_numpy(), "score": inside["score"].to_numpy(),
        "p-value": np.nan, "q-value": np.nan, "matched_sequence": "",
    })[FIMO_COLUMNS]
