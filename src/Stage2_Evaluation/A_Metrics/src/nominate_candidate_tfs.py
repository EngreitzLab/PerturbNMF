"""Nominate candidate regulator TFs for each program from motif enrichment + expression + knockdown.

A TF whose motif is enriched near a program's genes is a stronger candidate when (a) the TF gene is
expressed and loads on the same program, and (b) knocking the TF down changes the program. This module
joins the three evidence types into one table, one row per (program, element_type, enriched motif TF,
TF gene symbol). For HOCOMOCO v11 every motif TF maps to exactly one gene, so that is one row per
enriched TF; a JASPAR/finemo dimer name such as ``FOS::JUN`` gives one row per partner gene.

Motif TF names -> gene symbols
    HOCOMOCO v11 motif ids (``ZN148_HUMAN.H11MO.0.C``) collapse to the UniProt entry mnemonic
    (``ZN148``), which is often not the HGNC symbol (ZN148 -> ZNF148, P63 -> TP63, ANDR -> AR).
    The official annotation table ``HOCOMOCOv11_full_annotation_HUMAN_mono.tsv`` (downloaded from
    hocomoco11.autosome.org/downloads, bundled in ``motif_databases/``) gives the TF gene symbol.
    Names absent from the table (JASPAR / finemo / MotifCompendium names that already are gene
    symbols) pass through unchanged, split on ``::`` for dimers.
    Fi-NeMo patterns named from the MotifCompendium database carry TF *family* names (``KLF-SP``,
    ``ETV-ELF-NFAT-ELK``, ``GATA``); :func:`map_motif_families_to_gene_symbols` maps each family to the
    expressed genes on the database's own TF list of the matched motif(s) (one row per gene, source
    ``motifcompendium``). No gene-name heuristics: ``GATA_0`` gives GATA1-6, GATAD2A, TAL1, TRPS1, ZFPM1.

Motif families (``motif_family`` column of the Stage 2 tables)
    FIMO / HOCOMOCO v11: the TFClass family of the TF (``TF family`` column of the HOCOMOCO annotation,
    without the ``{2.3.1}`` code, e.g. ``Three-zinc finger Krüppel-related factors``;
    :func:`read_hocomoco_tf_families`). Fi-NeMo: the database motif family (``KLF-SP``, ``GATA``).

Evidence tiers (first that applies)
    motif+regulator             TF knockdown significantly changes the program
    motif+expressed_in_program  TF gene is among the program's top ``n_top_genes`` genes
    motif+expressed             TF gene is in the cNMF gene universe (expressed) but not a top gene
    motif_only                  TF gene not in the gene universe
"""

import os
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

HOCOMOCO_V11_ANNOTATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "motif_databases", "HOCOMOCOv11_full_annotation_HUMAN_mono.tsv")

CANDIDATE_TF_COLUMNS = [
    "program", "element_type", "tf", "tf_gene_symbol", "tf_gene_symbol_source",
    "pvalue", "fdr", "enrichment",
    "tf_expressed", "tf_program_loading_rank", "tf_in_top_program_genes",
    "tf_perturbed", "knockdown_log2fc", "knockdown_fdr", "tf_knockdown_regulates_program",
    "evidence_tier",
]

EVIDENCE_TIER_ORDER = ["motif+regulator", "motif+expressed_in_program", "motif+expressed", "motif_only"]


# ---------------------------------------------------------------------------
# Motif TF name -> gene symbol
# ---------------------------------------------------------------------------

def read_hocomoco_tf_gene_symbols(path: str = HOCOMOCO_V11_ANNOTATION_PATH) -> pd.DataFrame:
    """HOCOMOCO v11 TF name (motif id before '_', e.g. ``ZN148``) -> TF gene symbol (``ZNF148``).

    Parameters
    ----------
    path : HOCOMOCO v11 annotation tsv (columns ``Model`` and ``Transcription factor``).

    Returns
    -------
    DataFrame with columns tf, tf_gene_symbol (one row per TF name).
    """
    annotation = pd.read_csv(path, sep="\t", usecols=["Model", "Transcription factor"], dtype=str)
    table = pd.DataFrame({
        "tf": annotation["Model"].str.split("_", n=1).str[0],
        "tf_gene_symbol": annotation["Transcription factor"].str.strip(),
    }).drop_duplicates()
    conflicting = table["tf"][table["tf"].duplicated()].unique()
    if len(conflicting):
        raise ValueError(f"HOCOMOCO TF names mapping to more than one gene: {list(conflicting)[:10]}")
    return table.reset_index(drop=True)


def map_tf_names_to_gene_symbols(
    tf_names: Iterable[str],
    hocomoco_tf_gene_symbols: Optional[pd.DataFrame] = None,
    dimer_separator: str = "::",
) -> pd.DataFrame:
    """Map motif TF names to gene symbols: HOCOMOCO table first, else pass the name through.

    Parameters
    ----------
    tf_names : motif TF names (HOCOMOCO mnemonics, JASPAR names such as ``FOS::JUN``, finemo labels).
    hocomoco_tf_gene_symbols : output of :func:`read_hocomoco_tf_gene_symbols`; default reads the
        bundled HOCOMOCO v11 table. Pass an empty DataFrame to disable the lookup (pure passthrough).
    dimer_separator : passthrough names are split on this into one row per partner gene.

    Returns
    -------
    DataFrame with columns tf, tf_gene_symbol, tf_gene_symbol_source (``hocomoco_v11`` | ``passthrough``);
    one row per (tf, gene symbol).
    """
    if hocomoco_tf_gene_symbols is None:
        hocomoco_tf_gene_symbols = read_hocomoco_tf_gene_symbols()
    lookup = (dict(zip(hocomoco_tf_gene_symbols["tf"], hocomoco_tf_gene_symbols["tf_gene_symbol"]))
              if len(hocomoco_tf_gene_symbols) else {})
    rows = []
    for tf in pd.unique(pd.Series(list(tf_names), dtype=str)):
        if tf in lookup:
            rows.append((tf, lookup[tf], "hocomoco_v11"))
            continue
        for symbol in tf.split(dimer_separator):
            if symbol.strip():
                rows.append((tf, symbol.strip(), "passthrough"))
    return pd.DataFrame(rows, columns=["tf", "tf_gene_symbol", "tf_gene_symbol_source"])


TFCLASS_CODE_RE = re.compile(r"\s*\{[\d.]+\}\s*$")


def strip_tfclass_code(name: str) -> str:
    """``bHLH-ZIP factors{1.2.6}; TBX6-related factors{6.5.5}`` -> ``bHLH-ZIP factors``."""
    return TFCLASS_CODE_RE.sub("", name.split(";")[0]).strip()


def read_hocomoco_tf_families(path: str = HOCOMOCO_V11_ANNOTATION_PATH) -> pd.DataFrame:
    """HOCOMOCO v11 TF name (``KLF4``, ``ZN148``) -> motif_family: the TFClass ``TF family`` without its
    code (``Three-zinc finger Krüppel-related factors{2.3.1}`` -> ``Three-zinc finger Krüppel-related
    factors``; the first of several ``;``-separated families); ``TF subfamily`` when the family is blank;
    the TF name when both are blank.
    One row per TF name (first model in file order)."""
    annotation = pd.read_csv(path, sep="\t", usecols=["Model", "TF family", "TF subfamily"], dtype=str)
    tf = annotation["Model"].str.split("_", n=1).str[0]
    family = annotation["TF family"].fillna("").map(strip_tfclass_code)
    subfamily = annotation["TF subfamily"].fillna("").map(strip_tfclass_code)
    family = family.where(family != "", subfamily)
    family = family.where(family != "", tf)
    return pd.DataFrame({"tf": tf, "motif_family": family}).drop_duplicates("tf").reset_index(drop=True)


def resolve_database_tf_to_gene(tf_name: str, gene_set: set, tf_name_to_symbol: dict,
                                gene_without_hyphen: dict) -> Optional[str]:
    """A TF name from a motif database -> expressed gene symbol, or None: the name itself; else its
    HOCOMOCO mnemonic's gene (``ANDR`` -> ``AR``, ``NF2L2`` -> ``NFE2L2``); else the gene with hyphens
    removed (``NKX61`` -> ``NKX6-1``)."""
    if tf_name in gene_set:
        return tf_name
    symbol = tf_name_to_symbol.get(tf_name)
    if symbol in gene_set:
        return symbol
    return gene_without_hyphen.get(tf_name)


def map_motif_families_to_gene_symbols(
    family_database_tfs: dict,
    genes: Iterable[str],
    tf_name_to_symbol: Optional[dict] = None,
) -> pd.DataFrame:
    """Map motif family names to the expressed genes on the motif database's TF list.

    Parameters
    ----------
    family_database_tfs : family name (the ``tf`` of the enrichment table, e.g. ``KLF-SP``) -> TF names
        listed by the database for the matched motif(s) (MotifCompendium ``TF`` column; union over the
        patterns named with this family). Names are gene symbols or HOCOMOCO mnemonics.
    genes : expressed-gene universe; database TFs not in it are dropped.
    tf_name_to_symbol : HOCOMOCO mnemonic -> gene symbol (default the bundled v11 table).

    Returns tf, tf_gene_symbol, tf_gene_symbol_source (``motifcompendium`` | ``passthrough``); families
    with no expressed TF (or no TF list, e.g. unnamed patterns) keep one passthrough row
    (tf_gene_symbol = family name).
    """
    if tf_name_to_symbol is None:
        hocomoco = read_hocomoco_tf_gene_symbols()
        tf_name_to_symbol = dict(zip(hocomoco["tf"], hocomoco["tf_gene_symbol"]))
    gene_set = set(map(str, genes))
    gene_without_hyphen = {gene.replace("-", ""): gene for gene in gene_set if "-" in gene}
    rows = []
    for family, database_tfs in family_database_tfs.items():
        members = [resolve_database_tf_to_gene(str(tf), gene_set, tf_name_to_symbol, gene_without_hyphen)
                   for tf in database_tfs]
        members = list(dict.fromkeys(member for member in members if member))
        if members:
            rows.extend((family, gene, "motifcompendium") for gene in members)
        else:
            rows.append((family, family, "passthrough"))
    return pd.DataFrame(rows, columns=["tf", "tf_gene_symbol", "tf_gene_symbol_source"])


# ---------------------------------------------------------------------------
# Evidence tables
# ---------------------------------------------------------------------------

def rank_genes_in_programs(gene_spectra_score: pd.DataFrame) -> pd.DataFrame:
    """Long table program, gene, loading_rank (1 = highest score; NaN scores rank last; ties by gene order).

    Uses the same ordering as ``motif_enrichment.select_top_program_genes``, so rank <= 300 means the
    gene is one of the 300 program genes tested for motif enrichment.
    """
    rows = []
    for program, scores in gene_spectra_score.iterrows():
        values = scores.to_numpy(dtype=float)
        order = np.argsort(np.where(np.isnan(values), np.inf, -values), kind="stable")
        rows.append(pd.DataFrame({
            "program": program, "gene": scores.index[order], "loading_rank": np.arange(1, len(order) + 1),
        }))
    return pd.concat(rows, ignore_index=True)


def nominate_candidate_tfs(
    motif_results: pd.DataFrame,
    gene_spectra_score: pd.DataFrame,
    perturbation_results: Optional[pd.DataFrame] = None,
    tf_gene_symbols: Optional[pd.DataFrame] = None,
    expressed_genes: Optional[Iterable[str]] = None,
    motif_fdr_threshold: float = 0.05,
    n_top_genes: int = 300,
    knockdown_fdr_threshold: float = 0.05,
    perturbation_columns: Optional[dict] = None,
) -> pd.DataFrame:
    """One row per (program, element_type, enriched TF, TF gene symbol) with expression + knockdown evidence.

    Parameters
    ----------
    motif_results : long table from ``motif_enrichment`` (program, element_type, tf, pvalue, fdr,
        enrichment[, significant][, motif_family][, motif_match_qvalue][, motif_source]). Enriched = the
        ``significant`` column when present (so the correlation method's r > 0 rule carries over), else
        fdr < ``motif_fdr_threshold`` and enrichment > 1 (paper definition). ``motif_family`` (placed after
        ``tf``), ``motif_match_qvalue`` and ``motif_source`` are carried into the output.
    gene_spectra_score : programs x genes (cNMF orientation); index must use the same program ids as
        ``motif_results`` (compared as strings). Its columns are the expressed-gene universe unless
        ``expressed_genes`` is given.
    perturbation_results : perturbation-program association table (PerturbNMF
        ``{K}_perturbation_association_results_{sample}.txt``: target_name, program_name, log2FC, adj_pval).
        None -> knockdown columns are NaN / False. The orchestrator concatenates all per-sample files;
        a (program, TF) pair that appears several times keeps the row with the smallest adj_pval
        (:func:`summarize_tf_knockdowns`), so ``motif+regulator`` means the knockdown is significant in
        ANY sample (no multiple-testing correction across samples).
    tf_gene_symbols : output of :func:`map_tf_names_to_gene_symbols`; default maps the enriched TFs with
        the bundled HOCOMOCO v11 table.
    expressed_genes : override for the expressed-gene universe.
    n_top_genes : program genes = top ``n_top_genes`` by gene_spectra_score.
    knockdown_fdr_threshold : knockdown regulates the program if adj_pval < this.
    perturbation_columns : rename map for ``perturbation_results`` onto
        {target_name, program_name, log2FC, adj_pval}, e.g. ``{"target": "target_name"}``.

    Returns
    -------
    DataFrame (``CANDIDATE_TF_COLUMNS`` + carried columns) sorted by program, element_type, evidence tier, fdr.
    """
    if "significant" in motif_results.columns:
        is_enriched = motif_results["significant"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        is_enriched = (motif_results["fdr"] < motif_fdr_threshold) & (motif_results["enrichment"] > 1)
    enriched = motif_results[is_enriched.to_numpy()].copy()
    source_columns = [c for c in ("motif_family", "motif_match_qvalue", "motif_source") if c in enriched.columns]
    enriched["program"] = enriched["program"].astype(str)
    if tf_gene_symbols is None:
        tf_gene_symbols = map_tf_names_to_gene_symbols(enriched["tf"])
    candidates = enriched[["program", "element_type", "tf", "pvalue", "fdr", "enrichment"] + source_columns].merge(
        tf_gene_symbols, on="tf", how="left")
    unmapped = candidates["tf_gene_symbol"].isna()
    candidates.loc[unmapped, "tf_gene_symbol"] = candidates.loc[unmapped, "tf"]
    candidates.loc[unmapped, "tf_gene_symbol_source"] = "passthrough"

    universe = set(gene_spectra_score.columns if expressed_genes is None else expressed_genes)
    candidates["tf_expressed"] = candidates["tf_gene_symbol"].isin(universe)

    scores = gene_spectra_score.copy()
    scores.index = scores.index.astype(str)
    ranks = rank_genes_in_programs(scores)
    ranks = ranks[ranks["gene"].isin(set(candidates["tf_gene_symbol"]))]
    candidates = candidates.merge(
        ranks.rename(columns={"gene": "tf_gene_symbol", "loading_rank": "tf_program_loading_rank"}),
        on=["program", "tf_gene_symbol"], how="left")
    candidates["tf_in_top_program_genes"] = candidates["tf_program_loading_rank"] <= n_top_genes

    candidates = candidates.merge(
        summarize_tf_knockdowns(perturbation_results, perturbation_columns),
        on=["program", "tf_gene_symbol"], how="left")
    candidates["tf_perturbed"] = candidates["tf_gene_symbol"].isin(
        perturbed_targets(perturbation_results, perturbation_columns))
    candidates["tf_knockdown_regulates_program"] = (candidates["knockdown_fdr"] < knockdown_fdr_threshold)

    candidates["evidence_tier"] = np.select(
        [candidates["tf_knockdown_regulates_program"], candidates["tf_in_top_program_genes"],
         candidates["tf_expressed"]],
        EVIDENCE_TIER_ORDER[:3], default=EVIDENCE_TIER_ORDER[3])
    tier_position = candidates["evidence_tier"].map({t: i for i, t in enumerate(EVIDENCE_TIER_ORDER)})
    candidates = (candidates.assign(tier_position=tier_position)
                  .sort_values(["program", "element_type", "tier_position", "fdr"], kind="stable")
                  .reset_index(drop=True))
    columns = list(CANDIDATE_TF_COLUMNS)
    if "motif_family" in source_columns:
        columns.insert(columns.index("tf") + 1, "motif_family")
    return candidates[columns + [c for c in source_columns if c != "motif_family"]]


def standardize_perturbation_results(perturbation_results: pd.DataFrame, perturbation_columns=None) -> pd.DataFrame:
    table = perturbation_results.rename(columns=perturbation_columns or {})
    missing = {"target_name", "program_name", "log2FC", "adj_pval"} - set(table.columns)
    if missing:
        raise ValueError(f"perturbation_results missing columns {sorted(missing)}; use perturbation_columns")
    return table.assign(program_name=table["program_name"].astype(str),
                        target_name=table["target_name"].astype(str))


def perturbed_targets(perturbation_results: Optional[pd.DataFrame], perturbation_columns=None) -> set:
    if perturbation_results is None:
        return set()
    return set(standardize_perturbation_results(perturbation_results, perturbation_columns)["target_name"])


def summarize_tf_knockdowns(perturbation_results: Optional[pd.DataFrame], perturbation_columns=None) -> pd.DataFrame:
    """Per (program, target gene): knockdown_log2fc, knockdown_fdr from the row with the smallest adj_pval.

    Rows repeat when per-sample association files are concatenated: taking the minimum adj_pval means a
    knockdown counts as regulating the program if it is significant in any one sample.
    """
    columns = ["program", "tf_gene_symbol", "knockdown_log2fc", "knockdown_fdr"]
    if perturbation_results is None:
        return pd.DataFrame(columns=columns).astype({"program": str, "tf_gene_symbol": str,
                                                     "knockdown_log2fc": float, "knockdown_fdr": float})
    table = standardize_perturbation_results(perturbation_results, perturbation_columns)
    table = (table.sort_values("adj_pval", kind="stable")
             .drop_duplicates(["program_name", "target_name"], keep="first"))
    return pd.DataFrame({
        "program": table["program_name"].to_numpy(),
        "tf_gene_symbol": table["target_name"].to_numpy(),
        "knockdown_log2fc": table["log2FC"].to_numpy(dtype=float),
        "knockdown_fdr": table["adj_pval"].to_numpy(dtype=float),
    })
