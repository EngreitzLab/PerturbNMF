"""Which TF motif results to show per program, from the Stage 2 motif enrichment tables.

Reads '{K}_motif_enrichment.txt' (run_motif_enrichment.py), optionally '{K}_candidate_tfs.txt'
(nominate_candidate_tfs.py), '{K}_motif_logos.json' and the run's '{K}_motif_enrichment_config.yml'. Per
program and element type (promoter / enhancer; one block per motif source when the table has several,
FIMO first, never pooled): motifs tested and significant, and the top significant motif families, each
with its best motifs and the candidate TFs (expressed TFs on the motifs' TF lists). The selection rules
follow the program annotator's motif evidence section, so the viewer shows what the annotator saw.
"""

import json
import re
from pathlib import Path

import pandas as pd

MOTIF_FAMILIES_PER_ELEMENT_TYPE = 5
MOTIFS_PER_FAMILY = 3
CANDIDATES_PER_FAMILY = 6
MOTIF_ELEMENT_TYPES = ("promoter", "enhancer")
MOTIF_SOURCE_ORDER = (
    "fimo",
    "finemo",
)  # Stage 2 motif_source values, shown in this order
FIMO_DATABASE_LABELS = {
    "motifcompendium": "FIMO/MotifCompendium",
    "hocomoco": "FIMO/HOCOMOCO",
}
MOTIF_SOURCE_LABELS = {"fimo": FIMO_DATABASE_LABELS["hocomoco"], "finemo": "Fi-NeMo"}
MOTIFCOMPENDIUM_CLUSTER_RE = r"_\d+$"  # KLF-SP_0: a MotifCompendium cluster name
# candidate TF tiers shown (nominate_candidate_tfs.py), strongest first; motif_only (not expressed) is hidden
CANDIDATE_TIERS_SHOWN = (
    "motif+regulator",
    "motif+expressed_in_program",
    "motif+expressed",
)
DEFAULT_MOTIF_TEST = {
    "method": "ttest",
    "n_top": 300,
}  # run_motif_enrichment.py defaults


def program_number(value):
    """Program id as an int: 12, "12" and "K60_12" all give 12; None if there is no number."""
    match = re.search(r"(\d+)$", str(value).strip())
    return int(match.group(1)) if match else None


def as_bool(values):
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_motif_test(motif_table_path):
    """How Stage 2 tested the motifs, {"method": "ttest" | "correlation", "n_top": int}: from the run's
    '<table stem>_config.yml' next to the table (written as JSON), else the Stage 2 defaults."""
    motif_test = dict(DEFAULT_MOTIF_TEST)
    config = Path(motif_table_path).with_name(
        Path(motif_table_path).stem + "_config.yml"
    )
    if config.is_file():
        try:
            arguments = json.loads(config.read_text()).get("arguments", {})
        except ValueError:
            arguments = {}
        motif_test.update(
            {
                key: arguments[name]
                for key, name in (("method", "motif_method"), ("n_top", "n_top"))
                if arguments.get(name) is not None
            }
        )
    motif_test["n_top"] = int(motif_test["n_top"])
    return motif_test


def read_motif_tables(motif_enrichment_path, candidate_tfs_path=None):
    """(motif table, candidate table or None), both with an int `program_id` column."""
    motifs = pd.read_csv(motif_enrichment_path, sep="\t")
    motifs["program_id"] = motifs["program"].map(program_number)
    motifs["element_type"] = motifs["element_type"].astype(str).str.lower()
    if "significant" in motifs.columns:
        motifs["significant"] = as_bool(motifs["significant"])
    else:
        motifs["significant"] = (motifs["fdr"] < 0.05) & (motifs["enrichment"] > 1)
    candidates = None
    if candidate_tfs_path:
        candidates = pd.read_csv(candidate_tfs_path, sep="\t")
        candidates["program_id"] = candidates["program"].map(program_number)
    return motifs, candidates


def fimo_source_label(motifs):
    """ "FIMO/MotifCompendium" when most FIMO motif names are MotifCompendium clusters (KLF-SP_0), else
    "FIMO/HOCOMOCO" (TF names such as KLF4)."""
    fimo = (
        motifs[motifs["motif_source"].astype(str) == "fimo"]
        if "motif_source" in motifs.columns
        else motifs
    )
    names = fimo["tf"].astype(str).drop_duplicates()
    is_cluster = names.str.contains(MOTIFCOMPENDIUM_CLUSTER_RE, regex=True)
    return FIMO_DATABASE_LABELS[
        "motifcompendium" if len(names) and is_cluster.mean() > 0.5 else "hocomoco"
    ]


def list_motif_sections(motifs):
    """(key, element_type, source) per motif block: ('promoter', 'promoter', None), ('enhancer', ...) for one
    motif source; with several (`motif_source` column), one per source x element type, FIMO first, keyed
    '<element_type>_<source>'."""
    sources = [None]
    if "motif_source" in motifs.columns and motifs["motif_source"].nunique() > 1:
        present = list(motifs["motif_source"].astype(str).unique())
        sources = sorted(
            present,
            key=lambda s: (
                MOTIF_SOURCE_ORDER.index(s)
                if s in MOTIF_SOURCE_ORDER
                else len(MOTIF_SOURCE_ORDER),
                s,
            ),
        )
    return [
        (
            element_type if source is None else f"{element_type}_{source}",
            element_type,
            source,
        )
        for source in sources
        for element_type in MOTIF_ELEMENT_TYPES
    ]


def candidate_entry(row):
    """One candidate TF: symbol, tier, the motif it rests on, knockdown or loading-rank support."""
    symbol = row["tf_gene_symbol"] if pd.notna(row.get("tf_gene_symbol")) else row["tf"]
    entry = {"tf": str(symbol), "tier": row["evidence_tier"], "motif": str(row["tf"])}
    if row["evidence_tier"] == "motif+regulator":
        entry.update(
            log2fc=round(float(row["knockdown_log2fc"]), 3),
            adj_p=float(f"{row['knockdown_fdr']:.2g}"),
        )
    elif pd.notna(row.get("tf_program_loading_rank")):
        entry["loading_rank"] = int(row["tf_program_loading_rank"])
    return entry


def select_program_motifs(program_id, motifs, candidates):
    """Per block (list_motif_sections): motifs tested and significant, and the top
    MOTIF_FAMILIES_PER_ELEMENT_TYPE significant families (`motif_family`, else the motif name), ordered by
    their best motif (FDR, ties higher enrichment). Each family: `family`, `n_significant`, its best
    MOTIFS_PER_FAMILY motifs [motif, enrichment, FDR] and up to CANDIDATES_PER_FAMILY candidate TFs of those
    significant motifs (strongest tier, then FDR, then loading rank; one per TF gene).
    `section_sources` maps each block to its Stage 2 motif_source; with several sources, `sections` lists
    [key, element_type, source label]."""
    rows = motifs[motifs["program_id"] == program_id]
    sections = list_motif_sections(motifs)
    multi_source = sections[0][2] is not None
    labels = {**MOTIF_SOURCE_LABELS, "fimo": fimo_source_label(motifs)}
    program_candidates = None
    if candidates is not None:
        program_candidates = candidates[
            (candidates["program_id"] == program_id)
            & candidates["evidence_tier"].isin(CANDIDATE_TIERS_SHOWN)
        ].copy()
        program_candidates["tier_rank"] = program_candidates["evidence_tier"].map(
            {t: i for i, t in enumerate(CANDIDATE_TIERS_SHOWN)}
        )
        program_candidates["symbol"] = (
            program_candidates["tf_gene_symbol"].fillna(program_candidates["tf"])
            if "tf_gene_symbol" in program_candidates
            else program_candidates["tf"]
        )
        tie_break = (
            ["tf_program_loading_rank"]
            if "tf_program_loading_rank" in program_candidates
            else []
        )
        program_candidates = program_candidates.sort_values(
            ["tier_rank", "fdr"] + tie_break, kind="stable"
        )
    single_source = (
        str(motifs["motif_source"].iloc[0])
        if "motif_source" in motifs.columns and len(motifs)
        else None
    )
    selection = {
        "n_tested": {},
        "n_significant": {},
        "n_families": {},
        "families": {},
        "section_sources": {},
    }
    for key, element_type, source in sections:
        typed = rows[rows["element_type"] == element_type]
        if source is not None:
            typed = typed[typed["motif_source"].astype(str) == source]
        hits = typed[typed["significant"]].sort_values(
            ["fdr", "enrichment"], ascending=[True, False], kind="stable"
        )
        hits = hits.assign(
            family=hits["motif_family"].fillna(hits["tf"])
            if "motif_family" in hits
            else hits["tf"]
        )
        selection["n_tested"][key] = int(typed["tf"].nunique())
        selection["n_significant"][key] = int(len(hits))
        selection["n_families"][key] = int(hits["family"].nunique())
        selection["section_sources"][key] = (
            source if source is not None else single_source
        )
        families = []
        for family, members in list(hits.groupby("family", sort=False))[
            :MOTIF_FAMILIES_PER_ELEMENT_TYPE
        ]:
            entry = {
                "family": str(family),
                "n_significant": int(len(members)),
                "motifs": [
                    [str(tf), round(float(enrichment), 2), float(f"{fdr:.2g}")]
                    for tf, enrichment, fdr in zip(
                        members["tf"], members["enrichment"], members["fdr"]
                    )
                ][:MOTIFS_PER_FAMILY],
                "candidates": [],
            }
            if program_candidates is not None:
                chosen = program_candidates[
                    (program_candidates["element_type"] == element_type)
                    & program_candidates["tf"].isin(set(members["tf"]))
                ]
                if source is not None and "motif_source" in chosen:
                    chosen = chosen[chosen["motif_source"].astype(str) == source]
                chosen = chosen.drop_duplicates("symbol").head(CANDIDATES_PER_FAMILY)
                entry["candidates"] = [
                    candidate_entry(row) for _, row in chosen.iterrows()
                ]
            families.append(entry)
        selection["families"][key] = families
    if multi_source:
        selection["sections"] = [
            [key, element_type, labels.get(source, source)]
            for key, element_type, source in sections
        ]
    return selection


def read_motif_logos(path):
    """Stage 2 logo matrices {source: {motif: entry}} from '{K}_motif_logos.json'."""
    return json.loads(Path(path).read_text()).get("logos", {})


def shown_logos(programs, logos):
    """Only the logos of the motifs shown, keyed '<source>|<motif>' (keeps the page small). A block without
    a motif source uses the logo file's only source."""
    default_source = next(iter(logos)) if len(logos) == 1 else None
    out = {}
    for program in programs:
        motifs = program.get("motifs") or {}
        for key, families in (motifs.get("families") or {}).items():
            source = motifs.get("section_sources", {}).get(key) or default_source
            for family in families:
                for tf, _, _ in family["motifs"]:
                    entry = (logos.get(source) or {}).get(tf)
                    if entry:
                        out[f"{source}|{tf}"] = {
                            "kind": entry.get("kind", "information_content"),
                            "matrix": entry["matrix"],
                        }
    return out
