"""Motif logo matrices for Stage 3 (``{K}_motif_logos.json``): one small matrix per motif that is
significant in any program, so the annotation viewer can draw sequence logos without the motif files.

Matrices (positions x A, C, G, T, rounded to :data:`LOGO_DECIMALS`)
    ``information_content``: letter heights = probability x (2 - entropy) bits, from a MEME PFM
        (FIMO databases: MotifCompendium clusters by name; HOCOMOCO v11 TFs by their best model,
        quality A > B > C > D; Fi-NeMo patterns without a CWM: the matched MotifCompendium PFM), flanks
        trimmed where the column information is below :data:`IC_TRIM_FRACTION` of the maximum.
    ``cwm``: the TF-MoDISco contribution weight matrix of the Fi-NeMo pattern (from the TF-MoDISco h5 in
        ENCODE's "sequence motifs" tar), flanks trimmed where the column |contribution| is below
        :data:`CWM_TRIM_FRACTION` of the maximum. Values can be negative.

JSON layout::

    {"version": 1, "logos": {"fimo": {"KLF-SP_0": {"kind": "information_content", "motif_id": "KLF-SP_0",
                                                  "matrix": [[0.1, 0.0, 1.2, 0.0], ...]}, ...},
                             "finemo": {"GATA_0": {"kind": "cwm", "pattern_id": "pos_patterns...", ...}}}}

Keys under each source are the ``tf`` values of ``{K}_motif_enrichment.txt``.
"""

import glob
import json
import logging
import os
import re
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

import motif_hit_calling

logger = logging.getLogger(__name__)

LOGO_DECIMALS = 2
CWM_TRIM_FRACTION = 0.3
IC_TRIM_FRACTION = 0.35     # MotifCompendium PFMs carry long low-information flanks
LOGO_JSON_VERSION = 1
HOCOMOCO_QUALITY_ORDER = "ABCDS"


def read_meme_motifs(path: str) -> Dict[str, dict]:
    """MEME motif file -> {motif id: {"alt_id": str, "matrix": positions x 4 probabilities}}."""
    motifs, current, rows_left = {}, None, 0
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "MOTIF":
                current = {"alt_id": fields[2] if len(fields) > 2 else "", "rows": []}
                motifs[fields[1]] = current
                rows_left = 0
            elif current is not None and line.startswith("letter-probability matrix"):
                width = re.search(r"w=\s*(\d+)", line)
                rows_left = int(width.group(1)) if width else 10 ** 9
            elif current is not None and rows_left > 0:
                try:
                    values = [float(value) for value in fields[:4]]
                except ValueError:
                    rows_left = 0
                    continue
                current["rows"].append(values)
                rows_left -= 1
    return {motif_id: {"alt_id": motif["alt_id"], "matrix": np.array(motif["rows"], dtype=float)}
            for motif_id, motif in motifs.items() if motif["rows"]}


def information_content_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Letter heights in bits: p x (2 - H), H = -sum p log2 p per position (rows renormalized)."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0, None)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0.0), axis=1)
    return probabilities * (2.0 - entropy)[:, None]


def trim_matrix(matrix: np.ndarray, fraction: float = CWM_TRIM_FRACTION) -> np.ndarray:
    """Drop flanking positions whose total |value| is below ``fraction`` of the maximum position total."""
    totals = np.abs(matrix).sum(axis=1)
    if not len(totals) or totals.max() <= 0:
        return matrix
    keep = np.flatnonzero(totals >= fraction * totals.max())
    return matrix[keep[0]:keep[-1] + 1]


def logo_entry(matrix: np.ndarray, kind: str, **identifiers) -> dict:
    return {"kind": kind, **identifiers, "matrix": np.round(matrix, LOGO_DECIMALS).tolist()}


def information_content_logo(probabilities: np.ndarray, **identifiers) -> dict:
    """Trimmed information-content logo entry of a PFM."""
    return logo_entry(trim_matrix(information_content_matrix(probabilities), IC_TRIM_FRACTION), "information_content",
                      **identifiers)


def select_hocomoco_model(tf: str, motifs: Dict[str, dict]) -> Optional[str]:
    """Best HOCOMOCO model of a TF name (``KLF4`` -> ``KLF4_HUMAN.H11MO.0.A``): quality letter A first, then id;
    a motif whose id or alt id equals the name wins (JASPAR / user files)."""
    if tf in motifs:
        return tf
    exact_alt = sorted(motif_id for motif_id, motif in motifs.items() if motif["alt_id"] == tf)
    if exact_alt:
        return exact_alt[0]
    models = [motif_id for motif_id in motifs if motif_id.split("_", 1)[0] == tf]
    if not models:
        return None
    def quality_rank(motif_id: str) -> int:
        quality = motif_id[-1]
        return HOCOMOCO_QUALITY_ORDER.index(quality) if quality in HOCOMOCO_QUALITY_ORDER else len(HOCOMOCO_QUALITY_ORDER)

    return sorted(models, key=lambda motif_id: (quality_rank(motif_id), motif_id))[0]


def fimo_logos(tfs: Iterable[str], motif_file: str, collapse_motif_ids: bool) -> Dict[str, dict]:
    """Information-content logos of FIMO motifs (cluster ids as is, or TF names -> best model)."""
    motifs = read_meme_motifs(motif_file)
    logos = {}
    for tf in tfs:
        motif_id = select_hocomoco_model(tf, motifs) if collapse_motif_ids else (tf if tf in motifs else None)
        if motif_id is None:
            continue
        logos[tf] = information_content_logo(motifs[motif_id]["matrix"], motif_id=motif_id)
    return logos


def find_modisco_h5_files(root: str, head: str = "counts") -> list:
    """TF-MoDISco h5 files for ``head`` inside an extracted ENCODE "sequence motifs" tar (or the file itself)."""
    if os.path.isfile(root):
        return [root]
    files = sorted(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))
    preferred = [path for path in files if head in os.path.basename(path)]
    return preferred or files


PATTERN_NUMBER_RE = re.compile(r"(pos|neg)_patterns.*?pattern_(\d+)$")


def read_modisco_cwms(h5_path: str) -> Dict[str, np.ndarray]:
    """{``pos_patterns.pattern_3``: CWM (positions x 4)} from a tfmodisco(-lite) h5 (``contrib_scores``)."""
    import h5py
    cwms = {}
    with h5py.File(h5_path, "r") as handle:
        for polarity in ("pos_patterns", "neg_patterns"):
            if polarity not in handle:
                continue
            for pattern in handle[polarity]:
                group = handle[polarity][pattern]
                if "contrib_scores" in group:
                    cwms[f"{polarity}.{pattern}"] = np.array(group["contrib_scores"], dtype=float)
    return cwms


COMPENDIUM_CLUSTER_RE = re.compile(r"^cluster_(\d+)$")


def modisco_keys(pattern_id: str) -> list:
    """h5 keys that can hold a pattern's CWM. ENCODE pattern id (``pos_patterns.<...>_counts_pattern_3``) ->
    ``pos_patterns.pattern_3``; local compendium motif id (``cluster_12``, export_motif_hits_for_perturbnmf.py)
    -> ``pos_patterns.12`` / ``neg_patterns.12`` (the compiled compendium h5 is keyed by cluster id, one id
    space shared by pos and neg)."""
    match = PATTERN_NUMBER_RE.search(str(pattern_id))
    if match:
        return [f"{match.group(1)}_patterns.pattern_{match.group(2)}"]
    match = COMPENDIUM_CLUSTER_RE.match(str(pattern_id))
    return [f"{polarity}_patterns.{match.group(1)}" for polarity in ("pos", "neg")] if match else []


def finemo_logos(pattern_names: pd.DataFrame, tfs: Iterable[str], finemo_motifs: Optional[str] = None,
                 pfm_file: Optional[str] = None, head: str = "counts") -> Dict[str, dict]:
    """Fi-NeMo logos per tested name (cluster): the CWM of its best-matching pattern (lowest TOMTOM q) when
    TF-MoDISco h5 files are given, else the information-content logo of the matched database PFM."""
    wanted = set(map(str, tfs))
    names = pattern_names[pattern_names["tf"].astype(str).isin(wanted)].copy()
    names["q_sort"] = pd.to_numeric(names.get("top_match_qvalue"), errors="coerce").fillna(np.inf)
    # local annotation tables have a match score instead of a TOMTOM q (higher = better)
    names["score_sort"] = (-pd.to_numeric(names["top_match_score"], errors="coerce").fillna(-np.inf)
                           if "top_match_score" in names.columns else 0.0)
    representative = names.sort_values(["tf", "q_sort", "score_sort"], kind="stable").drop_duplicates("tf")
    cwms = {}
    for h5_path in (find_modisco_h5_files(finemo_motifs, head) if finemo_motifs else []):
        try:
            cwms.update(read_modisco_cwms(h5_path))
        except (OSError, ImportError) as error:
            logger.warning("cannot read TF-MoDISco h5 %s: %s", h5_path, error)
    pfms = read_meme_motifs(pfm_file) if pfm_file and os.path.exists(pfm_file) else {}
    logos = {}
    for row in representative.itertuples(index=False):
        cwm = next((cwms[key] for key in modisco_keys(row.pattern_id) if key in cwms), None)
        if cwm is not None:
            logos[row.tf] = logo_entry(trim_matrix(cwm), "cwm", pattern_id=row.pattern_id)
        elif isinstance(row.top_match, str) and row.top_match in pfms and row.is_named:
            logos[row.tf] = information_content_logo(pfms[row.top_match]["matrix"], motif_id=row.top_match,
                                                     pattern_id=row.pattern_id)
    return logos


def build_motif_logos(results: pd.DataFrame, fimo_motif_file: Optional[str] = None,
                      fimo_collapse_motif_ids: bool = True, pattern_names: Optional[pd.DataFrame] = None,
                      finemo_motifs: Optional[str] = None, finemo_pfm_file: Optional[str] = None,
                      finemo_head: str = "counts") -> dict:
    """Logos of the motifs significant in any program, per motif source (``fimo`` when the table has no
    ``motif_source`` column and no pattern names, else as in ``motif_source``)."""
    significant = results[results["significant"].astype(bool)]
    if "motif_source" in significant.columns:
        by_source = {source: rows["tf"].unique() for source, rows in significant.groupby("motif_source")}
    else:
        by_source = {"finemo" if pattern_names is not None and fimo_motif_file is None else "fimo":
                     significant["tf"].unique()}
    logos = {}
    if len(by_source.get("fimo", [])) and fimo_motif_file and os.path.exists(fimo_motif_file):
        logos["fimo"] = fimo_logos(by_source["fimo"], fimo_motif_file, fimo_collapse_motif_ids)
    if len(by_source.get("finemo", [])) and pattern_names is not None:
        logos["finemo"] = finemo_logos(pattern_names, by_source["finemo"], finemo_motifs, finemo_pfm_file, finemo_head)
    for source, tfs in by_source.items():
        n_missing = len(set(tfs) - set(logos.get(source, {})))
        if n_missing:
            logger.info("%s: no logo matrix for %d of %d significant motifs", source, n_missing, len(tfs))
    return {"version": LOGO_JSON_VERSION, "logos": logos}


def build_motif_logos_for_run(results: pd.DataFrame, args, resources: dict, motif_database_kind: str,
                              pattern_names: Optional[pd.DataFrame]) -> dict:
    """:func:`build_motif_logos` with the orchestrator's arguments. Fi-NeMo PFM logos use ``--finemo_pfm_file``,
    else the FIMO MotifCompendium file -- but only if the patterns were named with the same database
    release (motif indices differ between releases), which is checked on the motif names."""
    sources = set(results["motif_source"]) if "motif_source" in results.columns else {args.motif_source}
    pfm_file = getattr(args, "finemo_pfm_file", None)
    if pattern_names is not None and not pfm_file and motif_database_kind == "motifcompendium" and args.motif_file \
            and os.path.exists(args.motif_file):
        release_of_patterns = str(pattern_names["motifcompendium_metadata"].iloc[0]) if len(pattern_names) else ""
        release_of_file = os.path.basename(motif_hit_calling.select_motifcompendium_metadata(
            read_meme_motifs(args.motif_file).keys()))
        if release_of_patterns in ("", release_of_file):
            pfm_file = args.motif_file
        else:
            logger.info("Fi-NeMo patterns were named with %s, %s is %s: no PFM logos for Fi-NeMo",
                        release_of_patterns, args.motif_file, release_of_file)
    return build_motif_logos(
        results, fimo_motif_file=args.motif_file if "fimo" in sources else None,
        fimo_collapse_motif_ids=motif_database_kind != "motifcompendium",
        pattern_names=pattern_names, finemo_motifs=resources.get("finemo_motifs"), finemo_pfm_file=pfm_file,
        finemo_head=getattr(args, "finemo_head", "counts"))


def write_motif_logos(logos: dict, path: str) -> None:
    with open(path, "w") as handle:
        json.dump(logos, handle, separators=(",", ":"))
