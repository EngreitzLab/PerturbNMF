#!/usr/bin/env python
"""Program TF-motif enrichment for a PerturbNMF run: regions -> motif hits -> enrichment -> candidate TFs.

For each K / density threshold, reads the programs from
``{out_dir}/{run_name}/Inference/Inference.gene_spectra_score.k_{K}.dt_{thresh}.txt`` and the knockdown
effects from ``{out_dir}/{run_name}/Evaluation/{K}_{thresh}/{K}_perturbation_association_results_*.txt``
and writes to ``Evaluation/{K}_{thresh}/``:

  {K}_motif_enrichment.txt           long table: program, element_type, tf, motif_family, pvalue, fdr,
                                     enrichment, n_program_genes_tested, n_background_genes,
                                     mean_count_program, mean_count_background, significant
                                     (+ motif_match_qvalue for Fi-NeMo, + motif_source with --motif_source both)
  {K}_candidate_tfs.txt              enriched motifs x TF genes x expression x knockdown evidence
                                     (nominate_candidate_tfs.py; + motif_family)
  {K}_motif_logos.json               logo matrices of the motifs significant in any program (motif_logos.py)
  {K}_motif_enrichment_config.yml    arguments, resolved input files, hit tables used
  {K}_finemo_pattern_names.tsv       (finemo only) TF-MoDISco pattern -> TOMTOM match -> cluster, family, TF list

Motif database (``--motif_db``, FIMO): ``motifcompendium`` (default) = MotifCompendium-Database-Human, one
test per database motif cluster (``KLF-SP_0``, ``GATA_1``); motif_family = cluster name without ``_<n>``;
candidate TF genes = the cluster's TF list (bundled database metadata) that are expressed. Fi-NeMo patterns
are named by the same kind of cluster, so the two sources share one vocabulary (the ENCODE reports were
matched against the 2025-09 release, the default PFM file is the 2026-05 release; family names mostly agree).
``hocomoco_v11`` = the Schnitzler 2024 setup (one test per TF, motif ids collapsed at ``_``; TFClass family).
A MEME file path = any other database (ids collapsed like HOCOMOCO; family from HOCOMOCO v11 when known).

Motif hit tables do not depend on the programs, so they are built once and cached in
``--motif_hit_cache_dir`` (default ``{out_dir}/{run_name}/Evaluation/motif_hits/``), one sub-directory
per (element type, motif source, parameters) with the parameters in ``params.json``; a rerun or another
K reuses them. Precomputed FIMO tables can be passed with ``--promoter_hits`` / ``--enhancer_hits``.

Method (defaults = the t-test method of Schnitzler et al. Nature 2024; see ../src/motif_enrichment.py):
  * promoters: strand-aware TSS-250..TSS+50 from --gene_annotation (``schnitzler2024`` = the published window)
  * enhancers: element-gene links (--enhancer_links, or rank-1 ``e2g_links`` of
    --regulatory_resources_manifest from find_regulatory_resources.py); promoter-class elements dropped
  * FIMO (MEME) with HOCOMOCO v11 full; hits kept at p < 1e-4 (promoter) / 1e-6 (enhancer)
    or Fi-NeMo hit calls (ENCODE ChromBPNet "sequence motifs instances" + "report" tars), no p filter
  * test: top --n_top genes per program vs background, Welch t-test (``ttest``) or correlation of
    motif count with the full loading vector (``correlation``); BH per element type (and source)

Example (hg38 run, element-gene links + FIMO + ENCODE ChromBPNet Fi-NeMo calls):
  python run_motif_enrichment.py --out_dir <path/to/Results> --run_name <run_name> \\
      --K <K> --sel_threshs 0.2 --enhancer_links <path/to/links.bedpe.gz> \\
      --genome_fasta <path/to/hg38.fa> --gene_annotation <path/to/genes.gtf.gz> \\
      --motif_file <path/to/MotifCompendium-Database-Human.meme.txt> \\
      --motif_source both --finemo_instances ENCFFxxx.tar.gz --finemo_report ENCFFyyy.tar.gz \\
      --fimo_binary "$(which fimo)" --n_jobs 16
  Local Fi-NeMo tables instead of ENCODE tars (export_motif_hits_for_perturbnmf.py):
      --finemo_instances motif_hits_<dataset>.tsv.gz --finemo_annotation motif_annotation.tsv

Resource defaults (used when the flag is not given): --genome_fasta $PERTURBNMF_GENOME_FASTA,
--gene_annotation $PERTURBNMF_GENE_ANNOTATION, --motif_file $PERTURBNMF_MOTIFCOMPENDIUM_MEME
(--motif_db motifcompendium) or $PERTURBNMF_HOCOMOCO_V11_MEME (--motif_db hocomoco_v11). Unset and
not given -> error when the input is needed. Download URLs: ../README.md (motif enrichment resources).
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import tarfile
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import motif_enrichment  # noqa: E402
import motif_hit_calling  # noqa: E402
import motif_logos  # noqa: E402
import nominate_candidate_tfs  # noqa: E402

logger = logging.getLogger("run_motif_enrichment")

# Site-specific resource paths come from environment variables (see the module docstring)
MOTIF_DATABASE_ENV_VARS = {
    "motifcompendium": "PERTURBNMF_MOTIFCOMPENDIUM_MEME",
    "hocomoco_v11": "PERTURBNMF_HOCOMOCO_V11_MEME",
}
MOTIF_DATABASE_FILES = {kind: os.environ.get(variable) for kind, variable in MOTIF_DATABASE_ENV_VARS.items()}
DEFAULT_MOTIF_DB = "motifcompendium"
DEFAULT_GENOME_FASTA = os.environ.get("PERTURBNMF_GENOME_FASTA")
DEFAULT_GENE_ANNOTATION = os.environ.get("PERTURBNMF_GENE_ANNOTATION")

ELEMENT_TYPES = ("promoter", "enhancer")
MOTIF_SOURCES = ("fimo", "finemo", "both")
HIT_TABLE_VERSION = 3     # bump when hit-table building changes, to invalidate caches (3: Fi-NeMo motif_id = cluster)


# ---------------------------------------------------------------------------
# Arguments (shared with cNMF_evaluation_pipeline.py --Perform_motif)
# ---------------------------------------------------------------------------

def add_motif_arguments(parser: argparse.ArgumentParser) -> None:
    """Motif-enrichment options. cNMF_evaluation_pipeline.py adds the same group, so both CLIs agree."""
    group = parser.add_argument_group("motif enrichment")
    group.add_argument("--motif_method", default="ttest", choices=list(motif_enrichment.ENRICHMENT_METHODS),
                       help="ttest: top-n program genes vs background, Welch t-test (Schnitzler 2024, default); "
                            "correlation: motif count vs full loading vector (loading-correlation variant)")
    group.add_argument("--motif_correlation", default="pearson", choices=["pearson", "spearman"],
                       help="correlation type for --motif_method correlation")
    group.add_argument("--motif_element_types", nargs="+", default=list(ELEMENT_TYPES), choices=list(ELEMENT_TYPES))
    group.add_argument("--motif_source", default="fimo", choices=list(MOTIF_SOURCES),
                       help="fimo: scan regions with FIMO; finemo: ENCODE/ChromBPNet Fi-NeMo hit calls; both")
    group.add_argument("--motif_fdr_threshold", type=float, default=0.05)
    group.add_argument("--motif_hit_cache_dir", default=None,
                       help="hit-table cache (default {out_dir}/{run_name}/Evaluation/motif_hits)")
    # regions
    group.add_argument("--gene_annotation", default=DEFAULT_GENE_ANNOTATION,
                       help="GTF(.gz) (canonical TSS) or BED6 gene bounds for promoters (default $PERTURBNMF_GENE_ANNOTATION)")
    group.add_argument("--promoter_window_mode", default="strand_aware",
                       choices=list(motif_hit_calling.PROMOTER_WINDOW_MODES))
    group.add_argument("--promoter_upstream", type=int, default=250)
    group.add_argument("--promoter_downstream", type=int, default=50)
    group.add_argument("--enhancer_links", default=None, help="ABC / ENCODE-rE2G / scE2G tsv or IGVF bedpe")
    group.add_argument("--regulatory_resources_manifest", default=None,
                       help="regulatory_resources_manifest.tsv (find_regulatory_resources.py): rank-1 rows supply "
                            "--enhancer_links / --finemo_instances / --finemo_report when those are not given")
    group.add_argument("--link_format", default="auto", choices=list(motif_hit_calling.LINK_FORMATS))
    group.add_argument("--link_score_threshold", type=float, default=None)
    group.add_argument("--link_score_column", default=None)
    group.add_argument("--merge_overlapping_links", action="store_true",
                       help="merge overlapping enhancer elements of the same target gene into one region (for links "
                            "pooled over several conditions / samples, so a hit is counted once per gene)")
    # FIMO
    group.add_argument("--genome_fasta", default=DEFAULT_GENOME_FASTA,
                       help="genome FASTA for FIMO (default $PERTURBNMF_GENOME_FASTA)")
    group.add_argument("--genome_build", default="hg38",
                       help="build of --genome_fasta and of all region / hit coordinates (hg38 / GRCh38, hg19 / "
                            "GRCh37). Checked against the FASTA chr1 length, manifest assemblies and build names in "
                            "input file names; a mismatch is an error. Stored in the hit-table cache key")
    group.add_argument("--motif_db", default=DEFAULT_MOTIF_DB,
                       help="FIMO motif database: motifcompendium (default; MotifCompendium-Database-Human, one test "
                            "per database motif cluster, TF lists from the database), hocomoco_v11 (Schnitzler 2024: "
                            "one test per TF), or the path of another MEME file (ids collapsed like HOCOMOCO)")
    group.add_argument("--motif_file", default=None,
                       help="MEME file to scan (default: $PERTURBNMF_MOTIFCOMPENDIUM_MEME or $PERTURBNMF_HOCOMOCO_V11_MEME for --motif_db); names must follow --motif_db")
    group.add_argument("--motifcompendium_metadata", default=None,
                       help="MotifCompendium metadata tsv (TF lists); default the bundled release that contains the "
                            "most of the motif names (motif_databases/)")
    group.add_argument("--promoter_pvalue_threshold", type=float, default=1e-4)
    group.add_argument("--enhancer_pvalue_threshold", type=float, default=1e-6)
    group.add_argument("--fimo_backend", default="auto", choices=["auto", "meme", "memelite"])
    group.add_argument("--fimo_binary", default="fimo")
    group.add_argument("--meme_default_mode", action="store_true",
                       help="MEME fimo default mode (q-values, --max-stored-scores cap; as in Schnitzler et al. 2024) "
                            "instead of --text. The cap applies per chunk: use --n_chunks 1 to mimic one FASTA")
    group.add_argument("--n_chunks", type=int, default=64)
    group.add_argument("--n_jobs", type=int, default=1)
    group.add_argument("--promoter_hits", default=None, help="precomputed FIMO tsv for promoters (sequence_name = gene)")
    group.add_argument("--enhancer_hits", default=None,
                       help="precomputed FIMO tsv for enhancers (sequence_name = region|class|element|gene)")
    # Fi-NeMo
    group.add_argument("--finemo_instances", default=None,
                       help="Fi-NeMo hits tsv, or ENCODE 'sequence motifs instances' tar / extracted directory")
    group.add_argument("--finemo_report", default=None,
                       help="TF-MoDISco report html, or ENCODE 'sequence motifs report' tar / extracted directory")
    group.add_argument("--finemo_annotation", default=None,
                       help="local motif annotation tsv (motif_id, database_motif, candidate_tfs[, posneg]; "
                            "export_motif_hits_for_perturbnmf.py) instead of --finemo_report")
    group.add_argument("--finemo_motifs", default=None,
                       help="for logos: ENCODE 'sequence motifs' tar / directory (TF-MoDISco h5 with pattern CWMs); "
                            "default: logos from the matched MotifCompendium PFM (--finemo_pfm_file)")
    group.add_argument("--finemo_pfm_file", default=None,
                       help="MEME file of the MotifCompendium release the Fi-NeMo patterns were matched against "
                            "(logos when --finemo_motifs is not given; default the --motif_db motifcompendium file)")
    group.add_argument("--finemo_head", default="counts", choices=["counts", "profile"])
    group.add_argument("--finemo_lambda", type=float, default=0.7, help="Fi-NeMo lambda of the instance set")
    group.add_argument("--finemo_pattern_prefixes", nargs="+", default=["pos_patterns"])
    group.add_argument("--finemo_qvalue_threshold", type=float, default=None,
                       help="name a pattern by its top TOMTOM match only if q < this (default: off, always name "
                            "by the top match; its q-value is kept as motif_match_qvalue)")
    # candidate TFs
    group.add_argument("--perturbation_results_path", nargs="*", default=None,
                       help="override perturbation association table(s) (default: all "
                            "{K}_perturbation_association_results_*.txt of the run)")
    group.add_argument("--knockdown_fdr_threshold", type=float, default=0.05)
    group.add_argument("--motif_min_universe_genes", type=int, default=1000,
                       help="error if fewer program genes than this have motif hits (e.g. gene ids vs symbols)")


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True, help="PerturbNMF output directory")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--K", nargs="+", type=int, required=True)
    parser.add_argument("--sel_threshs", nargs="+", type=float, required=True)
    parser.add_argument("--n_top", type=int, default=300, help="program genes per program (t-test)")
    parser.add_argument("--gene_spectra_score_path", default=None,
                        help="override programs x genes score table (single K / threshold only)")
    parser.add_argument("--skip_existing", action="store_true")
    add_motif_arguments(parser)
    args = parser.parse_args(argv)
    if args.gene_spectra_score_path and len(args.K) * len(args.sel_threshs) > 1:
        parser.error("--gene_spectra_score_path needs a single --K and --sel_threshs")
    return args


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def threshold_label(sel_thresh: float) -> str:
    return str(sel_thresh).replace(".", "_")


def evaluation_folder(args, k: int, sel_thresh: float) -> str:
    return os.path.join(args.out_dir, args.run_name, "Evaluation", f"{k}_{threshold_label(sel_thresh)}")


def motif_output_paths(args, k: int, sel_thresh: float) -> Dict[str, str]:
    folder = evaluation_folder(args, k, sel_thresh)
    return {"motif_enrichment": os.path.join(folder, f"{k}_motif_enrichment.txt"),
            "candidate_tfs": os.path.join(folder, f"{k}_candidate_tfs.txt"),
            "config": os.path.join(folder, f"{k}_motif_enrichment_config.yml"),
            "motif_logos": os.path.join(folder, f"{k}_motif_logos.json"),
            "finemo_pattern_names": os.path.join(folder, f"{k}_finemo_pattern_names.tsv")}


def gene_spectra_score_path(args, k: int, sel_thresh: float) -> str:
    if getattr(args, "gene_spectra_score_path", None):
        return args.gene_spectra_score_path
    return os.path.join(args.out_dir, args.run_name, "Inference",
                        f"Inference.gene_spectra_score.k_{k}.dt_{threshold_label(sel_thresh)}.txt")


def perturbation_results_paths(args, k: int, sel_thresh: float) -> List[str]:
    if args.perturbation_results_path is not None:
        return list(args.perturbation_results_path)
    return sorted(glob.glob(os.path.join(evaluation_folder(args, k, sel_thresh),
                                         f"{k}_perturbation_association_results_*.txt")))


def unique_temporary_path(target: str) -> str:
    """Sibling of ``target`` unique to this process (same file system, so os.rename is atomic)."""
    return f"{target}.tmp-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def build_directory_atomically(target_dir: str, build_into: Callable[[str], None]) -> str:
    """Create ``target_dir`` by calling ``build_into(temporary_dir)`` and renaming it into place.

    Safe for concurrent jobs (e.g. K array tasks sharing one cache): each builds in its own temporary
    sibling; the first rename wins and the others discard their copy and use the winner's. A target
    directory therefore only ever holds a complete build (a ``DONE`` marker is written before the
    rename; a target without it is a partial build left by an older version and is replaced).
    """
    done_marker = os.path.join(target_dir, "DONE")
    if os.path.exists(done_marker):
        return target_dir
    temporary_dir = unique_temporary_path(target_dir)
    os.makedirs(temporary_dir)
    try:
        build_into(temporary_dir)
        open(os.path.join(temporary_dir, "DONE"), "w").close()
        if os.path.isdir(target_dir) and not os.path.exists(done_marker):
            logger.warning("replacing incomplete cache directory %s", target_dir)
            shutil.rmtree(target_dir, ignore_errors=True)
        try:
            os.rename(temporary_dir, target_dir)
        except OSError:
            if not os.path.exists(done_marker):
                raise
            logger.info("%s was built concurrently by another job; using it", target_dir)
    finally:
        if os.path.exists(temporary_dir):
            shutil.rmtree(temporary_dir, ignore_errors=True)
    return target_dir


def write_file_atomically(target: str, write_to: Callable[[str], None]) -> str:
    """``write_to(temporary_path)`` then os.replace onto ``target`` (concurrent writers: last one wins,
    readers never see a partial file)."""
    temporary = unique_temporary_path(target)
    try:
        write_to(temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def read_manifest(manifest_path: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    for column in ("dataset_accession", "assembly", "local_path", "download_url"):
        if column not in manifest.columns:
            manifest[column] = ""
    manifest["rank_number"] = pd.to_numeric(manifest["rank"], errors="coerce")
    return manifest.sort_values("rank_number", kind="stable")


def select_manifest_rows(manifest: pd.DataFrame, resource_types: List[str]) -> Dict[str, Optional[pd.Series]]:
    """Best-ranked row per resource type, all from ONE dataset when several types are requested.

    For several types (Fi-NeMo instances + report), candidates of the first type are tried in rank
    order and the first whose ``dataset_accession`` also has every other type wins (each other type:
    its best-ranked row of that dataset). Rows without an accession (user overrides) pair with the
    best row of the other types. If no dataset has all types, the rank-1 rows are used (warned).
    """
    rows_by_type = {resource_type: manifest[manifest["resource_type"] == resource_type]
                    for resource_type in resource_types}
    first_type, other_types = resource_types[0], resource_types[1:]
    for _, candidate in rows_by_type[first_type].iterrows():
        accession = candidate["dataset_accession"]
        selected = {first_type: candidate}
        for other_type in other_types:
            others = rows_by_type[other_type]
            if accession:
                others = others[others["dataset_accession"].isin([accession, ""])]
            selected[other_type] = others.iloc[0] if len(others) else None
        if all(row is not None for row in selected.values()):
            return selected
    fallback = {resource_type: (rows.iloc[0] if len(rows) else None) for resource_type, rows in rows_by_type.items()}
    if len(resource_types) > 1 and all(row is not None for row in fallback.values()):
        logger.warning("no single dataset in the manifest has all of %s; using the rank-1 row of each", resource_types)
    return fallback


def fetch_manifest_row_path(row: Optional[pd.Series], cache_dir: str) -> Optional[str]:
    """A manifest row's local_path if it exists, else its download_url downloaded once into
    ``cache_dir/downloads`` (unique temporary name + atomic rename, safe for concurrent jobs)."""
    if row is None:
        return None
    if row.get("local_path") and os.path.exists(row["local_path"]):
        return row["local_path"]
    url = row.get("download_url", "")
    if not url:
        return None
    if os.path.exists(url):
        return url
    download_dir = os.path.join(cache_dir, "downloads")
    os.makedirs(download_dir, exist_ok=True)
    local = os.path.join(download_dir, os.path.basename(url.split("?")[0]))
    if not os.path.exists(local):
        logger.info("downloading %s -> %s", url, local)
        write_file_atomically(local, lambda temporary: urllib.request.urlretrieve(url, temporary))
    return local


def extract_if_tar(path: str, cache_dir: str) -> str:
    """Directory holding the contents of a .tar / .tar.gz (extracted once into the cache, atomically);
    other paths unchanged."""
    if not (path.endswith(".tar") or path.endswith(".tar.gz") or path.endswith(".tgz")):
        return path
    name = os.path.basename(path).split(".tar")[0].split(".tgz")[0]
    target = os.path.join(cache_dir, "extracted", name)
    if os.path.exists(os.path.join(target, ".extracted")):      # written by versions before atomic extraction
        return target

    def extract_into(directory: str) -> None:
        logger.info("extracting %s -> %s", path, target)
        with tarfile.open(path) as archive:
            archive.extractall(directory)

    os.makedirs(os.path.dirname(target), exist_ok=True)
    return build_directory_atomically(target, extract_into)


def resolve_resources(args, cache_dir: str) -> Dict[str, Optional[str]]:
    """enhancer_links, finemo_instances (tsv), finemo_report (html) after manifest lookup / tar extraction,
    plus ``assemblies`` (resource -> manifest assembly) for the genome-build check. Fi-NeMo instances and
    report come from the same manifest dataset (:func:`select_manifest_rows`)."""
    resolved = {"enhancer_links": args.enhancer_links, "finemo_instances": args.finemo_instances,
                "finemo_report": args.finemo_report}
    assemblies = {}
    if args.regulatory_resources_manifest:
        manifest = read_manifest(args.regulatory_resources_manifest)
        groups = [{"enhancer_links": "e2g_links"}]
        if args.motif_source != "fimo":
            groups.append({"finemo_instances": "motif_instances", "finemo_report": "motif_report"})
        for group in groups:
            missing = {key: resource_type for key, resource_type in group.items() if not resolved[key]}
            if not missing:
                continue
            rows = select_manifest_rows(manifest, list(missing.values()))
            for key, resource_type in missing.items():
                row = rows[resource_type]
                resolved[key] = fetch_manifest_row_path(row, cache_dir)
                if row is not None:
                    assemblies[key] = row.get("assembly", "")
                    logger.info("manifest %s (dataset %s, assembly %s) -> %s", resource_type,
                                row.get("dataset_accession", ""), row.get("assembly", ""), resolved[key])
    resolved["finemo_annotation"] = getattr(args, "finemo_annotation", None)
    resolved["finemo_motifs"] = getattr(args, "finemo_motifs", None)
    if args.motif_source in ("finemo", "both"):
        if resolved["finemo_annotation"]:
            resolved["finemo_report"] = None          # a local annotation table replaces the report
        if not (resolved["finemo_instances"] and (resolved["finemo_report"] or resolved["finemo_annotation"])):
            raise SystemExit("--motif_source finemo/both needs --finemo_instances and --finemo_report or "
                             "--finemo_annotation (or a manifest with motif_instances / motif_report rows)")
        instances = extract_if_tar(resolved["finemo_instances"], cache_dir)
        resolved["finemo_instances"] = (motif_hit_calling.find_finemo_instances_file(
            instances, args.finemo_head, args.finemo_lambda) if os.path.isdir(instances) else instances)
        if resolved["finemo_report"]:
            report = extract_if_tar(resolved["finemo_report"], cache_dir)
            resolved["finemo_report"] = (motif_hit_calling.find_finemo_report_file(report, args.finemo_head)
                                         if os.path.isdir(report) else report)
        if resolved["finemo_motifs"]:
            resolved["finemo_motifs"] = extract_if_tar(resolved["finemo_motifs"], cache_dir)
    resolved["assemblies"] = assemblies
    return resolved


def check_resource_genome_builds(args, resources, sources: List[str]) -> None:
    """Raise if the genome FASTA (chr1 length), manifest assemblies or build names in input file names
    disagree with ``--genome_build``."""
    observed = {f"{key} (manifest assembly)": assembly for key, assembly in resources.get("assemblies", {}).items()}
    for key, path in [("gene_annotation", args.gene_annotation), ("enhancer_links", resources.get("enhancer_links")),
                      ("finemo_instances", resources.get("finemo_instances")), ("genome_fasta", args.genome_fasta)]:
        observed[f"{key} (file name)"] = motif_hit_calling.infer_genome_build_from_path(path)
    if "fimo" in sources and args.genome_fasta and os.path.exists(args.genome_fasta):
        observed["genome_fasta (chr1 length)"] = motif_hit_calling.infer_genome_build_from_fasta(args.genome_fasta)
    motif_hit_calling.check_genome_builds(args.genome_build, observed)


# ---------------------------------------------------------------------------
# Hit tables (cached)
# ---------------------------------------------------------------------------

def file_fingerprint(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    status = os.stat(path)
    return {"path": os.path.abspath(path), "size": status.st_size, "mtime": int(status.st_mtime)}


def build_regions(element_type: str, args, resources) -> pd.DataFrame:
    if element_type == "promoter":
        is_bed = ".bed" in os.path.basename(args.gene_annotation)
        gene_tss = (motif_hit_calling.read_bed_gene_tss(args.gene_annotation, args.promoter_window_mode) if is_bed
                    else motif_hit_calling.read_gtf_gene_tss(args.gene_annotation))
        return motif_hit_calling.build_promoter_regions(gene_tss, args.promoter_upstream, args.promoter_downstream,
                                                        args.promoter_window_mode)
    links = motif_hit_calling.read_enhancer_gene_links(resources["enhancer_links"], args.link_format,
                                                       args.link_score_threshold, args.link_score_column,
                                                       merge_overlapping=args.merge_overlapping_links)
    return motif_hit_calling.build_enhancer_regions(links)


def hit_table_parameters(element_type: str, source: str, args, resources) -> dict:
    parameters = {"version": HIT_TABLE_VERSION, "element_type": element_type, "motif_source": source,
                  "genome_build": motif_hit_calling.normalize_genome_build(args.genome_build)}
    if element_type == "promoter":
        parameters.update(gene_annotation=file_fingerprint(args.gene_annotation),
                          promoter_window_mode=args.promoter_window_mode,
                          promoter_upstream=args.promoter_upstream, promoter_downstream=args.promoter_downstream)
    else:
        parameters.update(enhancer_links=file_fingerprint(resources["enhancer_links"]), link_format=args.link_format,
                          link_score_threshold=args.link_score_threshold, link_score_column=args.link_score_column)
        if args.merge_overlapping_links:     # only when set, so existing caches keep their key
            parameters["merge_overlapping_links"] = True
        link_format = (motif_hit_calling.detect_link_format(resources["enhancer_links"])
                       if args.link_format == "auto" else args.link_format)
        if link_format == "bedpe":   # bedpe elements get their promoter class from the target TSS
            parameters["bedpe_self_promoter_window"] = motif_hit_calling.BEDPE_SELF_PROMOTER_WINDOW
    if source == "fimo":
        parameters.update(motif_hit_calling.describe_fimo_backend(args.fimo_backend, args.fimo_binary))
        parameters.update(genome_fasta=file_fingerprint(args.genome_fasta), motif_file=file_fingerprint(args.motif_file),
                          meme_default_mode=args.meme_default_mode,
                          scan_threshold=max(args.promoter_pvalue_threshold, args.enhancer_pvalue_threshold),
                          n_chunks=args.n_chunks if args.meme_default_mode else None)
    else:
        parameters.update(finemo_instances=file_fingerprint(resources["finemo_instances"]),
                          finemo_report=file_fingerprint(resources["finemo_report"]),
                          finemo_annotation=file_fingerprint(resources.get("finemo_annotation")),
                          finemo_pattern_prefixes=list(args.finemo_pattern_prefixes),
                          finemo_qvalue_threshold=args.finemo_qvalue_threshold)
    return parameters


def resolve_motif_database(args) -> str:
    """Kind of the FIMO motif database (``motifcompendium`` | ``hocomoco_v11`` | ``custom``); fills
    ``args.motif_file`` from ``--motif_db`` when it is not given (idempotent)."""
    motif_db = getattr(args, "motif_db", None) or DEFAULT_MOTIF_DB
    kind = motif_db if motif_db in MOTIF_DATABASE_ENV_VARS else "custom"
    if not getattr(args, "motif_file", None):
        args.motif_file = MOTIF_DATABASE_FILES.get(motif_db) if kind != "custom" else motif_db
    return kind


def check_required_inputs(args, sources: List[str]) -> None:
    """Exit with a clear message when a site-specific input needed by this run is neither given nor set via its
    environment variable (there are no built-in paths)."""
    missing = []
    fimo_scans = [element_type for element_type in args.motif_element_types
                  if not (args.promoter_hits if element_type == "promoter" else args.enhancer_hits)]
    promoter_regions_needed = "promoter" in args.motif_element_types and (
        "finemo" in sources or "promoter" in fimo_scans)
    if promoter_regions_needed and not args.gene_annotation:
        missing.append("--gene_annotation (or $PERTURBNMF_GENE_ANNOTATION)")
    if "fimo" in sources and fimo_scans:
        if not args.genome_fasta:
            missing.append("--genome_fasta (or $PERTURBNMF_GENOME_FASTA)")
        if not args.motif_file:
            variable = MOTIF_DATABASE_ENV_VARS.get(getattr(args, "motif_db", None) or DEFAULT_MOTIF_DB)
            missing.append(f"--motif_file (or ${variable})" if variable else "--motif_file")
    if missing:
        raise SystemExit("missing motif-enrichment inputs: " + ", ".join(missing))


def collapses_motif_ids(source: str, motif_database_kind: str) -> bool:
    """HOCOMOCO-style ids are collapsed to the TF at the first ``_``; MotifCompendium clusters and Fi-NeMo
    tables keep the full id (one test per cluster)."""
    return source == "fimo" and motif_database_kind != "motifcompendium"


def name_finemo_patterns_for_run(args, resources) -> pd.DataFrame:
    """Pattern -> cluster name, family, TOMTOM q-value and database TF list (local annotation table or
    ENCODE report + bundled MotifCompendium metadata)."""
    if resources.get("finemo_annotation"):
        return motif_hit_calling.name_finemo_patterns_from_annotation_table(resources["finemo_annotation"],
                                                                            args.finemo_pattern_prefixes)
    annotation = motif_hit_calling.read_finemo_motif_annotation(resources["finemo_report"])
    names = motif_hit_calling.name_finemo_patterns(annotation, args.finemo_qvalue_threshold)
    prefixes = tuple(args.finemo_pattern_prefixes)
    names = names[names["pattern_id"].str.startswith(prefixes)].reset_index(drop=True)
    return motif_hit_calling.add_database_tfs_to_pattern_names(names, getattr(args, "motifcompendium_metadata", None))


def build_or_reuse_hit_table(element_type: str, source: str, args, resources, cache_dir: str) -> str:
    """Path of a FIMO-format hit table for (element type, source), built into the cache if needed.

    The table directory is built under a temporary name and renamed into place
    (:func:`build_directory_atomically`), so concurrent K jobs sharing the cache never read partial tables.
    """
    user_table = args.promoter_hits if element_type == "promoter" else args.enhancer_hits
    if source == "fimo" and user_table:
        return user_table
    parameters = hit_table_parameters(element_type, source, args, resources)
    digest = hashlib.sha1(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:10]
    table_dir = os.path.join(cache_dir, f"{element_type}_{source}_{digest}")
    if os.path.exists(os.path.join(table_dir, "DONE")):
        logger.info("reusing cached %s %s hits: %s", element_type, source, table_dir)
        return os.path.join(table_dir, "motif_hits.tsv")

    def build_into(directory: str) -> None:
        hits_path = os.path.join(directory, "motif_hits.tsv")
        with open(os.path.join(directory, "params.json"), "w") as handle:
            json.dump(parameters, handle, indent=2, sort_keys=True)
        regions = build_regions(element_type, args, resources)
        regions[["chrom", "start", "end", "sequence_name", "gene", "strand"]].to_csv(
            os.path.join(directory, "regions.bed"), sep="\t", header=False, index=False)
        logger.info("%d %s regions (%d genes)", len(regions), element_type, regions["gene"].nunique())
        if source == "fimo":
            n_hits = motif_hit_calling.scan_regions_with_fimo(
                regions, args.genome_fasta, args.motif_file, hits_path, backend=parameters["fimo_backend"],
                threshold=parameters["scan_threshold"], fimo_binary=args.fimo_binary,
                meme_text_mode=not args.meme_default_mode, n_chunks=args.n_chunks, n_jobs=args.n_jobs,
                work_dir=os.path.join(directory, "fimo_chunks"))
            shutil.rmtree(os.path.join(directory, "fimo_chunks"), ignore_errors=True)
        else:
            pattern_names = name_finemo_patterns_for_run(args, resources)
            pattern_names.to_csv(os.path.join(directory, "finemo_pattern_names.tsv"), sep="\t", index=False)
            if resources.get("finemo_annotation"):      # local table: keep the annotated (pos) motifs
                finemo_hits = motif_hit_calling.read_finemo_hits(resources["finemo_instances"])
                finemo_hits = finemo_hits[finemo_hits["motif_name"].astype(str).isin(set(pattern_names["pattern_id"]))]
            else:
                finemo_hits = motif_hit_calling.read_finemo_hits(resources["finemo_instances"],
                                                                 pattern_prefixes=args.finemo_pattern_prefixes)
            shared_chromosomes = set(finemo_hits["chrom"]) & set(regions["chrom"])
            if len(shared_chromosomes) < 5:
                logger.warning("Fi-NeMo hits and %s regions share only %d chromosomes -- genome builds match?",
                               element_type, len(shared_chromosomes))
            hits = motif_hit_calling.call_hits_from_finemo(
                finemo_hits, regions, motif_hit_calling.build_finemo_motif_name_map(pattern_names))
            if hits.empty:
                raise RuntimeError(f"0 of {len(finemo_hits)} Fi-NeMo hits fall inside the {len(regions)} {element_type} "
                                   f"regions: check that {resources['finemo_instances']} and the regions share a "
                                   f"genome build and chromosome naming")
            hits.to_csv(hits_path, sep="\t", index=False, na_rep="NA")
            n_hits = len(hits)
        logger.info("%d %s %s hits -> %s", n_hits, element_type, source, table_dir)

    build_directory_atomically(table_dir, build_into)
    return os.path.join(table_dir, "motif_hits.tsv")


def read_hit_counts(hits_path: str, element_type: str, source: str, args, cache_dir: str,
                    collapse_motif_ids: bool = True) -> pd.DataFrame:
    """Gene x motif counts of a hit table (all genes), cached next to the table as a pickle (atomic write).
    ``collapse_motif_ids`` False keeps full motif ids (MotifCompendium clusters, Fi-NeMo)."""
    threshold = None if source == "finemo" else (
        args.promoter_pvalue_threshold if element_type == "promoter" else args.enhancer_pvalue_threshold)
    use_motif_alt_id = source != "finemo"        # Fi-NeMo motif_alt_id is the raw pattern id
    key = hashlib.sha1(json.dumps([file_fingerprint(hits_path), element_type, threshold, use_motif_alt_id,
                                   collapse_motif_ids], sort_keys=True).encode()).hexdigest()[:10]
    counts_path = os.path.join(cache_dir, "counts", f"{element_type}_{source}_{key}.pkl")
    if os.path.exists(counts_path):
        return pd.read_pickle(counts_path)
    parser = None if element_type == "promoter" else motif_enrichment.parse_abc_sequence_name
    hits = motif_enrichment.read_fimo_hits(hits_path, threshold, sequence_name_parser=parser,
                                           use_motif_alt_id=use_motif_alt_id, collapse_motif_ids=collapse_motif_ids)
    excluded = () if element_type == "promoter" else ("promoter",)
    counts = motif_enrichment.count_hits_per_gene_tf(hits, excluded_element_classes=excluded)
    logger.info("%s %s: %d hits (threshold %s) -> %d genes x %d TFs", element_type, source, len(hits),
                threshold, *counts.shape)
    os.makedirs(os.path.dirname(counts_path), exist_ok=True)
    write_file_atomically(counts_path, lambda temporary: counts.to_pickle(temporary, compression=None))
    return counts


ENSEMBL_GENE_ID_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")


def strip_gene_id_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0] if ENSEMBL_GENE_ID_RE.match(str(gene_id)) else str(gene_id)


def read_gtf_gene_id_to_symbol(gtf_path: str) -> Dict[str, str]:
    """Ensembl gene id (version stripped) -> gene_name from the ``gene`` records of a GTF."""
    mapping = {}
    with motif_hit_calling.open_text(gtf_path) as handle:
        for line in handle:
            fields = line.split("\t", 8)
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attributes = dict(motif_hit_calling.GTF_ATTRIBUTE_RE.findall(fields[8]))
            if "gene_id" in attributes and "gene_name" in attributes:
                mapping.setdefault(strip_gene_id_version(attributes["gene_id"]), attributes["gene_name"])
    return mapping


def read_h5mu_gene_id_to_symbol(h5mu_path: str, data_key: str, gene_names_key: str) -> Dict[str, str]:
    """var_names (version stripped) -> var[gene_names_key] of the ``data_key`` modality of an h5mu."""
    import anndata
    import h5py
    read_elem = getattr(getattr(anndata, "io", None), "read_elem", None) or anndata.experimental.read_elem
    with h5py.File(h5mu_path, "r") as handle:
        var = read_elem(handle[f"mod/{data_key}/var"])
    if gene_names_key not in var.columns:
        return {}
    return {strip_gene_id_version(gene_id): str(symbol) for gene_id, symbol in zip(var.index, var[gene_names_key])}


def convert_gene_ids_to_symbols(gene_spectra_score: pd.DataFrame, args, h5mu_path: Optional[str] = None) -> pd.DataFrame:
    """If most gene columns are Ensembl gene ids, rename them to symbols (hit tables are keyed by symbol).

    Mapping source: the GTF ``--gene_annotation`` (gene_id -> gene_name), else ``var[--gene_names_key]``
    of the run's h5mu (when the evaluation pipeline supplies ``gene_names_key``). Unmapped ids are kept;
    duplicate symbols keep the first column.
    """
    columns = pd.Index(gene_spectra_score.columns.astype(str))
    is_gene_id = columns.str.match(ENSEMBL_GENE_ID_RE.pattern)
    if is_gene_id.mean() <= 0.5:
        return gene_spectra_score
    mapping = {}
    if args.gene_annotation and ".bed" not in os.path.basename(args.gene_annotation):
        mapping = read_gtf_gene_id_to_symbol(args.gene_annotation)
        source = args.gene_annotation
    if not mapping and h5mu_path and os.path.exists(h5mu_path) and getattr(args, "gene_names_key", None):
        mapping = read_h5mu_gene_id_to_symbol(h5mu_path, getattr(args, "data_key", "rna"), args.gene_names_key)
        source = h5mu_path
    if not mapping:
        logger.warning("gene_spectra_score columns are Ensembl ids but no id -> symbol mapping is available")
        return gene_spectra_score
    symbols = [mapping.get(strip_gene_id_version(column), column) for column in columns]
    n_mapped = sum(symbol != column for symbol, column in zip(symbols, columns))
    renamed = gene_spectra_score.set_axis(symbols, axis=1)
    duplicated = renamed.columns.duplicated(keep="first")
    logger.info("gene ids -> symbols via %s: %d of %d mapped, %d duplicate symbols dropped", source, n_mapped,
                len(columns), int(duplicated.sum()))
    return renamed.loc[:, ~duplicated]


def check_universe_size(counts: pd.DataFrame, gene_spectra_score: pd.DataFrame, hit_genes, element_type: str,
                        source: str, min_genes: int) -> None:
    """Raise if fewer than ``min_genes`` program genes have motif hits (usually a gene naming mismatch)."""
    if len(counts) >= min_genes:
        return
    raise SystemExit(
        f"{element_type} {source}: only {len(counts)} of {gene_spectra_score.shape[1]} program genes have motif hits "
        f"(--motif_min_universe_genes {min_genes}). Program genes look like {list(gene_spectra_score.columns[:3])}, "
        f"hit-table genes like {list(hit_genes[:3])}: gene names must match (symbols), and the regions must "
        f"cover the program genes")


def restrict_counts_to_genes(counts: pd.DataFrame, genes) -> pd.DataFrame:
    """Rows of expressed genes, then only TFs with >= 1 hit left (same as counting on the subset)."""
    subset = counts[counts.index.isin(set(genes))]
    return subset.loc[:, subset.sum(axis=0) > 0]


# ---------------------------------------------------------------------------
# Enrichment + candidate TFs
# ---------------------------------------------------------------------------

def compute_enrichment(counts: pd.DataFrame, gene_spectra_score: pd.DataFrame, element_type: str, args) -> pd.DataFrame:
    if args.motif_method == "ttest":
        program_genes = motif_enrichment.select_top_program_genes(gene_spectra_score, n_top=args.n_top)
        return motif_enrichment.test_motif_enrichment_ttest(counts, program_genes, element_type)
    return motif_enrichment.test_motif_enrichment_correlation(counts, gene_spectra_score, element_type,
                                                             args.motif_correlation)


def read_perturbation_results(paths: List[str]) -> Optional[pd.DataFrame]:
    if not paths:
        return None
    tables = [pd.read_csv(path, sep="\t") for path in paths]
    return pd.concat(tables, ignore_index=True)


def build_motif_vocabulary(tfs, source: str, motif_database_kind: str, args,
                           pattern_names: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per tested motif (the ``tf`` column): motif_family, database_tfs (list of TF names from the motif
    database, or None to map the name with HOCOMOCO v11) and motif_match_qvalue (Fi-NeMo: best TOMTOM q of
    the patterns named by this cluster).

    FIMO + MotifCompendium: cluster name without ``_<n>`` and the cluster's TF list (bundled metadata
    release that contains the most names, or ``--motifcompendium_metadata``). FIMO + HOCOMOCO v11 / other
    MEME: TFClass family from the HOCOMOCO v11 annotation (the name itself when unknown). Fi-NeMo: the
    pattern-name table.
    """
    tfs = [str(tf) for tf in pd.unique(pd.Series(list(tfs), dtype=str))]
    if source == "finemo":
        names = pattern_names if pattern_names is not None else pd.DataFrame(columns=["tf"])
        rows = []
        for tf in tfs:
            patterns = names[names["tf"] == tf]
            database_tfs = []
            for tf_list in patterns.get("database_tfs", pd.Series(dtype=str)).fillna(""):
                database_tfs.extend(motif_hit_calling.split_tf_list(tf_list))
            qvalues = pd.to_numeric(patterns.get("top_match_qvalue", pd.Series(dtype=float)), errors="coerce")
            rows.append({"tf": tf,
                         "motif_family": patterns["motif_family"].iloc[0] if len(patterns) else tf,
                         "database_tfs": list(dict.fromkeys(database_tfs)),
                         "motif_match_qvalue": qvalues.min() if qvalues.notna().any() else float("nan")})
        return pd.DataFrame(rows, columns=["tf", "motif_family", "database_tfs", "motif_match_qvalue"])
    if motif_database_kind == "motifcompendium":
        metadata_path = (getattr(args, "motifcompendium_metadata", None)
                         or motif_hit_calling.select_motifcompendium_metadata(tfs))
        tf_lists = motif_hit_calling.read_motifcompendium_metadata(metadata_path).set_index("name")["database_tfs"]
        n_known = sum(tf in tf_lists.index for tf in tfs)
        if tfs and n_known < 0.5 * len(tfs):
            raise SystemExit(f"--motif_db motifcompendium, but only {n_known} of {len(tfs)} FIMO motif names (e.g. "
                             f"{tfs[:3]}) are MotifCompendium clusters in {os.path.basename(metadata_path)}: pass "
                             f"--motif_db hocomoco_v11 (or the MEME file) for hit tables scanned with another database")
        return pd.DataFrame({"tf": tfs,
                             "motif_family": [motif_hit_calling.collapse_motifcompendium_name(tf) for tf in tfs],
                             "database_tfs": [motif_hit_calling.split_tf_list(tf_lists.get(tf, "")) for tf in tfs]})
    families = nominate_candidate_tfs.read_hocomoco_tf_families().set_index("tf")["motif_family"].to_dict()
    return pd.DataFrame({"tf": tfs, "motif_family": [families.get(tf, tf) for tf in tfs],
                         "database_tfs": [None] * len(tfs)})


def tf_gene_symbols_from_vocabulary(vocabulary: pd.DataFrame, genes) -> pd.DataFrame:
    """Motif -> TF gene rows: the database TF list (MotifCompendium / Fi-NeMo) restricted to expressed genes;
    HOCOMOCO-named motifs through the HOCOMOCO v11 annotation."""
    has_list = vocabulary["database_tfs"].map(lambda value: isinstance(value, list))
    tables = []
    if has_list.any():
        tables.append(nominate_candidate_tfs.map_motif_families_to_gene_symbols(
            dict(zip(vocabulary.loc[has_list, "tf"], vocabulary.loc[has_list, "database_tfs"])), genes))
    if (~has_list).any():
        tables.append(nominate_candidate_tfs.map_tf_names_to_gene_symbols(vocabulary.loc[~has_list, "tf"]))
    if not tables:
        return pd.DataFrame(columns=["tf", "tf_gene_symbol", "tf_gene_symbol_source"])
    return pd.concat(tables, ignore_index=True)


def add_vocabulary_columns(results: pd.DataFrame, vocabulary: pd.DataFrame, source: str) -> pd.DataFrame:
    """``motif_family`` after ``tf`` (and ``motif_match_qvalue`` for Fi-NeMo) on the enrichment table."""
    columns = ["tf", "motif_family"] + (["motif_match_qvalue"] if source == "finemo" else [])
    table = results.drop(columns=[c for c in columns[1:] if c in results.columns]).merge(
        vocabulary[columns], on="tf", how="left")
    table["motif_family"] = table["motif_family"].fillna(table["tf"])
    order = list(results.columns)
    order.insert(order.index("tf") + 1, "motif_family")
    return table[order + (["motif_match_qvalue"] if source == "finemo" else [])]


def run_motif_enrichment_for_programs(gene_spectra_score: pd.DataFrame, args, cache_dir: str,
                                      perturbation_results: Optional[pd.DataFrame] = None):
    """Enrichment + candidate TFs for one programs x genes table. Returns (results, candidates, info)."""
    motif_database_kind = resolve_motif_database(args)
    resources = resolve_resources(args, cache_dir)
    sources = ["fimo", "finemo"] if args.motif_source == "both" else [args.motif_source]
    check_required_inputs(args, sources)
    check_resource_genome_builds(args, resources, sources)
    pattern_names = name_finemo_patterns_for_run(args, resources) if "finemo" in sources else None
    expressed_genes = gene_spectra_score.columns
    results, candidates, hit_tables, vocabularies = [], [], {}, {}
    for source in sources:
        source_results = []
        for element_type in args.motif_element_types:
            if element_type == "enhancer" and not (resources["enhancer_links"]
                                                   or (source == "fimo" and args.enhancer_hits)):
                logger.warning("no enhancer links given: skipping enhancer motif enrichment (%s)", source)
                continue
            hits_path = build_or_reuse_hit_table(element_type, source, args, resources, cache_dir)
            hit_tables[f"{element_type}_{source}"] = hits_path
            all_counts = read_hit_counts(hits_path, element_type, source, args, cache_dir,
                                         collapse_motif_ids=collapses_motif_ids(source, motif_database_kind))
            counts = restrict_counts_to_genes(all_counts, expressed_genes)
            logger.info("%s %s universe: %d genes x %d motifs", element_type, source, *counts.shape)
            check_universe_size(counts, gene_spectra_score, all_counts.index, element_type, source,
                                args.motif_min_universe_genes)
            source_results.append(compute_enrichment(counts, gene_spectra_score, element_type, args))
        if not source_results:
            continue
        table = motif_enrichment.flag_significant(pd.concat(source_results, ignore_index=True),
                                                  args.motif_fdr_threshold, args.motif_method)
        vocabulary = build_motif_vocabulary(table["tf"], source, motif_database_kind, args, pattern_names)
        vocabularies[source] = vocabulary
        table = add_vocabulary_columns(table, vocabulary, source)
        if args.motif_source == "both":
            table["motif_source"] = source
        results.append(table)
        significant = vocabulary[vocabulary["tf"].isin(set(table.loc[table["significant"], "tf"]))]
        candidates.append(nominate_candidate_tfs.nominate_candidate_tfs(
            table, gene_spectra_score, perturbation_results,
            tf_gene_symbols=tf_gene_symbols_from_vocabulary(significant, expressed_genes),
            motif_fdr_threshold=args.motif_fdr_threshold, n_top_genes=args.n_top,
            knockdown_fdr_threshold=args.knockdown_fdr_threshold))
    if not results:
        raise SystemExit("no motif enrichment computed (no element types with inputs)")
    results = pd.concat(results, ignore_index=True)
    info = {"resources": resources, "hit_tables": hit_tables, "motif_database": motif_database_kind,
            "motif_file": args.motif_file,
            "motif_logos": motif_logos.build_motif_logos_for_run(results, args, resources, motif_database_kind,
                                                                 pattern_names)}
    if pattern_names is not None:
        info["finemo_pattern_names"] = pattern_names
    return results, pd.concat(candidates, ignore_index=True), info


def run_motif_enrichment_for_k(args, k: int, sel_thresh: float) -> Dict[str, str]:
    """Run for one K / density threshold of a PerturbNMF run; returns the output paths."""
    outputs = motif_output_paths(args, k, sel_thresh)
    if getattr(args, "skip_existing", False) and all(os.path.exists(outputs[key]) for key in
                                                     ("motif_enrichment", "candidate_tfs")):
        logger.info("[K=%s, thresh=%s] motif outputs exist; skipping", k, sel_thresh)
        return outputs
    os.makedirs(os.path.dirname(outputs["motif_enrichment"]), exist_ok=True)
    cache_dir = args.motif_hit_cache_dir or os.path.join(args.out_dir, args.run_name, "Evaluation", "motif_hits")
    os.makedirs(cache_dir, exist_ok=True)
    score_path = gene_spectra_score_path(args, k, sel_thresh)
    gene_spectra_score = pd.read_csv(score_path, sep="\t", index_col=0)
    h5mu_path = os.path.join(args.out_dir, args.run_name, "Inference", "adata",
                             f"cNMF_{k}_{threshold_label(sel_thresh)}.h5mu")
    gene_spectra_score = convert_gene_ids_to_symbols(gene_spectra_score, args, h5mu_path)
    perturbation_paths = perturbation_results_paths(args, k, sel_thresh)
    if not perturbation_paths:
        logger.warning("no perturbation association results found: candidate TFs lack knockdown evidence")
    results, candidates, info = run_motif_enrichment_for_programs(
        gene_spectra_score, args, cache_dir, read_perturbation_results(perturbation_paths))
    results.to_csv(outputs["motif_enrichment"], sep="\t", index=False)
    candidates.to_csv(outputs["candidate_tfs"], sep="\t", index=False)
    if "finemo_pattern_names" in info:
        info["finemo_pattern_names"].to_csv(outputs["finemo_pattern_names"], sep="\t", index=False)
    write_file_atomically(outputs["motif_logos"], lambda temporary: motif_logos.write_motif_logos(
        info["motif_logos"], temporary))
    config = {"arguments": {key: value for key, value in vars(args).items()},
              "K": k, "sel_thresh": sel_thresh, "gene_spectra_score": score_path,
              "motif_database": info["motif_database"], "motif_file": info["motif_file"],
              "perturbation_results": perturbation_paths, "resources": info["resources"],
              "hit_tables": info["hit_tables"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "n_significant": {f"{element}{'_' + source if source else ''}": int(n) for (element, source), n in
                                results.assign(motif_source=results.get("motif_source", ""))
                                .groupby(["element_type", "motif_source"])["significant"].sum().items()}}
    with open(outputs["config"], "w") as handle:      # JSON is valid YAML; avoids a pyyaml dependency
        json.dump(config, handle, indent=2, default=str)
    logger.info("[K=%s, thresh=%s] %d rows (%d significant) -> %s; %d candidate rows -> %s", k, sel_thresh,
                len(results), int(results["significant"].sum()), outputs["motif_enrichment"], len(candidates),
                outputs["candidate_tfs"])
    return outputs


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_arguments(argv)
    for sel_thresh in args.sel_threshs:
        for k in args.K:
            run_motif_enrichment_for_k(args, k, sel_thresh)


if __name__ == "__main__":
    main()
