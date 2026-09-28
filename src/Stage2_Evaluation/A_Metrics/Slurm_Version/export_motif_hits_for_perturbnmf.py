"""Export Fi-NeMo hits + MotifCompendium annotation as the motif-hit tables PerturbNMF consumes.

Input: Fi-NeMo / MotifCompendium outputs (clustered TF-MoDISco patterns, Fi-NeMo hits and reports).
Cluster naming and hit QC:
  - cluster annotation: cluster-average motif matched to MotifCompendium-Database-Human (min score 0.8);
    TFs from the database metadata; tf_motif_id = the single TF if the match names one TF,
    else the database motif name (e.g. CTCF_0), else cluster_<id>.
  - hit QC: motifs with Fi-NeMo report cwm_similarity >= 0.8; overlapping hits of the same motif
    (> 3 bp) collapsed to the highest hit_similarity.
Fi-NeMo motif_name is "{pos,neg}_patterns.<cluster_id>" (export_compendium_clustered_modisco).

Outputs in --out-dir:
  motif_annotation.tsv                   one row per compendium cluster (+ per-dataset QC columns)
  motif_hits_<dataset>.tsv.gz (+ .tbi)   one row per QC-passing hit, GRCh38, 0-based half-open
"""
import argparse
import bisect
import os
import pickle
import re
import subprocess

import numpy as np
import pandas as pd
import MotifCompendium.utils.analysis as utils_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--compendium-dir", required=True, help="dir with modisco_compendium.mc")
parser.add_argument("--finemo-dirs", nargs="+", required=True, help="dataset=path/to/finemo_unified/<dataset>_all")
parser.add_argument("--db-meme", required=True)
parser.add_argument("--db-metadata", required=True)
parser.add_argument("--min-annotation-score", type=float, default=0.8)
parser.add_argument("--min-cwm-similarity", type=float, default=0.8)
parser.add_argument("--overlap-bp", type=int, default=3)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def split_tfs(tfs):
    return [t.strip() for t in re.split(r"[,@;]", tfs) if t.strip()] if isinstance(tfs, str) else []


def build_cluster_annotation():
    with open(os.path.join(args.compendium_dir, "modisco_compendium.mc"), "rb") as fh:
        mc = pickle.load(fh)
    mc_avg = mc.cluster_averages(
        "cluster_id",
        aggregations=[("name", "count", "n_patterns"), ("model", "concat", "datasets_present"),
                      ("num_seqlets", "sum", "total_seqlets"), ("posneg", "concat", "posneg")],
        weight_col="num_seqlets",
    )
    utils_analysis.assign_label_from_pfms(mc=mc_avg, pfm_file=args.db_meme, save_col_prefix="annotation",
                                          min_score=args.min_annotation_score, save_images=False)
    clusters = mc_avg.metadata.rename(columns={"source_cluster": "cluster_id"}).copy()
    # unmatched clusters come back as "" rather than NaN
    clusters["annotation_name0"] = clusters["annotation_name0"].replace("", np.nan)
    db = pd.read_csv(args.db_metadata, sep="\t", usecols=["name", "TF"])
    db = db.rename(columns={"name": "annotation_name0", "TF": "candidate_tfs"})
    clusters = clusters.merge(db, on="annotation_name0", how="left")
    tf_lists = clusters["candidate_tfs"].map(split_tfs)
    single_tf = tf_lists.map(len) == 1
    motif_label = clusters["annotation_name0"].where(~single_tf, tf_lists.map(lambda t: t[0] if t else None))
    clusters["motif_id"] = "cluster_" + clusters["cluster_id"].astype(str)
    clusters["motif_label"] = motif_label.fillna(clusters["motif_id"])
    clusters["posneg"] = clusters["posneg"].map(lambda s: "pos" if "pos" in str(s).split(",") else "neg")
    return clusters[["motif_id", "cluster_id", "motif_label", "annotation_name0", "annotation_score0",
                     "candidate_tfs", "posneg", "n_patterns", "total_seqlets", "datasets_present"]].rename(
        columns={"annotation_name0": "database_motif", "annotation_score0": "database_match_score"})


def collapse_overlapping_hits(hits):
    """Per (chrom, motif): best hit_similarity first; drop hits overlapping an already-kept hit by > overlap_bp."""
    max_width = int((hits["end"] - hits["start"]).max())
    kept = []
    for _, group in hits.sort_values("hit_similarity", ascending=False).groupby(["chrom", "motif_id"], sort=False):
        kept_starts, kept_ends = [], []  # kept hits, sorted by start
        keep = np.zeros(len(group), dtype=bool)
        for i, (s, e) in enumerate(zip(group["start"].to_numpy(), group["end"].to_numpy())):
            j = bisect.bisect_left(kept_starts, s - max_width)
            clash = False
            while j < len(kept_starts) and kept_starts[j] < e:
                if min(e, kept_ends[j]) - max(s, kept_starts[j]) > args.overlap_bp:
                    clash = True
                    break
                j += 1
            if not clash:
                k = bisect.bisect_left(kept_starts, s)
                kept_starts.insert(k, s)
                kept_ends.insert(k, e)
                keep[i] = True
        kept.append(group[keep])
    return pd.concat(kept, ignore_index=True)


clusters = build_cluster_annotation()
label_by_motif = clusters.set_index("motif_id")

for spec in args.finemo_dirs:
    dataset, finemo_dir = spec.split("=", 1)
    report = pd.read_csv(os.path.join(finemo_dir, "finemo_report", "motif_report.tsv"), sep="\t")
    report["motif_id"] = "cluster_" + report["motif_name"].str.split(".").str[1]
    report = report.set_index("motif_id")
    clusters[f"cwm_similarity_{dataset}"] = clusters["motif_id"].map(report["cwm_similarity"])
    clusters[f"n_hits_{dataset}"] = clusters["motif_id"].map(report["num_hits_total"])
    clusters[f"pass_qc_{dataset}"] = clusters[f"cwm_similarity_{dataset}"] >= args.min_cwm_similarity
    passing = set(clusters.loc[clusters[f"pass_qc_{dataset}"], "motif_id"])

    hits = pd.read_csv(os.path.join(finemo_dir, "hits_unique.tsv"), sep="\t",
                       usecols=["chr", "start", "end", "strand", "motif_name", "hit_coefficient",
                                "hit_similarity", "hit_importance", "peak_name"])
    hits = hits.rename(columns={"chr": "chrom", "hit_coefficient": "score"})
    hits["motif_id"] = "cluster_" + hits["motif_name"].str.split(".").str[1]
    n_raw = len(hits)
    hits = hits[hits["motif_id"].isin(passing)]
    n_qc = len(hits)
    hits = collapse_overlapping_hits(hits)
    hits["motif_label"] = hits["motif_id"].map(label_by_motif["motif_label"])
    hits["posneg"] = hits["motif_name"].str.split("_").str[0]
    hits = hits[["chrom", "start", "end", "strand", "motif_id", "motif_label", "posneg", "score",
                 "hit_similarity", "hit_importance", "peak_name"]].sort_values(["chrom", "start", "end"])
    out_tsv = os.path.join(args.out_dir, f"motif_hits_{dataset}.tsv")
    with open(out_tsv, "w") as fh:
        fh.write("#")
        hits.to_csv(fh, sep="\t", index=False)
    subprocess.run(["bgzip", "-f", out_tsv], check=True)
    subprocess.run(["tabix", "-f", "-s1", "-b2", "-e3", "-0", "-c#", out_tsv + ".gz"], check=True)
    print(f"{dataset}: {n_raw} hits -> {n_qc} in {len(passing)} QC-passing motifs -> {len(hits)} after collapsing overlaps")

clusters.to_csv(os.path.join(args.out_dir, "motif_annotation.tsv"), sep="\t", index=False)
print(f"{len(clusters)} motifs; {(clusters['motif_label'] != clusters['motif_id']).sum()} annotated")
