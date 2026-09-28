#!/usr/bin/env python
"""Compare a FIMO-format motif hit table to a reference one (e.g. the fimo.tsv of a published run).

Reports, per p-value threshold: hit counts, hit-level agreement (motif_id, sequence_name, start,
stop, strand), per-motif hit counts, and gene x TF count agreement (TF = motif_id before '_'; enhancer hits summed over
elements of the target gene). Optionally restricts to sequence names present in both tables and,
with --reference_fasta + --our_regions_bed + --genome_fasta, to regions whose sequence is identical
(case-insensitive, either orientation) in both builds, so FIMO differences can be separated from region differences.

Writes a json summary to --out_json and prints it.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import motif_hit_calling  # noqa: E402

USE_COLUMNS = ["motif_id", "sequence_name", "start", "stop", "strand", "p-value"]


def read_hit_table(path, thresholds, excluded_element_classes, chunksize=5_000_000):
    """Stream a FIMO tsv; returns {threshold: (gene x TF counts, hit-key hashes, motif counts)}, sequence names."""
    counts = {t: [] for t in thresholds}
    motif_counts = {t: [] for t in thresholds}
    keys = {t: [] for t in thresholds}
    names = set()
    loosest = max(thresholds)
    reader = pd.read_csv(path, sep="\t", comment="#", usecols=USE_COLUMNS, chunksize=chunksize,
                         dtype={"motif_id": str, "sequence_name": str, "strand": str})
    for chunk in reader:
        chunk = chunk[chunk["p-value"] < loosest]
        if excluded_element_classes and chunk["sequence_name"].str.contains("|", regex=False).any():
            element_class = chunk["sequence_name"].str.split("|", n=2).str[1]
            chunk = chunk[~element_class.isin(excluded_element_classes)]
        names.update(chunk["sequence_name"].unique())
        chunk = chunk.assign(
            tf=chunk["motif_id"].str.split("_", n=1).str[0],
            gene=motif_hit_calling.target_gene_from_sequence_name(chunk["sequence_name"]).to_numpy())
        for threshold in thresholds:
            kept = chunk[chunk["p-value"] < threshold]
            counts[threshold].append(kept.groupby(["gene", "tf"]).size())
            motif_counts[threshold].append(kept["motif_id"].value_counts())
            keys[threshold].append(pd.util.hash_pandas_object(
                kept[["motif_id", "sequence_name", "start", "stop", "strand"]], index=False).to_numpy())
    result = {}
    for threshold in thresholds:
        summed = pd.concat(counts[threshold]).groupby(level=[0, 1]).sum() if counts[threshold] else pd.Series(dtype=int)
        per_motif = pd.concat(motif_counts[threshold]).groupby(level=0).sum() if motif_counts[threshold] else pd.Series(dtype=int)
        result[threshold] = (summed, np.concatenate(keys[threshold]) if keys[threshold] else np.array([], np.uint64),
                             per_motif)
    return result, names


def identical_sequence_names(reference_fasta, our_regions_bed, genome_fasta):
    """Names whose region sequence is identical (case-insensitive) in both builds, same or reverse-complement strand."""
    import pyfaidx
    reference = pyfaidx.Fasta(reference_fasta, as_raw=True, duplicate_action="first")
    genome = pyfaidx.Fasta(genome_fasta, as_raw=True)
    regions = pd.read_csv(our_regions_bed, sep="\t", header=None, usecols=[0, 1, 2, 3],
                          names=["chrom", "start", "end", "sequence_name"], dtype={"chrom": str, "sequence_name": str})
    complement = str.maketrans("ACGTN", "TGCAN")
    identical, compared, n_reverse_complement = set(), 0, 0
    for chrom, start, end, name in regions.itertuples(index=False):
        if name not in reference or chrom not in genome:
            continue
        compared += 1
        ours, theirs = genome[chrom][int(start):int(end)].upper(), reference[name][:].upper()
        if ours == theirs:
            identical.add(name)
        elif ours.translate(complement)[::-1] == theirs:
            identical.add(name)
            n_reverse_complement += 1
    return identical, compared, len(reference.keys()), n_reverse_complement


def compare_counts(ours, reference):
    joined = pd.concat({"ours": ours, "reference": reference}, axis=1).fillna(0)
    if joined.empty:
        return {}
    return {
        "n_gene_tf_pairs_union": int(len(joined)),
        "n_pairs_only_ours": int(((joined["ours"] > 0) & (joined["reference"] == 0)).sum()),
        "n_pairs_only_reference": int(((joined["ours"] == 0) & (joined["reference"] > 0)).sum()),
        "fraction_pairs_equal_count": float((joined["ours"] == joined["reference"]).mean()),
        "pearson_r": float(np.corrcoef(joined["ours"], joined["reference"])[0, 1]),
        "total_ours": int(joined["ours"].sum()), "total_reference": int(joined["reference"].sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1e-4, 1e-6])
    parser.add_argument("--exclude_element_classes", nargs="*", default=[],
                        help="drop hits in these element classes (enhancer names) before comparing")
    parser.add_argument("--reference_fasta")
    parser.add_argument("--our_regions_bed")
    parser.add_argument("--genome_fasta")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args(argv)

    ours, our_names = read_hit_table(args.ours, args.thresholds, args.exclude_element_classes)
    reference, reference_names = read_hit_table(args.reference, args.thresholds, args.exclude_element_classes)
    shared_names = our_names & reference_names
    summary = {"ours": args.ours, "reference": args.reference,
               "n_sequence_names_with_hits": {"ours": len(our_names), "reference": len(reference_names),
                                              "shared": len(shared_names)}}
    identical = None
    if args.reference_fasta and args.our_regions_bed and args.genome_fasta:
        identical, compared, n_reference, n_reverse_complement = identical_sequence_names(args.reference_fasta, args.our_regions_bed,
                                                                     args.genome_fasta)
        summary["sequence_identity"] = {"reference_records": n_reference, "names_in_both_builds": compared,
                                        "identical_sequences": len(identical),
                                        "of_which_reverse_complement": n_reverse_complement}
    for threshold in args.thresholds:
        our_counts, our_keys, our_motif_counts = ours[threshold]
        reference_counts, reference_keys, reference_motif_counts = reference[threshold]
        motifs = pd.concat({"ours": our_motif_counts, "reference": reference_motif_counts}, axis=1).fillna(0).astype(int)
        differing_motifs = motifs[motifs["ours"] != motifs["reference"]].sort_values("reference")
        shared_keys = np.intersect1d(our_keys, reference_keys)
        block = {
            "hits_ours": int(len(our_keys)), "hits_reference": int(len(reference_keys)),
            "hits_shared_exact": int(len(shared_keys)),
            "fraction_reference_hits_recovered": float(len(shared_keys) / max(len(reference_keys), 1)),
            "fraction_our_hits_in_reference": float(len(shared_keys) / max(len(our_keys), 1)),
            "gene_tf_counts_all": compare_counts(our_counts, reference_counts),
            "motifs": {"n_motifs": int(len(motifs)), "n_equal_hit_count": int(len(motifs) - len(differing_motifs)),
                       "max_reference_count_among_equal": int(motifs.loc[motifs["ours"] == motifs["reference"], "reference"].max())
                       if len(differing_motifs) < len(motifs) else None,
                       "min_reference_count_among_differing": int(differing_motifs["reference"].min()) if len(differing_motifs) else None,
                       "min_our_count_among_differing": int(differing_motifs["ours"].min()) if len(differing_motifs) else None},
        }
        if identical is not None:
            block["gene_tf_counts_identical_sequences"] = compare_counts(
                our_counts[our_counts.index.get_level_values(0).isin(identical)],
                reference_counts[reference_counts.index.get_level_values(0).isin(identical)])
        summary[f"p<{threshold:g}"] = block
    with open(args.out_json, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
