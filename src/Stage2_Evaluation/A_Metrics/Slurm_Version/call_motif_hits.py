#!/usr/bin/env python
"""Call TF-motif hits in promoter or enhancer regions -> FIMO-format tsv for motif enrichment.

Outputs in --out_dir:
  {region_type}_regions.bed       chrom, start, end, sequence_name, gene, strand (0-based half-open)
  {region_type}_motif_hits.tsv    MEME FIMO tsv columns (motif_id ... matched_sequence)

Examples
--------
Promoters (strand-aware TSS-250..TSS+50 from a GTF), MEME fimo, 16 parallel chunks:
  python call_motif_hits.py --region_type promoter --gene_annotation genes.gtf.gz \
      --genome_fasta hg38.fa --motif_file HOCOMOCOv11_full_HUMAN_mono_meme_format.meme \
      --fimo_binary "$(which fimo)" --n_chunks 64 --n_jobs 16 --out_dir out/

Enhancers from ABC predictions (drop promoter-class elements, ABC.Score >= 0.015):
  python call_motif_hits.py --region_type enhancer --enhancer_links Predictions.txt \
      --link_score_threshold 0.015 --genome_fasta hg19.fa --motif_file motifs.meme --out_dir out/

Fi-NeMo hit calls intersected with promoters (p-value column is NA):
  python call_motif_hits.py --region_type promoter --gene_annotation genes.gtf.gz \
      --hit_caller finemo --finemo_hits hits.tsv --out_dir out/
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import motif_hit_calling  # noqa: E402


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region_type", required=True, choices=["promoter", "enhancer"])
    parser.add_argument("--out_dir", required=True)

    promoter = parser.add_argument_group("promoter regions")
    promoter.add_argument("--gene_annotation", help="GTF(.gz) (canonical-transcript TSS) or BED6 gene bounds (.bed)")
    promoter.add_argument("--promoter_window_mode", default="strand_aware",
                          choices=list(motif_hit_calling.PROMOTER_WINDOW_MODES))
    promoter.add_argument("--promoter_upstream", type=int, default=250)
    promoter.add_argument("--promoter_downstream", type=int, default=50)
    promoter.add_argument("--gene_types", nargs="*", default=None,
                          help="GTF gene_type values to keep (default: all)")

    enhancer = parser.add_argument_group("enhancer regions")
    enhancer.add_argument("--enhancer_links", help="ABC / ENCODE-rE2G / scE2G tsv or IGVF bedpe")
    enhancer.add_argument("--link_format", default="auto", choices=list(motif_hit_calling.LINK_FORMATS))
    enhancer.add_argument("--link_score_threshold", type=float, default=None)
    enhancer.add_argument("--link_score_column", default=None)
    enhancer.add_argument("--keep_promoter_elements", action="store_true",
                          help="keep links whose element class is 'promoter' (dropped by default)")
    enhancer.add_argument("--merge_overlapping_links", action="store_true",
                          help="merge overlapping elements of the same target gene (links pooled over conditions)")

    hits = parser.add_argument_group("hit calling")
    hits.add_argument("--hit_caller", default="fimo", choices=["fimo", "finemo"])
    hits.add_argument("--genome_fasta")
    hits.add_argument("--motif_file", help="MEME-format motif file")
    hits.add_argument("--fimo_backend", default="auto", choices=["auto", "meme", "memelite"])
    hits.add_argument("--fimo_binary", default="fimo")
    hits.add_argument("--fimo_threshold", type=float, default=1e-4)
    hits.add_argument("--meme_default_mode", action="store_true",
                      help="run MEME fimo without --text (q-values; per-motif --max-stored-scores cap)")
    hits.add_argument("--max_stored_scores", type=int, default=None)
    hits.add_argument("--background_file", default=None, help="fimo --bgfile (default: motif file background)")
    hits.add_argument("--n_chunks", type=int, default=1)
    hits.add_argument("--n_jobs", type=int, default=1)
    hits.add_argument("--finemo_hits", help="Fi-NeMo hits.tsv (or BED6)")
    hits.add_argument("--finemo_score_column", default="hit_coefficient")
    hits.add_argument("--motif_name_map", help="tsv: finemo motif_name <tab> motif_id (no header)")
    return parser.parse_args(argv)


def build_regions(args) -> pd.DataFrame:
    if args.region_type == "promoter":
        if not args.gene_annotation:
            raise SystemExit("--gene_annotation is required for --region_type promoter")
        is_bed = ".bed" in os.path.basename(args.gene_annotation)
        gene_tss = (motif_hit_calling.read_bed_gene_tss(args.gene_annotation, args.promoter_window_mode) if is_bed
                    else motif_hit_calling.read_gtf_gene_tss(args.gene_annotation, args.gene_types))
        return motif_hit_calling.build_promoter_regions(gene_tss, args.promoter_upstream,
                                                        args.promoter_downstream, args.promoter_window_mode)
    if not args.enhancer_links:
        raise SystemExit("--enhancer_links is required for --region_type enhancer")
    links = motif_hit_calling.read_enhancer_gene_links(
        args.enhancer_links, args.link_format, args.link_score_threshold, args.link_score_column,
        drop_promoters=not args.keep_promoter_elements, merge_overlapping=args.merge_overlapping_links)
    return motif_hit_calling.build_enhancer_regions(links)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_arguments(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    regions = build_regions(args)
    regions_bed = os.path.join(args.out_dir, f"{args.region_type}_regions.bed")
    regions[["chrom", "start", "end", "sequence_name", "gene", "strand"]].to_csv(
        regions_bed, sep="\t", header=False, index=False)
    logging.info("%d %s regions (%d genes) -> %s", len(regions), args.region_type,
                 regions["gene"].nunique(), regions_bed)

    out_tsv = os.path.join(args.out_dir, f"{args.region_type}_motif_hits.tsv")
    if args.hit_caller == "fimo":
        if not (args.genome_fasta and args.motif_file):
            raise SystemExit("--genome_fasta and --motif_file are required for --hit_caller fimo")
        n_hits = motif_hit_calling.scan_regions_with_fimo(
            regions, args.genome_fasta, args.motif_file, out_tsv, backend=args.fimo_backend,
            threshold=args.fimo_threshold, fimo_binary=args.fimo_binary,
            meme_text_mode=not args.meme_default_mode, max_stored_scores=args.max_stored_scores,
            background_file=args.background_file, n_chunks=args.n_chunks, n_jobs=args.n_jobs,
            work_dir=os.path.join(args.out_dir, f"{args.region_type}_fimo_chunks"))
    else:
        if not args.finemo_hits:
            raise SystemExit("--finemo_hits is required for --hit_caller finemo")
        motif_name_map = None
        if args.motif_name_map:
            mapping = pd.read_csv(args.motif_name_map, sep="\t", header=None, dtype=str)
            motif_name_map = dict(zip(mapping[0], mapping[1]))
        hits = motif_hit_calling.call_hits_from_finemo(
            motif_hit_calling.read_finemo_hits(args.finemo_hits, args.finemo_score_column), regions, motif_name_map)
        hits.to_csv(out_tsv, sep="\t", index=False, na_rep="NA")
        n_hits = len(hits)
    logging.info("%d hits -> %s", n_hits, out_tsv)


if __name__ == "__main__":
    main()
