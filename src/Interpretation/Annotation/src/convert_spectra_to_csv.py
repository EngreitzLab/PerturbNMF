#!/usr/bin/env python3
"""Convert a cNMF gene_spectra_score TSV (programs × genes matrix) to
the long-format CSV expected by ProgExplorer.

Input:  TSV with program indices as rows, gene names as columns.
Output: CSV with columns Name, Score, RowID where RowID is the program
        index, Name is the gene, and Score is the matrix entry.

By default, only the top-N genes per program (by score) are kept.
Use --n-top 0 to keep all genes.
"""
#%%
import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert gene_spectra_score TSV to long-format CSV"
    )
    parser.add_argument(
        "input_txt",
        help="Path to gene_spectra_score .txt (TSV) file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (default: <input_stem>_top<N>.csv in same dir)",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=300,
        help="Keep top N genes per program by score (default: 300, 0 = all)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_txt)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read the matrix (programs × genes)
    df = pd.read_csv(input_path, sep="\t", index_col=0)
    print(f"Loaded matrix: {df.shape[0]} programs × {df.shape[1]} genes")

    # Melt to long format: RowID (program), Name (gene), Score (value)
    df.index.name = "RowID"
    long = df.reset_index().melt(
        id_vars="RowID", var_name="Name", value_name="Score"
    )

    # Shift RowID to 0-based indexing
    long["RowID"] = long["RowID"] - 1

    # Keep only top-N genes per program (by descending score)
    if args.n_top > 0:
        long = (
            long.sort_values(["RowID", "Score"], ascending=[True, False])
            .groupby("RowID", sort=False)
            .head(args.n_top)
        )
        print(f"Kept top {args.n_top} genes per program")

    # Sort: RowID ascending, Score descending within each program
    long = long.sort_values(
        ["RowID", "Score"], ascending=[True, False]
    ).reset_index(drop=True)

    # Reorder columns to match expected format: Name, Score, RowID
    long = long[["Name", "Score", "RowID"]]

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        n_label = f"top{args.n_top}" if args.n_top > 0 else "all"
        out_path = input_path.with_name(
            f"{input_path.stem}_loading_gene_k{df.shape[0]}_{n_label}.csv"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_path, index=False)
    print(f"Wrote {len(long)} rows to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
