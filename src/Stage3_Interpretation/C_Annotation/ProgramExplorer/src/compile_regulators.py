#!/usr/bin/env python3
"""Compile SCEPTRE regulator results across time points into a single TSV.

Given a directory of per-timepoint regulator files, merges them into one
combined file with a 'day' column and standardised column names.
"""
#%%
import argparse
import sys
from pathlib import Path

import pandas as pd


def compile_regulator_days(
    data_dir: Path,
    days: list[str],
    file_pattern: str = "50_CRT_{day}.txt",
    output: Path | None = None,
    sep: str = "\t",
    rename_columns: dict[str, str] | None = None,
    significance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Compile per-timepoint regulator files into a single DataFrame.

    Parameters
    ----------
    data_dir : Path
        Directory containing the per-day files.
    days : list[str]
        List of time point labels (e.g. ["D0", "D1", "D2", "D3"]).
    file_pattern : str
        Filename pattern with ``{day}`` placeholder.
    output : Path or None
        If provided, write the combined table here.
    sep : str
        Separator for input/output files.
    rename_columns : dict or None
        Column renames to apply (e.g. {"target_name": "target_gene_name"}).
    significance_threshold : float
        p-value threshold to add a ``significant`` boolean column.

    Returns
    -------
    pd.DataFrame
    """
    frames = []
    for day in days:
        f = data_dir / file_pattern.format(day=day)
        df = pd.read_csv(f, sep=sep)
        df["day"] = day
        frames.append(df)
        print(f"Loaded {day}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)

    if rename_columns:
        combined = combined.rename(columns=rename_columns)

    if "adj_pval" in combined.columns:
        combined["significant"] = combined["adj_pval"] < significance_threshold

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output, sep=sep, index=False)
        print(f"Wrote {len(combined)} rows to {output}")

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile per-timepoint regulator files into one TSV"
    )
    parser.add_argument("data_dir", type=Path, help="Directory with per-day files")
    parser.add_argument(
        "--days", nargs="+", default=["D0", "D1", "D2", "D3"],
        help="Time point labels (default: D0 D1 D2 D3)",
    )
    parser.add_argument(
        "--pattern", default="50_CRT_{day}.txt",
        help="Filename pattern with {day} placeholder",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--rename", nargs="*", default=["target_name=target_gene_name", "approx_log2FC=log2fc"],
        help="Column renames as key=value pairs",
    )
    args = parser.parse_args()

    rename_columns = {}
    if args.rename:
        for item in args.rename:
            k, v = item.split("=", 1)
            rename_columns[k] = v

    output = args.output
    if output is None:
        output = args.data_dir / "50_CRT_all_days.txt"

    compile_regulator_days(
        data_dir=args.data_dir,
        days=args.days,
        file_pattern=args.pattern,
        output=output,
        rename_columns=rename_columns,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
