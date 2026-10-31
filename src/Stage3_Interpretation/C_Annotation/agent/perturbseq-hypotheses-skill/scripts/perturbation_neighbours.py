#!/usr/bin/env python3
"""Perturbation neighbours: which other knockdowns produce the same program-effect vector?

REFERENCE IMPLEMENTATION. Adapt the I/O, commit the adapted copy into the analysis dir.

Why this exists
---------------
This is the cheapest and most informative axis in the whole report, and it is the one that
was discovered last. For a gene with no literature in your tissue, it is the single most
useful result available: it places the gene inside a pathway the reader already knows.

From a real report, for a gene with zero prior liver publications:

    Neighbours. Insr +0.70, Irs1 +0.69, Pik3ca +0.63, Irs2 +0.62, Mtor +0.57 -- knockdown
    mirrors loss of insulin signalling. For a gene with no liver literature this is the
    single most informative result available: the screen places it in the insulin/PI3K axis.

Negative neighbours matter as much as positive ones: a candidate that anti-correlates
cleanly with a well-understood partner is the `mirror` evidence class, and it comes with a
built-in control.

CHECK FOR A SHIPPED MATRIX FIRST
--------------------------------
Some PerturbNMF runs already write the target x target correlation matrix in Stage 3 --
`corr_matrix_{cond}.txt` or `corr_gene_matrix_{cond}.txt` under the Eval/Evaluation dir. If
one exists, USE IT and do not recompute: Stage 0's rule is "if the statistic exists in the
run, use it", and a recomputed matrix that disagrees with the shipped one is a discrepancy
you then have to explain. Run this script only when the run ships no such file.

One caveat when you do use the shipped matrix: it is usually computed over ALL programs,
including technical ones (CNV, cell cycle, ambient). That is the single thing
--exclude-programs exists to fix here. Where it matters, report the shipped matrix and a
technical-program-excluded version side by side, and state the difference.

Method
------
Correlate each target's K-dimensional program-effect vector against every other target's.
Two decisions that matter, both exposed as flags:

  * `--exclude-programs` -- drop technical programs (CNV/aneuploidy, cell cycle, ambient)
    before correlating. Leaving them in makes every high-effect gene look like every other
    high-effect gene.
  * `--min-effect` -- a target whose effect vector is all noise correlates with noise.
    Targets below this are reported but marked `low_signal`.

Correlation is Pearson on the signed effect vector, each vector centred on its own mean.
State the estimator that precisely in the report: a vague definition is how two scripts
compute "the same" number differently.

Input
-----
Either a long table (target, program, effect) or a wide target x program matrix.

Usage
-----
    python perturbation_neighbours.py \
        --effects outputs/gene_program_effects.csv \
        --target-col gene --program-col program --effect-col log2FC \
        --exclude-programs 35 48 50 \
        --k 6 \
        --out outputs/perturbation_neighbours.csv \
        --matrix-out outputs/perturbation_effect_correlation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def load_matrix(
    path: Path, target_col: str, program_col: str, effect_col: str
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if {target_col, program_col, effect_col} <= set(df.columns):
        mat = df.pivot_table(
            index=target_col, columns=program_col, values=effect_col, aggfunc="mean"
        )
    elif target_col in df.columns:
        mat = df.set_index(target_col)
        mat = mat.select_dtypes(include=[np.number])
    else:
        die(
            f"{path.name}: expected a long table with "
            f"({target_col}, {program_col}, {effect_col}) or a wide matrix indexed by "
            f"{target_col!r}. Present: {sorted(df.columns)[:15]}"
        )
    mat.columns = [str(c) for c in mat.columns]
    return mat


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--effects", type=Path, required=True)
    ap.add_argument("--target-col", default="gene")
    ap.add_argument("--program-col", default="program")
    ap.add_argument("--effect-col", default="log2FC")
    ap.add_argument(
        "--exclude-programs",
        nargs="*",
        default=[],
        help="technical programs to drop before correlating (CNV, cell cycle, ambient)",
    )
    ap.add_argument("--exclude-targets", nargs="*", default=["non-targeting"])
    ap.add_argument(
        "--min-effect",
        type=float,
        default=0.0,
        help="max |effect| below which a target is marked low_signal",
    )
    ap.add_argument(
        "--k", type=int, default=6, help="neighbours to report per direction"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--matrix-out",
        type=Path,
        default=None,
        help="also write the full target x target correlation matrix",
    )
    args = ap.parse_args()

    mat = load_matrix(args.effects, args.target_col, args.program_col, args.effect_col)

    drop = [str(p) for p in args.exclude_programs]
    unknown = [p for p in drop if p not in mat.columns]
    if unknown:
        die(
            f"--exclude-programs not present in the matrix: {unknown}. "
            f"Columns look like: {list(mat.columns)[:8]}"
        )
    if drop:
        mat = mat.drop(columns=drop)
        print(f"  dropped {len(drop)} technical programs: {drop}")

    mat = mat.drop(
        index=[t for t in args.exclude_targets if t in mat.index], errors="ignore"
    )

    if mat.isna().any().any():
        n = int(mat.isna().sum().sum())
        print(
            f"  WARNING: {n} missing values -> 0. Prefer an effect table with no gaps."
        )
        mat = mat.fillna(0.0)

    print(f"  {mat.shape[0]} targets x {mat.shape[1]} programs")

    max_abs = mat.abs().max(axis=1)
    low_signal = (
        set(max_abs[max_abs < args.min_effect].index) if args.min_effect > 0 else set()
    )
    if low_signal:
        print(
            f"  {len(low_signal)} targets marked low_signal (max |effect| < {args.min_effect})"
        )

    # Pearson across programs, each target's vector centred on its own mean.
    # Blank the diagonal on the ndarray before wrapping it: a DataFrame built straight from
    # np.corrcoef can expose a read-only buffer, and fill_diagonal then raises.
    cm = np.corrcoef(mat.values)
    np.fill_diagonal(cm, np.nan)
    corr = pd.DataFrame(cm, index=mat.index, columns=mat.index)

    rows = []
    for target in corr.index:
        s = corr.loc[target].dropna().sort_values(ascending=False)
        pos = s.head(args.k)
        neg = s.tail(args.k).sort_values()
        rows.append(
            {
                "target": target,
                "max_abs_effect": round(float(max_abs.loc[target]), 4),
                "low_signal": target in low_signal,
                "top_positive_neighbours": "; ".join(
                    f"{g} {v:+.2f}" for g, v in pos.items()
                ),
                "top_negative_neighbours": "; ".join(
                    f"{g} {v:+.2f}" for g, v in neg.items()
                ),
                "best_positive_r": round(float(pos.iloc[0]), 4) if len(pos) else None,
                "best_negative_r": round(float(neg.iloc[0]), 4) if len(neg) else None,
            }
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")

    if args.matrix_out:
        corr.to_csv(args.matrix_out)
        print(f"-> {args.matrix_out}")

    print(
        "\nIn the report: quote the neighbours as a plain sentence, not a coefficient dump."
    )
    print(
        '  "Knockdown mirrors loss of insulin signalling (Insr +0.70, Irs1 +0.69, Pik3ca +0.63)."'
    )
    print(
        "Strong negative neighbours with a well-understood partner -> the `mirror` class."
    )


if __name__ == "__main__":
    main()
