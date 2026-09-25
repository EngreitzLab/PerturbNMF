"""Turn the long regulator table into one effect profile per perturbed gene, with its noise level.

Input is the regulator table ProgramAnnotatorV3 already takes (`program_id, [condition,]
target_gene, log2_fc, p_value, adj_pval, significant`; one row per program x condition x target),
so both annotators read the same data directory. A multi-condition screen gives one feature per
program x condition, in the order the conditions are listed.

Outputs (in --output-dir):
  effect_matrix.tsv     regulator x feature log2FC (features `P<program>|<condition>`)
  significance.tsv      regulator x feature adjusted p
  regulator_summary.tsv per regulator: n significant features, signal (RMS log2FC), standard
                        error, reliability, effect-strength tier

Noise model. Correlation between two effect profiles is attenuated by noise: a regulator with
weak effects has a noisy profile, so it correlates weakly even with its true partners. To correct
for that downstream (define_regulator_groups.py), each regulator gets
  standard error  SE = median over features of |log2FC| / |z|, z = Phi^-1(1 - p/2), using the
                  features with |z| >= 0.5 (where the ratio is stable)
  reliability     rel = 1 - SE^2 / var(profile), clipped to [0, 1] — the share of the profile's
                  variance that is signal rather than noise (classical test-theory reliability)

Programs whose usage tracks batch (e.g. PerturbNMF's categorical association, or the share of
usage variance explained by sample) can be left out with --exclude-programs: every knockdown
"moves" them with the batch it was sequenced in, which makes unrelated regulators correlate.

Usage:
    python build_regulator_effect_matrix.py --regulators regulators_by_condition.csv \
        --conditions D0 D1 D2 D3 --output-dir regulator_groups
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

MIN_Z_FOR_SE = 0.5
P_VALUE_COLUMNS = ("p_value", "pval", "p-value")


def feature_name(program_id, condition) -> str:
    return f"P{program_id}|{condition}"


def load_regulators(path: Path, conditions: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_csv(path)
    p_column = next((c for c in P_VALUE_COLUMNS if c in table.columns), None)
    if p_column is None:
        raise SystemExit(f"{path}: needs a raw p-value column ({', '.join(P_VALUE_COLUMNS)}) for the noise model")
    table = table.rename(columns={p_column: "p_value"})
    if "condition" not in table.columns:
        table["condition"] = "all"
    table["condition"] = table["condition"].astype(str)
    conditions = conditions or sorted(table["condition"].unique())
    missing = set(table["condition"].unique()) - set(conditions)
    if missing:
        raise SystemExit(f"conditions in the table but not in --conditions: {sorted(missing)}")
    table["significant"] = table["significant"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    table["p_value"] = table["p_value"].clip(lower=1e-300, upper=1.0)
    return table, conditions


def build_matrices(table: pd.DataFrame, conditions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    programs = sorted(table["program_id"].unique())
    features = [feature_name(p, c) for c in conditions for p in programs]
    table = table.assign(feature=[feature_name(p, c) for p, c in zip(table["program_id"], table["condition"])])
    effects = table.pivot_table(index="target_gene", columns="feature", values="log2_fc", aggfunc="first")
    adjusted = table.pivot_table(index="target_gene", columns="feature", values="adj_pval", aggfunc="first")
    raw_p = table.pivot_table(index="target_gene", columns="feature", values="p_value", aggfunc="first")
    effects = effects.reindex(columns=features)
    adjusted = adjusted.reindex(columns=features)
    raw_p = raw_p.reindex(index=effects.index, columns=features)
    return effects, adjusted, raw_p


def noise_model(effects: pd.DataFrame, raw_p: pd.DataFrame) -> pd.DataFrame:
    z = norm.isf(raw_p.to_numpy() / 2.0)
    x = effects.to_numpy()
    ratio = np.where(np.abs(z) >= MIN_Z_FOR_SE, np.abs(x) / np.maximum(np.abs(z), 1e-12), np.nan)
    standard_error = np.nanmedian(ratio, axis=1)
    variance = np.nanvar(x, axis=1)
    reliability = np.clip(1.0 - standard_error ** 2 / np.maximum(variance, 1e-12), 0.0, 1.0)
    reliability = np.where(np.isnan(standard_error), 0.0, reliability)
    return pd.DataFrame({
        "standard_error": standard_error,
        "signal_rms": np.sqrt(np.nanmean(x ** 2, axis=1)),
        "reliability": reliability,
    }, index=effects.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regulators", required=True, type=Path, help="long regulator table (the V3 input)")
    parser.add_argument("--conditions", nargs="*", help="condition order (default: sorted)")
    parser.add_argument("--significance", type=float, default=0.05, help="adjusted-p cutoff for a significant feature")
    parser.add_argument("--min-significant-features", type=int, default=1,
                        help="keep regulators significant in at least this many program x condition features")
    parser.add_argument("--exclude-pattern", default=r"(?i)^(?:non[-_]?targeting|NTC|safe[-_]?targeting)",
                        help="regex for control targets to drop")
    parser.add_argument("--exclude-programs", default="",
                        help="comma list of program ids left out of the profiles, e.g. batch-associated programs "
                             "(sample explains much of their usage) — they add shared, non-biological structure")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    table, conditions = load_regulators(args.regulators, args.conditions)
    table = table[~table["target_gene"].astype(str).str.contains(args.exclude_pattern, regex=True)]
    excluded_programs = {int(p) for p in args.exclude_programs.split(",") if p.strip()}
    table = table[~table["program_id"].astype(int).isin(excluded_programs)]
    effects, adjusted, raw_p = build_matrices(table, conditions)
    summary = noise_model(effects, raw_p)
    summary["n_significant"] = (adjusted < args.significance).sum(axis=1)
    summary = summary[summary["n_significant"] >= args.min_significant_features]
    summary["strength_tier"] = pd.qcut(summary["signal_rms"].rank(method="first"), 3, labels=["weak", "medium", "strong"])
    effects = effects.loc[summary.index].fillna(0.0)
    adjusted = adjusted.loc[summary.index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output_dir / "effect_matrix.tsv", sep="\t")
    adjusted.to_csv(args.output_dir / "significance.tsv", sep="\t")
    summary.sort_values("signal_rms", ascending=False).to_csv(args.output_dir / "regulator_summary.tsv", sep="\t")
    print(f"{len(summary)} regulators (significant in >= {args.min_significant_features} of "
          f"{effects.shape[1]} features); reliability median {summary['reliability'].median():.2f}, "
          f"{(summary['reliability'] < 0.2).sum()} below 0.2 -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
