#!/usr/bin/env python
"""Merge per-condition motif-enrichment runs into one Stage 2 table set, taking each program's
condition-specific rows from the program's peak condition.

Why: some motif sources are condition-specific (e.g. Fi-NeMo hits from a ChromBPNet model trained per
condition), while the programs span all conditions. run_motif_enrichment.py is then run once per
condition (same programs, that condition's hits); this tool keeps, for each program, the rows of the
condition-specific sources (``--per_condition_sources``, default ``finemo``) from the program's peak
condition. Rows of every other source (e.g. ``fimo``: same sequences in every condition) come from the
first run. FDRs are kept as computed in each run (BH over that run's programs x motifs per element type).

Peak condition of a program:
  --peak_conditions   CSV/TSV with columns ``program, condition`` (condition must name one of the runs)
  --program_activity  CSV/TSV with ``program, condition, mean_score``; used for programs without a
                      usable peak in --peak_conditions (or for all programs when it is not given):
                      peak = the run condition with the highest mean score (``--peak_rule max_mean_score``)

Inputs: --condition_runs A=<Evaluation/{K}_{thresh} dir of run A> B=<...> (the first run supplies the
condition-independent rows, the Fi-NeMo pattern names and the base config). Each dir holds the files
run_motif_enrichment.py writes: {K}_motif_enrichment.txt, {K}_candidate_tfs.txt and optionally
{K}_motif_logos.json, {K}_finemo_pattern_names.tsv, {K}_motif_enrichment_config.yml.

Writes into --out_evaluation_dir (the Stage 3 inputs):
  {K}_motif_enrichment.txt, {K}_candidate_tfs.txt   merged rows (+ column peak_condition, "" for rows
                                                    from the first run's condition-independent sources)
  {K}_motif_enrichment_by_condition.txt             every run's condition-specific rows (column condition)
  {K}_motif_logos.json                              union of the runs' logo files (if any)
  {K}_finemo_pattern_names.tsv                      copied from the first run (if present)
  {K}_motif_enrichment_config.yml                   merge sources, rule and peak per program (JSON)

Example:
  python merge_motif_tables_at_peak_condition.py --K 10 \\
      --condition_runs ctrl=<run_ctrl>/Evaluation/10_0_2 stim=<run_stim>/Evaluation/10_0_2 \\
      --program_activity program_activity.csv --out_evaluation_dir <merged>/Evaluation/10_0_2
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

PEAK_RULES = ("max_mean_score",)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, required=True, help="K of the runs (file prefix {K}_)")
    parser.add_argument("--condition_runs", nargs="+", required=True,
                        help="<condition>=<evaluation dir of that condition's run>; the first supplies the "
                             "condition-independent rows")
    parser.add_argument("--peak_conditions", default=None,
                        help="CSV/TSV with columns program, condition (peak condition per program)")
    parser.add_argument("--program_activity", default=None,
                        help="CSV/TSV with columns program, condition, mean_score (peak = highest mean score)")
    parser.add_argument("--peak_rule", default="max_mean_score", choices=list(PEAK_RULES),
                        help="rule applied to --program_activity for programs without a --peak_conditions entry")
    parser.add_argument("--per_condition_sources", nargs="+", default=["finemo"],
                        help="motif_source values that differ between conditions (default: finemo)")
    parser.add_argument("--out_evaluation_dir", required=True)
    return parser.parse_args(argv)


def parse_condition_runs(items: List[str]) -> Dict[str, str]:
    runs = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--condition_runs entries must be <condition>=<dir>, got {item!r}")
        condition, path = item.split("=", 1)
        if condition in runs:
            raise SystemExit(f"condition {condition!r} given twice in --condition_runs")
        runs[condition] = path
    return runs


def read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def read_peak_conditions(conditions: List[str], peak_conditions_path: Optional[str] = None,
                         program_activity_path: Optional[str] = None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Peak condition per program and how it was chosen ("table" | "max_mean_score")."""
    if not peak_conditions_path and not program_activity_path:
        raise SystemExit("give --peak_conditions and/or --program_activity")
    peaks, rules = {}, {}
    if peak_conditions_path:
        table = read_table(peak_conditions_path)
        missing = {"program", "condition"} - set(table.columns)
        if missing:
            raise SystemExit(f"{peak_conditions_path}: missing columns {sorted(missing)}")
        for program, condition in zip(table["program"].astype(int), table["condition"].astype(str)):
            if condition in conditions:
                peaks[program], rules[program] = condition, "table"
    if program_activity_path:
        activity = read_table(program_activity_path)
        missing = {"program", "condition", "mean_score"} - set(activity.columns)
        if missing:
            raise SystemExit(f"{program_activity_path}: missing columns {sorted(missing)}")
        activity = activity.assign(program=activity["program"].astype(int), condition=activity["condition"].astype(str))
        top = (activity[activity["condition"].isin(conditions)]
               .sort_values("mean_score", ascending=False, kind="stable").drop_duplicates("program"))
        for program, condition in zip(top["program"], top["condition"]):
            if program not in peaks:
                peaks[program], rules[program] = condition, "max_mean_score"
    return peaks, rules


def merge_table(tables: Dict[str, pd.DataFrame], peaks: Dict[int, str], per_condition_sources: List[str],
                file_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(merged rows, all condition-specific rows with a `condition` column)."""
    first = next(iter(tables))
    if "motif_source" not in tables[first].columns:
        raise SystemExit(f"{file_name}: no motif_source column; run run_motif_enrichment.py with --motif_source "
                         "both (or finemo) so the condition-specific rows can be told apart")
    independent = tables[first][~tables[first]["motif_source"].isin(per_condition_sources)].assign(peak_condition="")
    by_condition = pd.concat([table[table["motif_source"].isin(per_condition_sources)].assign(condition=condition)
                              for condition, table in tables.items()], ignore_index=True)
    peak = by_condition["program"].astype(int).map(peaks)
    if peak.isna().any():
        raise SystemExit(f"{file_name}: programs without a peak condition: "
                         f"{sorted(by_condition.loc[peak.isna(), 'program'].astype(int).unique())}")
    at_peak = by_condition[by_condition["condition"] == peak].rename(columns={"condition": "peak_condition"})
    return pd.concat([independent, at_peak], ignore_index=True), by_condition


def merge_logos(paths: List[str]) -> Optional[dict]:
    merged = None
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            logos = json.load(handle)
        if merged is None:
            merged = logos
            continue
        for source, entries in logos.get("logos", {}).items():
            merged.setdefault("logos", {}).setdefault(source, {}).update(entries)
    return merged


def read_run_config(path: str):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv=None):
    args = parse_arguments(argv)
    runs = parse_condition_runs(args.condition_runs)
    conditions = list(runs)
    peaks, rules = read_peak_conditions(conditions, args.peak_conditions, args.program_activity)
    os.makedirs(args.out_evaluation_dir, exist_ok=True)
    k = args.K

    def out(name: str) -> str:
        return os.path.join(args.out_evaluation_dir, name)

    merged = {}
    for kind in ("motif_enrichment", "candidate_tfs"):
        file_name = f"{k}_{kind}.txt"
        tables = {condition: pd.read_csv(os.path.join(path, file_name), sep="\t") for condition, path in runs.items()}
        merged[kind], by_condition = merge_table(tables, peaks, args.per_condition_sources, file_name)
        merged[kind].to_csv(out(file_name), sep="\t", index=False)
        if kind == "motif_enrichment":
            by_condition.to_csv(out(f"{k}_motif_enrichment_by_condition.txt"), sep="\t", index=False)

    logos = merge_logos([os.path.join(path, f"{k}_motif_logos.json") for path in runs.values()])
    if logos is not None:
        with open(out(f"{k}_motif_logos.json"), "w") as handle:
            json.dump(logos, handle)
    first_dir = next(iter(runs.values()))
    pattern_names = os.path.join(first_dir, f"{k}_finemo_pattern_names.tsv")
    if os.path.exists(pattern_names):
        pd.read_csv(pattern_names, sep="\t").to_csv(out(f"{k}_finemo_pattern_names.tsv"), sep="\t", index=False)

    results = merged["motif_enrichment"]
    config = {
        "merge": {"condition_runs": runs, "peak_conditions": args.peak_conditions,
                  "program_activity": args.program_activity, "peak_rule": args.peak_rule,
                  "per_condition_sources": list(args.per_condition_sources),
                  "rule": "condition-independent rows from the first run; condition-specific rows of each program "
                          "from its peak condition; per-run FDRs kept",
                  "peak_condition": {int(p): c for p, c in sorted(peaks.items())},
                  "peak_chosen_by": {int(p): r for p, r in sorted(rules.items())}},
        "first_run_config": read_run_config(os.path.join(first_dir, f"{k}_motif_enrichment_config.yml")),
    }
    if "significant" in results.columns and "element_type" in results.columns:
        config["n_significant"] = {f"{element}_{source}": int(n) for (element, source), n in
                                   results.groupby(["element_type", "motif_source"])["significant"].sum().items()}
    with open(out(f"{k}_motif_enrichment_config.yml"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(json.dumps(config.get("n_significant", {}), indent=2))
    print("MERGE_OK")


if __name__ == "__main__":
    main()
