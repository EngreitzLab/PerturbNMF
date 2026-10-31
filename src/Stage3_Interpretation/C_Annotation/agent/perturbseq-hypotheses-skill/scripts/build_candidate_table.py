#!/usr/bin/env python3
"""Build the admission table for a hypothesis report from a PerturbNMF run.

REFERENCE IMPLEMENTATION. Adapt the I/O to your run, commit the adapted copy into the
analysis directory, and run it in the analysis repo's own environment.

What this encodes, and why
--------------------------
One gate, taken from the run's OWN calibrated test: significant at a threshold whose
empirical false-positive rate is read off the run's control arm, in at least
`--min-conditions` of the conditions present. The output carries the covariates a reader
needs but which must NOT filter: number of significant calls, knockdown efficiency,
essentiality.

Deliberately absent:

  * No composite score. Interest is a declared evidence class, not a number
    (see ../references/evidence-classes.md).
  * No ranking on counts of significant calls -- that is power x group size.
  * No knockdown-efficiency gate -- it correlates ~ +0.08 with whether a downstream
    signature reproduces; gating discards real biology.
  * No home-made null. If the run ships a calibrated test, this script uses it.

Runs differ more than you expect
--------------------------------
Everything below is auto-detected and reported, because two real PerturbNMF runs disagreed
on every one of them:

  * **Significance columns.** Some runs ship a `*_raw` pair alongside a second,
    q-value-style pair that was never validated against a control arm. Where both exist the
    `_raw` pair is used. Where only one exists it is used and said so. Override with
    --pval-column / --adj-pval-column.
  * **Effect column.** `log2FC`, `approx_log2FC`, or `beta_hat` depending on the run and on
    whether a post-hoc conversion was applied. Override with --log2fc-column.
  * **One condition or several.** A single-condition run cannot use cross-condition
    replication; --min-conditions clamps to what exists and the report must then say which
    substitute is carrying that weight (see ../references/failure-modes.md).
  * **Ordered vs exchangeable conditions.** Pass --ordered when --conditions is a time course
    given in order. A plain ">= N conditions, same sign" rule silently discards two things
    that are not failures in a time course, and it cost a real run its headline results:
      - an effect present ONLY at the terminal timepoint (endpoint biology -- one target had
        13 significant programs at the last day and one the day before, and cleared no gate);
      - an effect that CHANGES SIGN partway through (a temporal switch -- negative early,
        positive later -- which is usually the most interesting thing in the run).
    So pairs are CLASSIFIED, not filtered: `persistent` (contiguous conditions),
    `persistent_gapped` (non-adjacent -- admitted but carries NO replication control),
    `sign_flip` (admitted, labelled), `endpoint_only` (admitted on breadth,
    --endpoint-min-programs), `transient` (held out). Never enforce same-sign as a filter
    across an ordered series.
  * **The control arm.** Usually a target inside the main table (`safe-targeting`,
    `non-targeting`), not a separate file. Pass --ntc-in-table. A separate file still works
    via --ntc-glob.
  * **Target naming.** The alternative-TSS convention is detected, not assumed -- one library
    used `-P<n>`, another `-TSS`. Whichever matches is collapsed onto the base gene and the
    detected convention is printed. Dual-gene `X-and-Y` targets are a different hazard and are
    NOT collapsed -- they are reported so you decide.

Two things it will refuse to do quietly
---------------------------------------
  * If the control arm is one pooled pseudo-target, the "empirical FPR" is estimated from a
    handful of tests. It is reported as a DIAGNOSTIC, not a calibrated null.
  * If the control arm itself shows structured significant signal, it is not a null at all --
    that happens when the library was selected for an effect the controls also feel. Detected
    two ways, because either alone misses cases: sign coherence, AND depth (very small q).
    A real arm ran 6 up / 2 down -- incoherent by sign, but down to q = 4e-12.

Usage
-----
    # multi-condition, control inside the table (CRT-style run)
    python build_candidate_table.py \
        --crt-dir  .../Eval/50_0_2 --crt-glob '50_CRT_{cond}.txt' \
        --conditions D0 D1 D2 D3 --ntc-in-table non-targeting \
        --n-programs 50 --exclude-programs 12 34 \
        --out outputs/candidate_table.csv

    # single condition, one file (U-test-style run)
    python build_candidate_table.py \
        --effects .../60_perturbation_association_results_EC.txt --condition-name EC \
        --ntc-in-table safe-targeting --n-programs 60 \
        --out outputs/candidate_table.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Libraries label alternative TSSs differently: `-P1`, `-TSS`, `-TSS2`, `-alt-TSS`. Detect
# which convention (if any) this library uses rather than assuming one and dying on the rest.
PROMOTER_SUFFIX_PATTERNS = [
    ("-P<n>", re.compile(r"-P\d+$")),
    ("-TSS<n>", re.compile(r"-(?:alt-)?TSS\d*$", re.IGNORECASE)),
    ("-prom<n>", re.compile(r"-prom(?:oter)?\d*$", re.IGNORECASE)),
]
# Anything else trailing a hyphen that looks like a systematic suffix, used only to decide
# whether "no convention found" is believable or is an unhandled convention.
TRAILING_TOKEN = re.compile(r"-[A-Za-z]{1,6}\d*$")
DUAL_GENE = re.compile(r"-and-", re.IGNORECASE)

PVAL_CANDIDATES = [
    "p-value_raw",
    "pval_raw",
    "p_value_raw",
    "p-value",
    "pval",
    "p_value",
]
ADJ_CANDIDATES = [
    "adj_pval_raw",
    "adj_p_value_raw",
    "adj_pval",
    "adj_p_value",
    "padj",
    "FDR",
]
EFFECT_CANDIDATES = ["log2FC", "approx_log2FC", "beta_hat", "logFC"]

# Two independent tests for "the control arm is showing structure, not noise". Both are
# heuristics and both are named here rather than buried as literals, because they decide what
# a REPORTED caveat says.
#   SIGN_COHERENCE_MIN: mean sign of the significant control effects. 0.6 means roughly 4:1
#     one-directional. Chosen as a value a genuinely bidirectional arm will not reach while a
#     one-directional one comfortably will; it is a judgement, not a calibrated cutoff, and a
#     real arm at 6 up / 2 down (mean 0.5) sits just under it -- which is why the depth test
#     below exists as well.
#   DEPTH_FACTOR: smallest control q-value below threshold/DEPTH_FACTOR counts as deep. 100 is
#     two orders of magnitude past the calling threshold, far outside what a null produces.
SIGN_COHERENCE_MIN = 0.6
DEPTH_FACTOR = 100


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def pick_column(
    df: pd.DataFrame, candidates: list[str], override: str | None, label: str
) -> tuple[str, str]:
    """Return (column, why). Prefers a *_raw variant where the run ships both."""
    if override:
        if override not in df.columns:
            die(
                f"--{label}-column {override!r} not in the table. Present: {sorted(df.columns)}"
            )
        return override, "set explicitly on the command line"
    present = [c for c in candidates if c in df.columns]
    if not present:
        die(
            f"no {label} column found. Tried {candidates}. Present: {sorted(df.columns)}"
        )
    chosen = present[0]
    if len(present) > 1 and chosen.endswith("_raw"):
        why = (
            f"the run ships both {present}; the _raw pair is used because the other is "
            f"typically a q-value-style transform never validated against the control arm"
        )
    elif len(present) > 1:
        why = f"the run ships {present}; the first was taken -- CHECK this is the one you want"
    else:
        why = f"the only {label} column in this run"
    return chosen, why


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_table(path: Path, pval_col, adj_col, effect_col, target_col, program_col):
    df = pd.read_csv(path, sep=None, engine="python")
    p, p_why = pick_column(df, PVAL_CANDIDATES, pval_col, "pval")
    a, a_why = pick_column(df, ADJ_CANDIDATES, adj_col, "adj-pval")
    e, e_why = pick_column(df, EFFECT_CANDIDATES, effect_col, "log2fc")
    for c in (target_col, program_col):
        if c not in df.columns:
            die(f"{path.name}: no column {c!r}. Present: {sorted(df.columns)}")
    out = df[[target_col, program_col, e, p, a]].copy()
    out.columns = ["target_name", "program", "effect", "p_value", "adj_pval"]
    return out, {"pval": (p, p_why), "adj_pval": (a, a_why), "effect": (e, e_why)}


def collapse_target_names(df: pd.DataFrame, allow_no_suffix: bool) -> pd.DataFrame:
    """`Nr1h4-P2` -> `Nr1h4`, keeping the most significant row per (gene, program).

    The alternative-TSS convention is DETECTED, not assumed. One library used `-P<n>`,
    another used `-TSS`, and hardcoding the first made the script exit on the second unless
    you knew to pass --allow-no-promoter-suffix. Now: try each known pattern, use whichever
    matches, and only refuse to continue when no pattern matches AND the target names still
    look like they carry some systematic suffix we do not handle.

    Dual-gene `X-and-Y` targets are NOT collapsed. They are a distinct construct -- the
    perturbation hit two genes -- and silently mapping them onto one gene would invent a
    single-gene claim. They are counted and reported so the analyst decides.
    """
    df = df.copy()
    df["transcript_id"] = df["target_name"]

    names = df["target_name"].unique()
    matched_name, matched_re, n_matched = None, None, 0
    for label, rx in PROMOTER_SUFFIX_PATTERNS:
        n = sum(bool(rx.search(t)) for t in names)
        if n > n_matched:
            matched_name, matched_re, n_matched = label, rx, n

    if n_matched:
        print(
            f"  promoter-suffix convention detected: {matched_name} "
            f"({n_matched} of {len(names)} target labels)"
        )
        df["gene"] = df["target_name"].str.replace(matched_re, "", regex=True)
    else:
        suspicious = sorted(
            {t for t in names if TRAILING_TOKEN.search(t) and not DUAL_GENE.search(t)}
        )
        if suspicious and not allow_no_suffix:
            die(
                "no known alternative-TSS convention matched, but these labels look like they "
                f"carry one: {suspicious[:8]}. Either add the pattern to "
                "PROMOTER_SUFFIX_PATTERNS or pass --allow-no-promoter-suffix if they are "
                "genuinely whole-gene targets."
            )
        print(
            f"  no promoter-suffix convention in this library "
            f"({len(names)} target labels, none suffixed)"
        )
        df["gene"] = df["target_name"]

    dual = sorted(
        df.loc[df["target_name"].str.contains(DUAL_GENE), "target_name"].unique()
    )
    if dual:
        print(
            f"  NOTE: {len(dual)} dual-gene targets left uncollapsed, e.g. {dual[:3]}. "
            f"A hit on one of these is a claim about the PAIR, not either gene alone."
        )

    suffixed = df.loc[df["transcript_id"] != df["gene"], "transcript_id"].unique()

    before = len(df)
    df = (
        df.sort_values("adj_pval")
        .drop_duplicates(subset=["gene", "program"], keep="first")
        .reset_index(drop=True)
    )
    if len(suffixed):
        print(
            f"  collapsed {before} -> {len(df)} rows "
            f"({len(suffixed)} suffixed labels over {df['gene'].nunique()} genes)"
        )
    return df


# ---------------------------------------------------------------------------
# Calibration against whatever control arm the run has
# ---------------------------------------------------------------------------


def measure_control_arm_call_rate(
    ctrl: pd.DataFrame,
    threshold: float,
    min_conditions: int,
    grid=(0.01, 0.05, 0.10, 0.20),
) -> tuple[pd.DataFrame, list[str]]:
    """Per-condition empirical FPR over the control arm, plus loud caveats.

    Returns (table, caveats). The caveats are not decoration -- they decide whether the
    number above is a calibrated null or a diagnostic, and the report must say which.
    """
    caveats: list[str] = []
    n_ctrl = ctrl["gene"].nunique()
    n_cond = ctrl["condition"].nunique()

    if n_ctrl <= 1:
        caveats.append(
            f"The control arm is {n_ctrl} pooled pseudo-target over {len(ctrl)} tests. That "
            f"cannot distinguish a 5% from a 10% false-positive rate. Treat the numbers below "
            f"as a DIAGNOSTIC, not a calibrated null, and say so in Methods."
        )

    rows = []
    for thr in grid:
        sig = ctrl[ctrl["adj_pval"] < thr]
        per_cond = {}
        for cond, g in ctrl.groupby("condition"):
            s = g[g["adj_pval"] < thr]
            per_cond[cond] = f"{len(s)}/{len(g)}"
        n_by_pair = sig.groupby(["gene", "program"])["condition"].nunique()
        called = n_by_pair[n_by_pair >= min(min_conditions, n_cond)].index
        rows.append(
            {
                "threshold": thr,
                "ctrl_per_test_fpr": round(len(sig) / max(len(ctrl), 1), 5),
                "ctrl_pairs_called": len(called),
                **{f"ctrl_sig_{c}": v for c, v in per_cond.items()},
            }
        )
    cal = pd.DataFrame(rows)

    # Is the control arm actually null? A library selected for one phenotype can move its
    # own controls, and then this arm is not a null at any threshold.
    at_thr = ctrl[ctrl["adj_pval"] < threshold]
    if len(at_thr):
        by_cond = at_thr.groupby("condition").size()
        worst = by_cond.idxmax()
        frac = by_cond.max() / max(len(ctrl[ctrl["condition"] == worst]), 1)
        if frac > 0.05:
            sub = at_thr.loc[at_thr["condition"] == worst]
            signs = np.sign(sub["effect"])
            # Two independent signs that this is structure rather than noise. Sign coherence
            # alone is too weak a test: a control arm that is 6 up and 2 down averages 0.5 and
            # slips past, while its q-values run to 1e-12. Depth catches that case.
            coherent = abs(signs.mean()) > SIGN_COHERENCE_MIN
            deep = float(sub["adj_pval"].min()) < threshold / DEPTH_FACTOR
            note = ""
            if coherent and deep:
                note = (
                    " The effects are sign-coherent AND reach q = "
                    f"{sub['adj_pval'].min():.1e}."
                )
            elif coherent:
                note = " The effects are sign-coherent."
            elif deep:
                note = (
                    f" The effects are mixed in sign but reach q = "
                    f"{sub['adj_pval'].min():.1e} -- structured, not noise. Bidirectional "
                    "non-nullity is still non-nullity."
                )
            caveats.append(
                f"The control arm is NOT null in condition {worst!r}: {by_cond.max()} of "
                f"{len(ctrl[ctrl['condition'] == worst])} tests significant at {threshold} "
                f"({frac:.0%})."
                + note
                + (
                    " This looks like real signal the controls also feel, e.g. a library selected "
                    "for one phenotype. Do not use this arm as a null."
                    if (coherent or deep)
                    else ""
                )
            )
    return cal, caveats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("input (either a per-condition set, or one table)")
    src.add_argument("--crt-dir", type=Path)
    src.add_argument(
        "--crt-glob", help="filename pattern with a literal {cond} placeholder"
    )
    src.add_argument("--conditions", nargs="+", default=None)
    src.add_argument(
        "--effects", type=Path, help="a single effects table (one condition)"
    )
    src.add_argument(
        "--condition-name",
        default="all",
        help="label for the single-table case (default: all)",
    )

    ap.add_argument(
        "--ntc-in-table",
        default=None,
        help="control target name inside the main table, e.g. non-targeting",
    )
    ap.add_argument(
        "--ntc-glob",
        default=None,
        help="separate control table pattern with {cond}, if the run ships one",
    )
    ap.add_argument("--n-programs", type=int, required=True)
    ap.add_argument(
        "--exclude-programs",
        nargs="*",
        default=[],
        help="technical programs (CNV, cell cycle, ambient) to drop entirely",
    )
    ap.add_argument("--target-column", default="target_name")
    ap.add_argument("--program-column", default="program_name")
    ap.add_argument("--pval-column", default=None)
    ap.add_argument("--adj-pval-column", default=None)
    ap.add_argument("--log2fc-column", default=None)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument(
        "--min-conditions",
        type=int,
        default=2,
        help="clamped to the number of conditions actually present",
    )
    ap.add_argument(
        "--ordered",
        action="store_true",
        help="the conditions are an ORDERED series (a time course), given in order "
        "via --conditions. Changes the gate: see the module docstring.",
    )
    ap.add_argument(
        "--endpoint-min-programs",
        type=int,
        default=5,
        help="with --ordered, a target significant only at the terminal condition is "
        "admitted if it moves at least this many programs there (default 5)",
    )
    ap.add_argument("--allow-no-promoter-suffix", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Create the output directory up front: the calibration table is written before the
    # main one, so deferring this to the end makes calibration fail on a fresh run.
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- assemble the per-condition tables --------------------------------------
    sources: list[tuple[str, Path]] = []
    if args.effects:
        sources = [(args.condition_name, args.effects)]
    elif args.crt_dir and args.crt_glob and args.conditions:
        sources = [
            (c, args.crt_dir / args.crt_glob.format(cond=c)) for c in args.conditions
        ]
    else:
        die("give either --effects, or --crt-dir + --crt-glob + --conditions")

    frames, ntc_frames, provenance = [], [], None
    for cond, path in sources:
        if not path.exists():
            die(f"missing table for {cond}: {path}")
        print(f"[{cond}] {path.name}")
        d, prov = load_table(
            path,
            args.pval_column,
            args.adj_pval_column,
            args.log2fc_column,
            args.target_column,
            args.program_column,
        )
        provenance = provenance or prov
        d = collapse_target_names(d, args.allow_no_promoter_suffix)
        d["condition"] = cond

        if args.ntc_in_table:
            mask = d["gene"] == args.ntc_in_table
            if not mask.any():
                die(
                    f"--ntc-in-table {args.ntc_in_table!r} is not a target in {path.name}"
                )
            ntc_frames.append(d[mask].copy())
            d = d[~mask]
        frames.append(d)

        if args.ntc_glob:
            npath = args.crt_dir / args.ntc_glob.format(cond=cond)
            if not npath.exists():
                die(f"missing control table for {cond}: {npath}")
            n, _ = load_table(
                npath,
                args.pval_column,
                args.adj_pval_column,
                args.log2fc_column,
                args.target_column,
                args.program_column,
            )
            n = collapse_target_names(n, allow_no_suffix=True)
            n["condition"] = cond
            ntc_frames.append(n)

    print("\nColumns used (record these in Methods):")
    for k, (col, why) in provenance.items():
        print(f"  {k:9s} -> {col:18s} ({why})")

    crt = pd.concat(frames, ignore_index=True)

    drop = [str(p) for p in args.exclude_programs]
    if drop:
        crt["program"] = crt["program"].astype(str)
        unknown = [p for p in drop if p not in set(crt["program"])]
        if unknown:
            die(f"--exclude-programs not present: {unknown}")
        crt = crt[~crt["program"].isin(drop)]
        print(f"\n  excluded {len(drop)} technical programs: {drop}")

    n_prog = crt["program"].nunique()
    expected = args.n_programs - len(drop)
    if n_prog != expected:
        die(f"found {n_prog} programs after exclusions, expected {expected}")
    if crt[["gene", "program", "condition"]].duplicated().any():
        die("duplicate (gene, program, condition) rows after the collapse")
    if crt[["effect", "p_value", "adj_pval"]].isna().any().any():
        die("nulls in a value column after the join")

    conditions = list(dict.fromkeys(crt["condition"]))
    min_cond = min(args.min_conditions, len(conditions))
    if min_cond < args.min_conditions:
        print(
            f"\n  NOTE: --min-conditions {args.min_conditions} clamped to {min_cond} -- this "
            f"run has {len(conditions)} condition(s)."
        )
        if len(conditions) == 1:
            print(
                "  Cross-condition replication is UNAVAILABLE. The report must name what is "
                "carrying that weight instead (see references/failure-modes.md)."
            )

    # --- calibrate ---------------------------------------------------------------
    if ntc_frames:
        ctrl = pd.concat(ntc_frames, ignore_index=True)
        cal, caveats = measure_control_arm_call_rate(ctrl, args.threshold, min_cond)
        cal.to_csv(args.out.with_name("control_arm_calibration.csv"), index=False)
        print("\nControl-arm calibration (report this beside the gate):")
        print(cal.to_string(index=False))
        for c in caveats:
            print(f"\n  ** {c}")
    else:
        print(
            "\nWARNING: no control arm supplied (--ntc-in-table / --ntc-glob). The threshold "
            "is ASSUMED, not calibrated. Say so in Methods."
        )

    # --- the gate ----------------------------------------------------------------
    crt["sig"] = crt["adj_pval"] < args.threshold
    sig = crt[crt["sig"]]

    # Which conditions, not just how many: for an ordered time course "significant in >=2"
    # does not distinguish early, late, and split.
    def sign_pattern(g):
        by_cond = {c: np.sign(v) for c, v in zip(g["condition"], g["effect"])}
        return ",".join(
            f"{'+' if by_cond[c] > 0 else '-'}" for c in conditions if c in by_cond
        )

    per_pair = (
        sig.groupby(["gene", "program"])
        .apply(
            lambda g: pd.Series(
                {
                    "n_conditions": g["condition"].nunique(),
                    "conditions": ",".join(
                        c for c in conditions if c in set(g["condition"])
                    ),
                    "signs": sign_pattern(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_pair["sign_flips"] = per_pair["signs"].str.contains(r"\+") & per_pair[
        "signs"
    ].str.contains("-")

    # Classify rather than filter. A time course makes two things look like failures that are
    # not: an effect present only at the endpoint, and an effect that changes sign partway.
    # Both were silently discarded by a plain ">= N conditions, same sign" rule, and both cost
    # a real run its headline results -- see ../references/failure-modes.md.
    terminal = conditions[-1]
    order = {c: i for i, c in enumerate(conditions)}

    def is_contiguous(cond_str: str) -> bool:
        idx = sorted(order[c] for c in cond_str.split(","))
        return idx == list(range(idx[0], idx[0] + len(idx)))

    def classify(r):
        if r["n_conditions"] >= min_cond:
            if r["sign_flips"]:
                return "sign_flip"
            # An ordered series makes "two conditions" mean two different things. Adjacent
            # conditions are enrichment over independent draws; a GAP is not. Measured on a
            # real run: adjacent day pairs were enriched 2.3-3.8x over an independence null
            # while non-adjacent pairs sat at 0.95-1.29x -- i.e. indistinguishable from two
            # independent per-condition draws, carrying no replication evidence at all. 44%
            # of that run's gate pairs were in the gapped class, pooled with the contiguous
            # ones as if they meant the same thing.
            if args.ordered and not is_contiguous(r["conditions"]):
                return "persistent_gapped"
            return "persistent"
        if args.ordered and r["conditions"] == terminal:
            return "endpoint_only"
        return "transient"

    per_pair["pair_class"] = per_pair.apply(classify, axis=1)

    # endpoint_only is admitted on BREADTH: one program at the last timepoint is noise, but a
    # target moving many programs there is the endpoint biology this design exists to capture.
    endpoint_breadth = (
        per_pair[per_pair["pair_class"] == "endpoint_only"]
        .groupby("gene")["program"]
        .nunique()
    )
    admitted_endpoint = set(
        endpoint_breadth[endpoint_breadth >= args.endpoint_min_programs].index
    )
    per_pair["admitted"] = per_pair["pair_class"].isin(
        ["persistent", "persistent_gapped", "sign_flip"]
    ) | (
        (per_pair["pair_class"] == "endpoint_only")
        & per_pair["gene"].isin(admitted_endpoint)
    )

    print("\nPair classes (counts of target x program pairs):")
    for k, v in per_pair["pair_class"].value_counts().items():
        adm = int(per_pair.loc[per_pair["pair_class"] == k, "admitted"].sum())
        print(f"  {k:14s} {v:6d}   admitted {adm}")
    n_gapped = int((per_pair["pair_class"] == "persistent_gapped").sum())
    if n_gapped:
        print(
            f"  NOTE: {n_gapped} admitted pairs are `persistent_gapped` -- significant in two "
            f"or more NON-ADJACENT conditions. Admitted, but they carry NO replication "
            f"control: on a real run this class was statistically indistinguishable from two "
            f"independent per-condition draws. Label them on the hypothesis card; do not pool "
            f"them with the contiguous pairs."
        )
    if per_pair["sign_flips"].any():
        print(
            f"  NOTE: {int(per_pair['sign_flips'].sum())} pairs change sign across conditions. "
            f"These are ADMITTED and labelled, not dropped -- a sign flip across an ordered "
            f"series is a temporal switch, which is a finding."
        )
    if (
        args.ordered
        and not admitted_endpoint
        and (per_pair["pair_class"] == "endpoint_only").any()
    ):
        print(
            f"  NOTE: endpoint-only pairs exist but no target reaches "
            f"--endpoint-min-programs {args.endpoint_min_programs}; none admitted."
        )

    robust = per_pair[per_pair["admitted"]]

    per_gene = (
        robust.groupby("gene")
        .agg(
            n_programs_robust=("program", "nunique"),
            conditions_hit=("conditions", lambda s: ";".join(sorted(set(s))[:3])),
            pair_classes=("pair_class", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
    )
    breadth = (
        sig.groupby("gene")["program"].nunique().rename("n_programs_any").reset_index()
    )
    top = (
        crt.assign(abs_eff=crt["effect"].abs())
        .sort_values("abs_eff", ascending=False)
        .drop_duplicates("gene")[["gene", "program", "effect", "abs_eff", "condition"]]
        .rename(
            columns={
                "program": "top_program",
                "effect": "top_effect",
                "abs_eff": "max_abs_effect",
                "condition": "top_condition",
            }
        )
    )

    # EVERY target gets a row, including those with no significant call -- the report owes a
    # reader the denominator, and a table built only from survivors cannot supply it.
    table = pd.DataFrame({"gene": sorted(set(crt["gene"]))})
    table = (
        table.merge(breadth, on="gene", how="left")
        .merge(per_gene, on="gene", how="left")
        .merge(top, on="gene", how="left")
    )
    table["n_programs_any"] = table["n_programs_any"].fillna(0).astype(int)
    table["n_programs_robust"] = table["n_programs_robust"].fillna(0).astype(int)
    table["is_dual_gene"] = table["gene"].str.contains(DUAL_GENE)
    table["top_direction"] = np.where(
        table["top_effect"].isna(),
        None,
        np.where(table["top_effect"] > 0, "brake", "driver"),
    )
    table["passes_gate"] = table["n_programs_robust"] >= 1
    table["excluded_because"] = np.where(
        table["passes_gate"],
        "",
        np.where(
            table["n_programs_any"] == 0,
            f"no program significant at adj < {args.threshold}",
            f"significant, but no pair admitted (see pair_class in the pairs file)",
        ),
    )
    table = table.sort_values(
        ["passes_gate", "max_abs_effect"], ascending=[False, False]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    robust.to_csv(args.out.with_name("significant_pairs_by_condition.csv"), index=False)

    n_pass = int(table["passes_gate"].sum())
    print(
        f"\n{n_pass} of {len(table)} targets pass "
        f"(adj < {args.threshold} in >= {min_cond} of {len(conditions)}) -> {args.out}"
    )
    print(
        f"  the other {len(table) - n_pass} rows carry `excluded_because` and are the "
        f"denominator table the report needs"
    )
    print(
        f"  per-pair condition membership -> {args.out.with_name('significant_pairs_by_condition.csv').name}"
    )
    if table["is_dual_gene"].any():
        print(
            f"  {int(table['is_dual_gene'].sum())} passing rows are dual-gene targets -- a hit "
            f"is a claim about the pair"
        )

    print("\nNEXT, before any hypothesis is written:")
    print("  1. check_cis_neighbours.py     -- no locus claim without it")
    print("  2. expression filter           -- not expressed in this system -> out")
    print("  3. BH-correct any program-level claim")
    print("  4. the adversarial literature pass -- BEFORE writing")


if __name__ == "__main__":
    main()
