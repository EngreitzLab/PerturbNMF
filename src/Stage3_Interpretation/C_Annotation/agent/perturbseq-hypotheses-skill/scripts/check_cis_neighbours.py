#!/usr/bin/env python3
"""On-target check: did CRISPRi at this promoter also silence a genomic neighbour?

REFERENCE IMPLEMENTATION. Adapt the I/O, commit the adapted copy into the analysis dir.

Why this exists
---------------
CRISPRi at a promoter silences whatever else that promoter drives. In prior reports this
killed two hypotheses outright -- in one case the guides silenced a neighbouring gene whose
loss manufactures precisely the signature the hypothesis rested on -- and a third had to be
withdrawn because two genes at one locus were knocked down 38.6% and 5.9%, so the screen
could not adjudicate the locus at all.

It is one command, and in one prior run it "settled five disputes". Run it before any
locus-level claim, not after.

What it does
------------
For every candidate, find the genes whose TSS lies within --window of the candidate's TSS
(and flag bidirectional promoters, < --bidirectional-bp apart on opposite strands). Then
report how much each neighbour MOVED AS A MEASURED RESPONSE when the candidate was
targeted. A neighbour that moves is a confound; a neighbour that does not is the check
passing, and both belong in the report.

IF THE RUN SHIPS NO GENE-LEVEL DE TABLE
---------------------------------------
Several PerturbNMF runs emit only program-level results, so there is no per-(perturbed gene,
measured gene) table to pass as --responses. Do not give up on the check and do not invent a
statistic: the pipeline's own knockdown-efficiency estimator is the right one pointed at the
wrong gene. It compares a gene's CP10K expression in cells carrying a target's guides against
control cells; run it for the candidate's genomic NEIGHBOURS instead of for the candidate,
and feed the result in here as --responses. That is an adaptation of existing pipeline code,
not a new test.

If neither exists, the check cannot run -- and then the report makes NO locus-level claims at
all, and says so in Limitations. Do not soften this into "the neighbour was probably fine".

Inputs
------
  --gtf        GENCODE GTF for the right build and species. CHECK THE BUILD: an hg19
               coordinate reported as GRCh38 turned 13.7 kb into 27.9 kb in a shipped draft.
  --responses  Long table of per-(perturbed gene, measured gene) effects, i.e. the
               differential-expression output, not the program-level table. Needs columns
               --perturbed-col, --measured-col, --effect-col, and optionally --padj-col.
  --candidates One gene symbol per line, or a CSV with a `gene` column.

Usage
-----
    python check_cis_neighbours.py \
        --gtf /oak/.../gencode.vM31.annotation.gtf.gz \
        --responses outputs/de_results_long.csv \
        --candidates outputs/candidate_table.csv \
        --window 200000 \
        --out outputs/cis_neighbour_check.csv
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def read_tss(gtf: Path) -> pd.DataFrame:
    """One TSS per gene symbol, from the GTF's gene features."""
    opener = gzip.open if gtf.suffix == ".gz" else open
    rows = []
    with opener(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            attrs = f[8]
            name = None
            for key in ('gene_name "', 'gene_id "'):
                i = attrs.find(key)
                if i != -1:
                    j = attrs.find('"', i + len(key))
                    name = attrs[i + len(key) : j]
                    break
            if name is None:
                continue
            start, end, strand = int(f[3]), int(f[4]), f[6]
            rows.append(
                {
                    "gene": name,
                    "chrom": f[0],
                    "strand": strand,
                    "tss": start if strand == "+" else end,
                    "start": start,
                    "end": end,
                }
            )
    tss = pd.DataFrame(rows)
    if tss.empty:
        die(f"no gene features parsed from {gtf}")
    # keep the longest annotation per symbol
    tss["span"] = tss["end"] - tss["start"]
    tss = tss.sort_values("span", ascending=False).drop_duplicates("gene")
    print(f"  {len(tss)} genes from {gtf.name}")
    return tss.drop(columns="span")


HEADERS = {"gene", "genes", "symbol", "target", "target_gene", "gene_name"}


def read_candidates(path: Path) -> list[str]:
    """Accept a CSV with a gene-ish column, or one symbol per line.

    A one-column CSV defeats pandas' delimiter sniffing, so the line-split fallback has to
    drop the header itself -- otherwise the word "gene" is carried through as a candidate
    and shows up as a spurious 'absent from the GTF' warning.
    """
    for sep in (None, ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
        for col in df.columns:
            if str(col).strip().lower() in HEADERS:
                return df[col].dropna().astype(str).str.strip().unique().tolist()
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if lines and lines[0].lower() in HEADERS:
        lines = lines[1:]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument(
        "--window",
        type=int,
        default=200_000,
        help="bp around the candidate TSS to call a neighbour (default 200 kb)",
    )
    ap.add_argument(
        "--bidirectional-bp",
        type=int,
        default=1000,
        help="TSS separation on opposite strands that counts as a shared "
        "bidirectional promoter (default 1 kb)",
    )
    ap.add_argument("--perturbed-col", default="perturbed_gene")
    ap.add_argument("--measured-col", default="measured_gene")
    ap.add_argument("--effect-col", default="log2FC")
    ap.add_argument("--padj-col", default=None)
    ap.add_argument(
        "--flag-effect",
        type=float,
        default=0.1,
        help="|effect| on a neighbour at or above which the candidate is flagged",
    )
    ap.add_argument(
        "--min-neighbour-expression",
        type=float,
        default=None,
        help="neighbours below this expression are UNADJUDICABLE, not clean. The "
        "floor is applied AFTER the confounded test, never before it.",
    )
    ap.add_argument(
        "--expression-col",
        default="neighbour_expression",
        help="column in --responses carrying the neighbour's expression level",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tss = read_tss(args.gtf)
    by_symbol = tss.set_index("gene")
    resp = pd.read_csv(args.responses, sep=None, engine="python")
    for col in (args.perturbed_col, args.measured_col, args.effect_col):
        if col not in resp.columns:
            die(
                f"{args.responses.name}: no column {col!r}. Present: {sorted(resp.columns)}"
            )

    candidates = read_candidates(args.candidates)
    print(f"  {len(candidates)} candidates, {len(resp)} response rows")

    missing = [
        g
        for g in candidates
        if not ({g} | set(g.split("-and-"))) & set(by_symbol.index)
    ]
    if missing:
        print(
            f"  WARNING: {len(missing)} candidates absent from the GTF "
            f"(symbol drift or build mismatch?): {missing[:10]}"
        )

    rows = []
    for gene in candidates:
        # A dual-gene `X-and-Y` construct has no GTF entry under its full label, so looking
        # it up whole silently skips it. Anchor on each half and keep the construct's name
        # in the output, so the pair stays auditable.
        anchors = [
            a for a in ({gene} | set(gene.split("-and-"))) if a in by_symbol.index
        ]
        if not anchors:
            continue
        for anchor in anchors:
            me = by_symbol.loc[anchor]
            # Exclude EVERY half of a dual-gene target, not just the current anchor. For an
            # `X-and-Y` construct both genes are the intended perturbation, so finding Y
            # beside X is the design, not a confound. One run excluded only the anchor and
            # so called all five of its gated dual-gene candidates "confounded in cis" --
            # inflating its published confounded count by 20% and removing five candidates
            # for a reason that was not a reason.
            self_names = {gene, anchor, *gene.split("-and-")}
            near = tss[
                (tss["chrom"] == me["chrom"])
                & (~tss["gene"].isin(self_names))
                & ((tss["tss"] - me["tss"]).abs() <= args.window)
            ]
            if near.empty:
                rows.append(
                    {
                        "gene": gene,
                        "neighbour": None,
                        "distance_bp": None,
                        "bidirectional": False,
                        "measured": False,
                        "neighbour_effect_signed": None,
                        "neighbour_effect_abs": None,
                        "neighbour_direction": None,
                        "neighbour_padj": None,
                        "verdict": "no neighbour within window",
                    }
                )
                continue

            obs = resp[resp[args.perturbed_col] == gene]
            for _, nb in near.iterrows():
                dist = int(nb["tss"] - me["tss"])
                bidir = (abs(dist) <= args.bidirectional_bp) and (
                    nb["strand"] != me["strand"]
                )
                hit = obs[obs[args.measured_col] == nb["gene"]]
                if hit.empty:
                    eff = signed = padj = None
                    direction = None
                    verdict = "unadjudicable - neighbour not measured"
                else:
                    # Keep the SIGN. A neighbour that goes DOWN and one that goes UP are
                    # different confounds, and collapsing to |effect| loses the distinction:
                    #   down -> CRISPRi repression spread from the shared/adjacent promoter.
                    #           The classic case; the candidate's phenotype may be the
                    #           neighbour's.
                    #   up   -> not repression spread. Relieved promoter competition, an
                    #           enhancer redistributing, or the neighbour being a direct target
                    #           of something the candidate represses. Still a confound for a
                    #           locus-level claim, but it needs a different explanation and
                    #           should not be written up as "the guide silenced the neighbour".
                    idx = hit[args.effect_col].abs().idxmax()
                    signed = float(hit.loc[idx, args.effect_col])
                    eff = abs(signed)
                    direction = "down" if signed < 0 else "up"
                    padj = (
                        float(hit[args.padj_col].min())
                        if args.padj_col and args.padj_col in hit.columns
                        else None
                    )
                    verdict = (
                        f"CONFOUNDED — neighbour moves {direction}"
                        if eff >= args.flag_effect
                        else "clean"
                    )
                    # A low-expression neighbour is UNADJUDICABLE, not clean -- and the order of
                    # these two tests matters. Put an expression floor ahead of the confounded
                    # test and a neighbour that moves hard but sits near the detection floor can
                    # never be called confounded, however far it moves. On a real run nine such
                    # rows exceeded |0.75| and every one was on a shortlisted candidate. So:
                    # confounded first, floor second, and the floor yields `unadjudicable`.
                    if verdict == "clean" and args.min_neighbour_expression is not None:
                        expr = (
                            hit[args.expression_col].max()
                            if args.expression_col in hit
                            else None
                        )
                        if (
                            expr is not None
                            and float(expr) < args.min_neighbour_expression
                        ):
                            verdict = "unadjudicable - neighbour below expression floor"
                rows.append(
                    {
                        "gene": gene,
                        "neighbour": nb["gene"],
                        "distance_bp": dist,
                        "bidirectional": bool(bidir),
                        "measured": not hit.empty,
                        "neighbour_effect_signed": signed,
                        "neighbour_effect_abs": eff,
                        "neighbour_direction": direction,
                        "neighbour_padj": padj,
                        "verdict": verdict,
                    }
                )

    out = pd.DataFrame(rows)
    # A neighbour in range of both halves of a dual-gene target appears once per anchor.
    # Keep the strongest observation per (candidate, neighbour) so counts are per pair.
    if not out.empty and out.duplicated(["gene", "neighbour"]).any():
        out = (
            out.sort_values("neighbour_effect_abs", ascending=False, na_position="last")
            .drop_duplicates(["gene", "neighbour"], keep="first")
            .sort_values(["gene", "distance_bp"])
            .reset_index(drop=True)
        )
    if out.empty:
        die(
            "no candidate resolved to a GTF entry -- check the symbol vocabulary of "
            "--candidates against the GTF (aliases, read-through and dual-gene names)."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # Per-candidate adjudicability. A candidate whose neighbours are ALL unadjudicable has not
    # passed the check -- it was never tested. Reporting that as a pass is absence of evidence
    # dressed as evidence of absence, and on a real run 24 of 71 candidates passed that way.
    adjud = (~out["verdict"].str.startswith("unadjudicable", na=False)) & out[
        "neighbour"
    ].notna()
    per_cand = (
        out.assign(adjudicable=adjud)
        .groupby("gene")
        .agg(
            n_neighbours=("neighbour", "count"),
            n_adjudicable=("adjudicable", "sum"),
            n_confounded=(
                "verdict",
                lambda v: int(v.str.startswith("CONFOUNDED", na=False).sum()),
            ),
        )
    )
    per_cand["cis_verdict"] = np.where(
        per_cand["n_confounded"] > 0,
        "confounded",
        np.where(
            per_cand["n_adjudicable"] > 0, "clean", "UNADJUDICABLE - never tested"
        ),
    )
    per_cand.to_csv(args.out.with_name(args.out.stem + "_by_candidate.csv"))

    flagged = sorted(
        out.loc[out["verdict"].str.startswith("CONFOUNDED", na=False), "gene"].unique()
    )
    bidir = sorted(out.loc[out["bidirectional"], "gene"].unique())
    unmeasured = sorted(
        out.loc[out["verdict"] == "neighbour not measured", "gene"].unique()
    )

    print(f"\n-> {args.out}")
    print(f"  CONFOUNDED (a neighbour moves >= {args.flag_effect}): {len(flagged)}")
    for g in flagged:
        sub = out[(out.gene == g) & out.verdict.str.startswith("CONFOUNDED", na=False)]
        for _, r in sub.iterrows():
            print(
                f"    {g}: {r.neighbour} at {int(r.distance_bp):+,} bp, "
                f"effect {r.neighbour_effect_signed:+.3f} ({r.neighbour_direction})"
                + ("  [bidirectional promoter]" if r.bidirectional else "")
            )
    print(f"  shares a bidirectional promoter: {len(bidir)} {bidir[:10]}")
    never = per_cand.index[
        per_cand["cis_verdict"].str.startswith("UNADJUDICABLE")
    ].tolist()
    print(
        f"  clean (at least one adjudicable neighbour, none moving): "
        f"{int((per_cand['cis_verdict'] == 'clean').sum())}"
    )
    print(
        f"  **UNADJUDICABLE — every neighbour untestable, so NOT a pass: {len(never)}** "
        f"{never[:8]}"
    )
    print(
        f"  -> per-candidate verdicts: {args.out.with_name(args.out.stem + '_by_candidate.csv').name}"
    )
    print(
        "  A candidate in the UNADJUDICABLE set must not carry a locus-level claim, and its "
        "hypothesis card must say the cis check was uninformative -- not that it was clean."
    )
    up = out[
        out["neighbour_direction"].eq("up")
        & out["verdict"].str.startswith("CONFOUNDED", na=False)
    ]
    if len(up):
        print(
            f"  of those, {up['gene'].nunique()} have a neighbour moving UP -- not repression "
            f"spread. Explain it as something else, or drop the locus claim; do not write it "
            f"up as the guide silencing the neighbour."
        )
    print(
        "\nReport the clean results too. 'Its neighbours move < 0.09' is evidence a reader "
        "needs, not boilerplate."
    )


if __name__ == "__main__":
    main()
