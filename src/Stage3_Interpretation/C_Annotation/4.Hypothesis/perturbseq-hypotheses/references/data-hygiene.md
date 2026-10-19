# Data hygiene — the rules the Explorer and Analyst follow

These are the expensive lessons from v1 runs about reading a PerturbNMF run correctly. They are
unchanged in v2; they moved here from the old stage list so `SKILL.md` can describe the loop.
The Explorer applies §1–§3 once, before any hypothesis exists, and runs §2's kill-checks on
**every** admitted regulator, so the Generator never proposes a claim one of them already kills.
The Analyst reruns a check as a falsification test only where the Explorer left a `flag`.

---

## 1. Read the upstream pipeline before doing anything

Locate and read, in this order:

1. The run directory's own `README` / `Script/README.md`.
2. Any calibration report the run ships (`*CRT_calibration*`, `*_ntc_significance_*`).
3. `GeneProgramExplorer/docs/INPUT.md` — the canonical PerturbNMF→downstream input contract.
4. The analysis repo's `.living/` (`decisions.md`, `learnings.md`, `findings/`) and any baton.

Then record, in the report's Methods, **which significance column you are using and why.**

> **Hard rule. If the statistic you want already exists in the run, use it. Do not build a new
> one.** This is the single most expensive mistake in the prior work: a bespoke replication
> statistic duplicated a better-calibrated test that shipped with the run, one directory away from
> files already open. About five times as many targets passed the pipeline's test as the home-made
> one, and three report rebuilds followed. If you believe a new statistic is really needed, say so explicitly
> and get agreement before writing it.

**Resolve every gene symbol against the library's own target names before writing "absent".**
Libraries carry deprecated aliases and read-through symbols. In one run a screen's headline gene
was listed under its superseded alias and a plain symbol match reported it as never tested — it
was in fact the largest effect in its class. In another, a target was present only as a
read-through (`X-Y`) symbol. Build the alias map once, fail loud when it goes stale, and never
conclude "this gene was not tested" from a failed string match.

**Dual-gene `X-and-Y` targets are not single-gene perturbations.** Do not collapse them onto
either gene — a hit is a claim about the pair. Count them and say so.

Two conventions to assert rather than assume, both fail-loud:

- **Use the `*_raw` significance pair *where the run offers both*.** The non-`_raw` column is
  often a different q-value-style transform never checked against the control arm, calling
  roughly twice as many pairs significant. Many runs ship only one pair — then use it and
  **record in Methods that only one existed**, so nobody assumes the `_raw` convention applied.
- **Check what the effect column actually is.** `log2FC`, `approx_log2FC` and `beta_hat` are not
  interchangeable, and a run may ship several. Where a post-hoc conversion produced the column,
  read the converting script before quoting magnitudes — one such script's docstring, its own
  sanity-check cell, and the line that executed disagreed with each other by 4.1%.
- **Collapse `-P<n>` promoter suffixes onto the base gene**, failing loud if the collapse invents a
  regulator, drops one, merges nothing, or rewrites the non-targeting control.


---

## 2. Which regulators are worth hypothesising about, and the kill-checks

This used to be the "admission gate". In v2 it is only the Explorer's filter for which
regulators get summarised for the Generator. It is never reported as a funnel.

One gate, from the run's own test: significant in ≥2 independent conditions/strata at the
threshold its control arm calibrates. Report the empirical false-positive rate beside it.
`scripts/build_candidate_table.py` is the reference implementation.

**Report *which* conditions, not just how many.** For an ordered time course, "significant in ≥2"
does not distinguish early, late, and split — and those are different claims.

**If the conditions are an ordered series, pass `--ordered` and classify rather than filter.** A
plain "≥2 conditions, same sign" rule discards two things that are not failures in a time course,
and in a real run it cost the report the parent manuscript's own headline result:

| Class | What it is | Verdict |
|---|---|---|
| `persistent` | significant ≥2 conditions, one sign | admitted |
| `sign_flip` | significant ≥2 conditions, sign changes partway | **admitted and labelled** — a temporal switch is a finding, not noise |
| `endpoint_only` | significant at the terminal condition only | **admitted on breadth** (`--endpoint-min-programs`) — one program at the last timepoint is noise, many is the endpoint biology the design exists to capture |
| `transient` | one interior condition only | held out, reported |

Never enforce same-sign as a filter across an ordered axis. In the run that found this, the
same-sign rule dropped four members of one signalling pathway on one program that go negative
early and positive later — precisely the transition the study was about.

**If the run has only one condition**, cross-condition replication is unavailable and the report
must name what carries that weight instead. In order of preference:

1. **Cross-factorization replication** — a second, independent factorization of the same cells
   scored through the identical pipeline. Controls for factorization instability, which is the
   biggest threat to a cNMF program-level claim. It does **not** control for biological or
   technical noise — both factorizations see the same cells — and the report must say so rather
   than calling it replication.
2. **Coherence across related programs** — one regulator moving several programs of one family.
3. **Neighbour support** — the effect vector correlating with a known partner's.

Never fabricate a second condition by splitting programs, cells, or guides after the fact.

**The control arm is usually a target inside the main table** (`safe-targeting`,
`non-targeting`), not a separate file. Two failure modes the script now checks for, both real:
a single pooled pseudo-target gives too few tests to calibrate anything (report it as a
diagnostic, not a null), and a library selected for one phenotype can move its own controls
(then it is not a null at any threshold).

Then four checks, each cheap, each of which kills hypotheses *before* they are written:

1. **On-target / cis-neighbour silencing** — `scripts/check_cis_neighbours.py`. Measure every
   candidate's genomic neighbours as responses. **No locus-level claim is made without this.**
2. **Expression** — not expressed in this system → out, unless the gene is canonically expressed
   here and the value is suspect. **State the unit in the constant's name and check it against
   the normalisation your matrix actually uses.** CPM and CP10K differ by 100×, and one run
   implemented a documented `0.5 CPM` floor as `0.5 CP10K`: 100× stricter, moving 70 of 192
   candidates' on-target verdicts. Report the verdict counts at both floors if the choice is
   arguable, so the sensitivity is visible rather than buried in a constant.
3. **Multiple testing on the program axis** — any "program X moves with \<covariate\>" claim is
   BH-corrected before it can carry a hypothesis.
4. **Per-guide concordance** — flag any target whose signal rests on one guide. Animal- or
   replicate-level reproducibility cannot separate on-target from guide off-target effects; only
   guide-vs-guide concordance can. **But calibrate the threshold and pick the conditions
   carefully, or the test is worse than nothing.** On a real run, scoring "fraction of guides
   agreeing in sign with the target-level coefficient" gave a *null* of 0.68, not 0.5 — because
   the comparison sign comes from an aggregate over those same guides' cells. A 0.5 cut flagged
   1 of 189 real pairs while flagging 19% of never-significant pairs. And scoring it on the
   conditions the pair was *admitted for* gave mean 0.95, against 0.66 on held-out conditions —
   indistinguishable from the null. So: score on **held-out or all** conditions, never only the
   admitted ones, and set the threshold from a **permutation null over never-significant
   pairs**, not at a round number.

**Run checks 1, 2 and 4 on every admitted regulator, at Explore, before generation**, not only
on regulators a hypothesis later names. The cis check covers every admitted target's neighbours,
not a shortlist. In one run these checks were left to the loop, and about a quarter of the
killed hypotheses died on a cis-neighbour or per-guide result that could have been computed
before any hypothesis was written. Write the results to `explore/regulator_eligibility.tsv`,
one row per admitted regulator:

| Column | Values |
|---|---|
| `target` | exactly as in the regulator table |
| `cis_verdict` | `clean` / `cis_separable` (a neighbour moves, but at least one guide separates the target from it) / `cis_likely` (a neighbour moves and no guide separates them) / `not_testable` |
| `guide_verdict` | `pass` / `fail` (below the permutation-null threshold) / `too_few_guides` |
| `expressed` | `yes` / `no` / `suspect` (canonically expressed here, implausible value) |
| `lead_eligible` | `yes` / `flag` / `no` |
| `reason` | one line, with the numbers |

`lead_eligible` is `no` if `cis_verdict` is `cis_likely`, `guide_verdict` is `fail`, or
`expressed` is `no`; `flag` if any verdict is `cis_separable`, `not_testable`, `too_few_guides`
or `suspect`; otherwise `yes`. A `no` regulator cannot be in a hypothesis's `members`. It may
still appear in the evidence as support, labelled with its reason. A `flag` regulator may be a
member, but the Falsifier registers the matching check as one of its tests. The validator
enforces the `no` rule (`--eligibility`).

These three are the only gates. Everything below stays a covariate.

**Not gates. Reported as covariates:** knockdown efficiency, number of significant calls, DepMap
essentiality. Knockdown efficiency in particular correlates ~+0.08 with whether a response
reproduces — gating on it discards real biology. A gene can have feedback maintaining its own
transcript and still show real downstream effects.


---

## 3. Evidence axes and program identity

Compute for every admitted candidate; write each to a committed CSV under `explore/`.

| Axis | Note |
|---|---|
| signed effect + breadth, per condition | the primary quantity |
| **excess pleiotropy** = residual of breadth regressed on effect magnitude | raw breadth is confounded with effect size, so penalising raw breadth rewards weak perturbations |
| **perturbation neighbours** (± top k by effect-vector correlation) | `scripts/perturbation_neighbours.py`. The cheapest and most informative axis for an understudied gene — it is what placed a gene with zero tissue literature into a known signalling pathway |
| condition / stratum contrast | reported in plain English (see `references/voice.md`), with the reproducibility statistic as a gate, not a headline |
| literature counts (curated + topic-restricted) | carry an ambiguous-symbol flag |
| human genetics / disease-locus membership | annotation layer. Small weight, never a gate |
| program identity, **per annotation source** | carry all sources and label each: the screen author's, the GPE/LLM-derived one, GO/geneset enrichment |

Program top genes: the two rankings fail in opposite directions, so **run both and say which one
carried the identity.** Ranking by loading **z** is specificity — it surfaces sub-5-CPM genes and
has produced a completely wrong program identity. Ranking by **TPM/abundance** is what the
program expresses — but raw TPM is often dominated by mitochondrial, MALAT1 and unnamed ENSG
transcripts, which labelled 50 of 50 programs mitochondrial in one run. Filtering those does not
always rescue it: in another run the filtered abundance ranking was still at least half
housekeeping for nearly every program, and the loading-ranked list was the only one that carried
identity. There is no rule that works on every run. Compute both, filter the junk, look at the
result, and record which you used per program.

Program titles: 2–4 words (6 only to separate similar ones), never the word "program" in the
title, globally unique and coordinated — disambiguate by a distinguishing gene, pathway, or
concept, not by a bare number.
