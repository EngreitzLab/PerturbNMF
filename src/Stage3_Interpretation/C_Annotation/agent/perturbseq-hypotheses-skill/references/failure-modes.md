# Failure modes — every one of these shipped in a draft

Each one comes from a real run. Specific genes and results are replaced by placeholders or
textbook biology so this file can be read before a blind run. Where an `examples/` directory is
present (it is absent from sterile runs), it holds the original drafts; nothing here depends on it.

---

## Statistics

**Do not build a statistic before reading the pipeline that produced the data.**
The most expensive error in the prior work. A bespoke replication statistic duplicated a
better-calibrated test that shipped with the run — its null matched nothing, while the pipeline's
matched guide prevalence, the variable that actually drives the statistic. About five times as
many targets passed the correct test as the home-made one. Three report rebuilds followed. The method was one
directory away from files already open.

**Do not rank on counts of significant calls.** It is power × global shift. It correlated +0.46
with the number of pairs tested. Three separate axes died to this.

**Do not use per-group counts to measure specificity.** Counting hits per stratum measures that
stratum's calibration, not the target's biology. The decisive test: compute the condition
contrast separately within each of two independent strata — they correlated at **r = +0.00**,
and agreed in sign for slightly fewer targets than chance (about half). Use **effect-size contrasts**, and
require the contrast to reproduce on the other axis.

**Do not apply an expression filter on a pooled mean across an ordered series.** A
stage-specific transcription factor is, by construction, near-zero at most timepoints — so its
pooled mean falls under any sensible floor and the filter removes it for being stage-specific,
which is the thing that made it interesting. In one run the pooled filter flagged a
stage-specific transcription factor that was among the study's own most important regulators. Compute expression **per condition** and admit on the maximum, and where the run ships
an independent check that the perturbation worked — an autoregulatory-hit table, a per-guide
knockdown table — use it as an override rather than trusting the floor.

**Do not let a composite pass/fail `fillna()` a missing sub-check.** In the same run the
composite survival flag filled the per-guide column with 0, so every target with no gate pair
failed the whole kill-check set on a check that had never run for it. A missing sub-check is
missing, not failed — carry it as NA and say which checks a candidate actually cleared.

**Do not gate on knockdown efficiency; report it.** It correlates +0.08 with whether a downstream
signature reproduces. Gating discarded 11 genes that replicate strongly below 25% knockdown.
A gene can also have feedback maintaining its own transcript, or a small expression change can
have large downstream effects — assuming the statistics are calibrated, the downstream effect is
what matters.

**Judge a replication arm at the multiplicity of the pairs you are testing, not the full grid.**
One run called a pair "present in run A only" when the matched pair in run B failed a q computed
over *all* 145,560 of run B's tests, rather than over the 452 pairs actually being checked. The
committed table showed **399 of 399 trustworthy pairs agreed in sign** and the "A-only" q-values
clustered just past the cut — so "A-only" meant *below threshold in B*, not *contradicted by B*,
and the write-up over-declared non-replication in the direction of the manuscript's own thesis.
Report the sign-concordance headline alongside the threshold split.

**Whatever you exclude from the recovery count, exclude from every other scoring statistic.** A
sign-convention landmark is information *supplied to* the run, so it cannot validate the run. In
one report the headline concordance of ρ = +0.81 (n = 10) was carried by the two landmark genes —
the only two points spanning the sign contrast, and the only two excluded from every recovery
count. Without them: ρ = 0.62, p = 0.10, n = 8. "Ten of ten signs agree" had the same defect, two
of the ten having been given in advance. Quote both numbers, or drop the landmarks from the
statistic.

**A concordance computed on the same cells is not external validation.** Two factorizations of
one dataset, or two estimators on one matrix, give an estimator-concordance check. Call it that.
A report that elsewhere states "this is NOT replication in independent samples and must never be
called that" still captioned its headline panel "the strongest external validation this run
produced".

**A "no association" claim needs a p-value or a CI, like any other claim.** One report titled a
section "neither covariate predicts what a target moves" on the strength of two bare correlation
coefficients — while its own gate log carried +5.05 programs per decade of cell count at
p = 4e-05, and the recomputed rank correlation was p = 0.020. Breadth was load-bearing evidence
throughout that report, so part of what read as biology was cell count. Split the claim: one
covariate can be null while the other is not.

**When you change a verdict vocabulary, make the consumers fail loud on an unknown value.** A
figure filtered on `verdict != "neighbour not measured"` after that string had been renamed, so
the filter matched nothing and 457 *unadjudicable* pairs were plotted in the "cleared" series —
the exact error the rename was made to prevent. Validate against the known set and raise on
anything outside it.

**Do not fabricate a second condition.** If the run has one condition, cross-condition
replication is unavailable — say so and name the substitute (cross-factorization replication,
program-family coherence, neighbour support). Splitting programs, cells or guides after the fact
to manufacture "replication" is not replication.

**Do not report "significant in ≥N conditions" for an ordered time course without saying which.**
Early, late, and split are different claims.

**Do not apply a plain ">= N conditions" gate to an ordered time course.** It silently discards
**endpoint biology**. In a real run the parent manuscript's own headline result and its positive
control were significant at the final timepoint only — one target had 13 significant programs on
the last day and one the day before, and cleared no gate at all. Pass `--ordered` and admit
endpoint-only pairs on breadth.

**Do not require the same sign across an ordered series.** A sign flip over time is a temporal
switch, which is usually the most interesting thing in the run — not noise to be filtered. In
the same run, requiring same-sign dropped 27 pairs, including four members of one signalling
pathway on one program that are negative early and positive from the midpoint: the temporal
switch the study was designed to capture. Classify sign-flippers and keep them.

**Do not make a program-level claim without BH correction.** In one design (every program ×
every animal), 0 survived at FDR 0.05 *or* 0.10; the smallest p-value in the set was 0.0031, which is FDR 0.177. The study's
headline question turned out unanswerable at that design's power. Learn that during exploration, not in
draft 3.

**Do not claim an interaction from a difference in significance.** Six genes classed as
"abolished by stimulus" had no significant interaction statistic at all. If the claim is an
interaction, test the interaction.

**Do not compare effect ranks across arms with very different knockdown.** 38.6% vs 5.9% knockdown
at the same locus means the screen cannot adjudicate that locus, whatever the ranks say.

**Do not trust an `adj_pval` column that has not been checked against the NTC arm.** One run's
non-`_raw` column is rank-identical to the raw p-value but falls below it on 31% of rows and calls
2.07× as many pairs significant.

**Do not assign samples to groups using the signal you are about to test.** Matching animals to group
means by their usage profiles inflated the count of condition-different programs from 12 to 17. It is
circular.

**Normalise a projection before calling it a score.** An unnormalised projection onto an axis
ranks "does a lot" as "does the right thing" — four canonical large-effect genes topped a
target-ranking list and all four dropped out under a cosine.

**Do not orient a principal component by a weak correlation, and standardise before PCA.** In
one run a "fitness axis" was PC1 of unscaled per-program effects, with its sign flipped to agree
with a correlation of magnitude 0.14–0.23. The sign flipped on two of four conditions and not
the other two, so the resulting columns were not sign-comparable — and were then merged
side-by-side and compared. PC1 explained 0.11–0.20 against 0.07 for isotropic noise of the same
shape, and was dominated by a single program (|loading| 0.87 vs 0.22 for the next). If a
bipolar contrast is standing in for a magnitude, use the magnitude: a null correlation between
breadth and a *signed* PC is the expected result whatever the biology, and the same data gave
ρ = +0.27 to +0.41 against mean |effect|.

**The leak audit needs one authority.** In one run the set of answer-key genes reachable before
the freeze existed twice — computed by an audit script, and hand-declared in the scoring script —
and the two already disagreed on five genes. The hand copy was the one driving `leak_status`, and
therefore the earned-versus-contaminated recovery split that is the run's whole integrity claim.
Read the audit's output; do not retype it. If you must keep literals, assert they equal the
audit.

**Define the target classes once.** Two scripts in one run parsed target names by different rules
— one keyed on a `<locus>` prefix and `-and-`, the other on `^<locus>|Enhancer` — and both partitions
were harvested into the same numbers manifest. Fifty-nine `Enhancer-*` targets were inside
"plain gene symbol" in one table and outside "gene targets" in the other, so the manifest
certified two mutually inconsistent definitions of the same word.

**Do not declare a config file the source of truth and then hardcode the same list in the
scripts.** One run declared the technical-program list in `study.yaml` and re-typed it
identically in five scripts, none of which imported yaml. Editing the config changed nothing. If
`study.yaml` declares it, load it from there — in exactly one place.

**Do not put the predictor inside the response.** Breadth that counts the program being scored
reinstates the coupling the residual exists to remove.

---

## Biology

**Do not make a locus claim without the cis-neighbour silencing check.** CRISPRi at one promoter
silences neighbours. One withdrawn hypothesis rested on guides that silenced a neighbour whose
loss manufactures precisely the signature the hypothesis claimed. This check "settled five
disputes in one command."

**Do not rank program top-genes by loading z when you mean abundance.** Loading z is specificity.
Ranking by it surfaced sub-5-CPM genes and produced a completely wrong program identity that then
propagated through the report. **But raw TPM has the opposite failure** — mitochondrial, MALAT1
and unnamed ENSG transcripts dominate it, labelling 50 of 50 programs mitochondrial in one run.
Filter those, then rank by abundance.

**Do not write a hypothesis before the literature check.** "Published in 2016" is a cheap death,
and it is cheaper before the writing than after.

**Do not let two hypotheses in the same report contradict each other.** One draft argued "the
canonical gene wins at every other locus" while a second asked to overturn the canonical gene at
that same locus.

**Check the genome build before quoting a distance.** An hg19 coordinate reported as GRCh38 turned
13.7 kb into 27.9 kb.

**If you claim to match on a stable identifier, assert that the match actually fires.** One run's
Methods stated "neighbours are matched by Ensembl gene id rather than by symbol, so a renamed
neighbour cannot be recorded as unmeasured" — while the lookup key was a positional row index, so
the ID path never ran once and every match fell through to bare symbol. The committed
`neighbour_gene_id` column held row numbers; zero of 3,339 values began with `ENSG`. Because the
count matrix used `VARS`/`QARS` and the annotation used `VARS1`/`QARS1`, four candidates were
listed as their own cis neighbour at 0 bp and then marked never tested. Assert that a non-trivial
fraction of joins resolve on the identifier, and fail if not.

**Exclude every half of a dual-gene target from its own neighbour list.** For an `X-and-Y`
construct both genes are the intended perturbation, so finding Y beside X is the design. One run
excluded only the current anchor and called all five of its gated dual-gene candidates
"confounded in cis", inflating its published confounded count by 20%. The same construct has no
annotation entry under its full label, so anchoring on the whole string skips it silently —
anchor on each half.

**Resolve gene symbols against the library's own target names before writing "absent".** In one
run a screen's headline gene was listed under a superseded alias and a plain symbol match
reported it as never tested — it was in fact the largest effect in its class. In another, a
target existed only as a read-through (`X-Y`) symbol, and 17 more had been renamed between the
library's own two metadata versions.

**Do not collapse dual-gene `X-and-Y` targets onto either gene.** The perturbation hit both; a
hit is a claim about the pair. One library had 108 of them.

**Do not treat a control arm as a null without checking that it is one.** Two ways it fails: a
single pooled pseudo-target gives too few tests to calibrate anything (60 tests cannot separate a
5% from a 10% FPR), and a library selected for one phenotype moves its own controls — in one run
8 of 50 programs were significant in the control arm at the latest timepoint, coherently.

**Check what the effect column is before quoting magnitudes.** `log2FC`, `approx_log2FC` and
`beta_hat` are different quantities and a run may ship several. Where a post-hoc script produced
the column, read it: one such script's docstring, its own sanity-check cell, and its executed
line disagreed by a factor of (K/(K−1))².

**Check library/design annotations against their source.** A "condition-specific" rationale bucket
contained a gene specifically *not* in the condition-specific set; a "drug target" bucket contained a
gene that is not itself the drug target.

**Do not invent a mechanism to rationalise an edge.** If the path is not supported, say the gap is
a gap. Hypothesis edges are fine when clearly labelled as such.

---

## Writing and arithmetic

**Do not narrate a count you did not recompute.** One draft carried five independent miscounts in
one pass — "nine have no chemical matter" with four named and one double-counted; "two titles
contain novel" when it was four; 13 of 15 when it was 11. *The pattern matters more than any one
item.* Recompute every "N of M" from the table and register it.

**Do not quote a number whose generating script you cannot name.** The written rule:

> Before quoting a number in a report, find the script that emits it. If there is none, write one
> and let the new value supersede the old one in print — do not carry the old number forward with
> a caveat. State the estimator precisely enough that the definition itself is checkable ("Pearson
> r between vectors, each centred on its own mean"), because a vague definition is how two scripts
> compute "the same" number differently.

**Do not quote a figure's number from memory.** A caption said z = 4.2 where the heatmap showed
3.91, because two scripts defined "the same" gene set differently.

**Do not re-read a derived table under a rule it was not written under.** A committed correctness
fix that predated the table's last write silently invalidated every downstream number. Re-run the
chain; state the drift in the caveats.

**Do not invent a rationale for a methodological choice.** One draft justified a ranking statistic
with a confound it does not actually remove. State the real reason, or say the choice was
arbitrary and that nothing turns on it.

**Do not bake a temporary worktree's path into a committed entrypoint.** A blind run executed in
an isolated worktree on scratch storage wrote six SLURM runners that all `cd` to that worktree — so the
Regenerate block's first step fails for anyone else, and succeeds *wrongly* for whoever still has
the directory, running the scratch copy rather than the one they just cloned. Scratch storage is
purged, so it breaks for the author too on a 90-day horizon. Use
`cd "$(dirname "$(readlink -f "$0")")"`, and point `--output`/`--error` at a path built from an environment
variable (e.g. the cluster's scratch variable) rather than a literal path.

**A documented chain of `sbatch` calls needs `--dependency`, and the order must be right.** Six
runners fired back to back can interleave, so a later stage reads an earlier stage's stale or
absent output. In one run the figure job was also scheduled *before* the job that writes one
figure's input, and the figure script soft-skipped with exit 0 — producing a report with a broken
image and no error. Chain with `jid=$(sbatch --parsable ...)` / `--dependency=afterok:$jid`, and
make a missing expected input fail loud rather than skip.

**If you gitignore the rendered report, the render must actually work from a clean clone.** One
repo excluded its HTML as "regenerable from the .qmd plus the committed tables" while Quarto was
in neither `pixi.toml` nor any repo mise config, and the renderer lived in the author's
`~/.claude`. The deliverable was therefore obtainable by exactly one person. Either pin the
renderer in the environment, or commit the HTML.

**Assert the font, do not let it fall back.** A figure module's docstring said "Arial, applied
rather than cited"; the committed PDFs embedded DejaVu Sans, because the compute environment had
no Arial and matplotlib substituted silently. Re-running the same script elsewhere gives visually
different panels. `assert "Arial" in {f.name for f in matplotlib.font_manager.fontManager.ttflist}`.

**Do not hand-copy a provenance string.** A stated `pixi.lock` fingerprint was wrong. Compute and
register it.

**Do not filter tool output with a pattern that can match the data.** `grep -v OMP` ate a gene
called POMP out of an indexed listing. Suppress warnings at the source or anchor the filter. If an
indexed listing has a gap, suspect the filter before the library.

---

## Figures

Nine of the 24 Majors in the nine-reviewer pass were figure-facing — more than any other category,
and five were legend-vs-plot contradictions:

- A legend citing a gene set that is not in the figure.
- A legend quoting held-out numbers while the plot shows leave-one-out.
- A title restating an odds ratio of 2.64 as "2.6× enriched" when the bars show 1.96×.
- Plotting two items the legend says were excluded, changing the n behind a reported null.
- Colouring a gene as unassigned in one figure and as the top hit of a class in another.
- An answer line naming the strongest regulator when a different bar is visibly longer.

Open the rendered figure and read it against its own legend. `/engreitzlab-review`'s
`figure-visual` reviewer does this; do not rely on it as the only pass.

---

## Process

**Do not let the analysis-level doc go stale.** Three rebuilds happened and the entry-point
`UPPER_SNAKE_CASE.md` still described the first one. Nothing carried it forward, and that alone is
much of why the work felt lost.

**Run the reviewers yourself before showing anyone.** Verbatim: *"remember to run the reviewer
yourself (and update the reviewer as needed with my feedback) before showing me another version"*
and *"Run all of the reviewers or create new ones, I'm tired of giving you the same feedback over
and over."*

**When feedback arrives, fix the general process, not the instance.** Verbatim:

> When I am giving you feedback, I don't want you to directly bake that knowledge into the skill
> (e.g., "always put <kinase> underneath <complex>"). I want you to learn from the feedback and make
> adjustments to the overall skill to improve the result not only on [this program] but all future projects.

So: a correction about one gene becomes a check, not a hard-coded exception. Update this file and
the relevant reviewer brief, not a lookup table.

---

## The v1 loop — what the first blind run taught (2026-09-23)

Every item here shipped in the v1 report and is now enforced in code or in a brief.

- **A literature verdict attached to an unchanged claim.** "`reframe` — cannot stand as a
  protein-level claim" printed beside the claim it said could not stand. Now the Reviser must act,
  and the validator rejects a round without an action.
- **The process as the headline.** The title and summary reported a blind-scoring result and a
  filtering funnel; the ranked list started 363 lines in. Now the lint fails a report whose first
  sections are not the top picks and the ranked table, or that uses process vocabulary.
- **Self-coined terms used as if defined.** "<gene> runs against the <X> model", where the model
  was never stated. Now the Critic checks define-before-use.
- **Observations as hypotheses.** "Loss of <core complex> moves sixteen programs." Now the
  Generator brief requires a mechanism and predictions.
- **Core machinery crowding the list.** Entries for ubiquitous complexes (protein degradation,
  chromatin silencing, RNA decay, generic stress responses) took slots from cell-type biology. Now
  they are positive controls unless there is a cell-type-specific angle.
- **No figures for any hypothesis.** Now each card needs a headline figure from a data trace.
- **Blindness by instruction.** The skill text, task brief, git history and sibling analyses all
  carried the answers. Now a sterile directory and a held-out-terms check are required.
- **A hand-typed final list.** `SHORTLIST = [...]` in a selector script. Now the cards are the
  ledger's `supported` entries in rank order, checked by the claim-traceability reviewer.
