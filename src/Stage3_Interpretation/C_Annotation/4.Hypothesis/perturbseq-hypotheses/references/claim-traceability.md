# Sub-agent: claim-traceability

Read this entire file before making findings. Output contract:

- `findings`: one per issue, each with `severity` (`major | minor`), `file`, `line`, `category`
  (the section number below), `summary` (one line), `evidence` (1–5 line verbatim snippet),
  `why_it_matters`, `suggested_fix`, `confidence` (`low | medium | high`).
- `not_flagged`: things you checked and found fine, one line each.
- `decisions`: judgement calls you made and why.

(Optional reference, if present: `mycelium-upstream/skills/core/references/review/README.md`.
It is not vendored in sterile runs; the contract above is sufficient.)

You are reviewing a **hypothesis report from a Perturb-seq screen** for one thing: **can every
factual assertion in the prose be traced to something that produced it?** You are not judging
whether the science is interesting or the statistics are well-chosen — other reviewers have those.

This remit exists because the dominant failure in three prior review cycles was not bad reasoning.
It was stale numbers, narrated arithmetic, and citations nobody checked.

You have shell access and web access. **Actually run the checks.** A traceability review done by
reading is worthless.

---

## 1. Every number traces to a generating script

Walk the prose, the ranked table, the cards and the figure legends. For each claim-shaped number
(`n =`, `p =`, `FDR <`, `r =`, a percentage, an `N-fold`, an effect size, a count):

1. Does it match the `summary` or `stats.json` of a test under `traces/<id>/<test>/`, or the
   report's `numbers.json` / `.manifest.json` (mycelium `register_value`) if there is one?
2. If not, can you name the committed script and the line that emits it?

- **Major** — the number appears in no manifest and no script emits it. The standing rule:
  *"a report number with no generating script is a number you cannot defend."*
- **Major** — the value in the prose disagrees with the value in the manifest or the CSV.
- **Minor** — the number is real but unregistered.
- **Major** — the manifest is so dense that membership means nothing. Measure it: report the
  entry count, the *distinct value* count, and what fraction of small integers and one-decimal
  values it covers. One run's manifest expanded every harvested value into `round0/1/2/3`, `abs`
  and `percent` variants, reaching 17,968 entries over 4,970 distinct values — covering 48% of
  all integers 0–300 and 54% of one-decimal values 0–100. A fabricated number had roughly a
  coin-flip chance of matching something, so the linter's green light carried almost no
  information. Register the exact value and the single rounding the prose uses, and match by
  name rather than by scanning the value set.
- **Minor** — the estimator is described too vaguely to check ("the correlation between them").
  A definition like "Pearson r between 150-dimension effect vectors, each centred on its own mean"
  is checkable; "the correlation" is not — and a vague definition is how two scripts compute "the
  same" number differently.

## 2. Staleness

- Compare the mtime of each `explore/*.tsv` and `traces/**/*.csv` / `stats.json` against the
  mtime of the script that writes it, and against the last commit that changed that script. **Major** if a derived table predates a
  committed change to its generator — every downstream number is then pre-fix.
- Compare figure mtimes against the scripts that make them and the prose that describes them.
  **Minor** if a figure is older than the section around it. (Files whose path contains
  `schematic`, `cartoon` or `concept` are hand-authored; exempt.)
- **Major** if the report quotes a program index, a K, or a threshold from a run other than the
  one named in Methods. Program numbering is not stable across runs of the same decomposition.

## 3. Narrated arithmetic — recompute it

Every "N of M", "all five", "three of the four", "a quarter of", "the only gene that". **Do not
read these — recompute them from the table the report itself supplies.**

- **Major** for any count that does not reproduce.
- Report the whole set in one finding if several fail: the pattern matters more than any one item.
  One prior draft carried five independent miscounts in a single pass.

Also check: does a stated denominator match the code's denominator? One draft's prose said 57 and
the code said 56, and the missing gene had never been triaged.

## 4. Citations resolve, and say what the report says they say

Run `scripts/verify_citations.py` (or the equivalent) over every citation in the report.

- DOIs → Crossref. PMIDs → NCBI E-utilities. NCT numbers → ClinicalTrials.gov v2.
- Compare title, first author, journal and year.
- **Major** for any identifier that does not resolve, or resolves to a different paper.
- **Major** where the cited paper does not support the claim it is attached to — spot-check the
  abstract for the 5 load-bearing citations at minimum.
- **Major** for a reagent claim that is factually wrong in kind: a blocking antibody named for an
  intracellular protein; a probe described as a selective inhibitor of the wrong target; a
  compound attributed to the wrong enzyme arm. These have all shipped.
- **Minor** if a citation exists only in a reference list and not inline at the claim.

## 5. Provenance strings

- **Major** if the `Regenerate` block's environment fingerprint (e.g. `pixi.lock` sha256 prefix)
  does not match the repo. Compute it: `shasum -a 256 pixi.lock | cut -c1-12`. Do not accept a
  hand-copied value; it has been wrong.
- **Major** if a command in `Regenerate` cannot run as written (missing binary, wrong path,
  a tool not on PATH and not in the environment). Try them.
- **Minor** if a committed input sits inside `explore/` or `traces/` next to regenerable
  artifacts — an `rm -rf traces/` that destroys an input is a real hazard. Inputs belong in
  `inputs/MANIFEST.txt`.
- **Minor** if a dataset named in Methods lacks its assay/system, source or citation, or access
  date.

## 6. Table ↔ prose ↔ figure agreement

- Pick every value the prose quotes from a table and check it against the table.
- Pick every value a legend quotes and check it against the data the figure was plotted from —
  not against the prose. **Major** on disagreement. A caption said `z = 4.2` where the heatmap
  showed 3.91, because two scripts defined the same gene set differently.
- **Major** where a figure's own on-plot labels are hardcoded while its marks come from data.
  One bar chart drew its bar widths from JSON and typed the step labels as strings, so a rerun that
  changed the counts would produce a figure whose labels contradicted its own bars. Same defect
  in a panel title hardcoding `ρ = −0.05` against a computed −0.052. The labels must be built
  from the same values as the marks.
- **Major** where a legend describes an ordering or encoding the code does not implement. One
  legend said rows were ordered by "net signed effect" while the key was
  `np.sign(x) * |x|` — i.e. `|x|` — putting the *most negative* gene in row 1 as if it were the
  most positive. Another claimed "box width scales with the survivor count" where the widths
  spanned 0.508–0.553 in, a 4% difference no reader can see.
- **Major** where a legend describes a panel, series, or exclusion that is not what the figure
  shows.

## 7. A headline number must come from the evidence the claim rests on

**Major** where a summary column or headline figure is computed over a wider set than the claim.
One ranked table's `max |β|` was the largest coefficient over *all* programs and *all*
conditions, significant or not — so a hypothesis whose claim was explicitly "the regulator moves
no program in two or more conditions" carried a headline number drawn from a single-condition
effect on a different program, which the report elsewhere says the significance filter
discards. Restrict the number to the pairs the
hypothesis actually cites, or rename the column to say what it really spans, and define it in
the column legend either way.

## 8. Verdict strings are printed verbatim

**Major** where a report renders a check's outcome through a branch that collapses distinct
verdicts into one word. In one run only 5 of 71 genes carried the stored verdict `clean`, but
the generator's `else` branch printed "clean" for every gene that was not `CONFOUNDED` — so
`watch`, `distal` and `underpowered` all reached the reader as clean, including a gene whose own
line read "clean … neighbour at −0.47". That directly contradicted the report's own Caveats.
Print `str(verdict)`, or enumerate every branch; never let `else` mean "fine".

## 9. Self-consistency of the candidate set

- The report is one file: top picks, ranked table, cards. Every card appears in the ranked
  table, and vice versa. **Major** on a mismatch.
- Every link from the ranked table resolves to a card anchor, and every `[trace]` link resolves
  to a directory under `traces/`. **Major** on a dead link.
- Every number on a card matches the `summary` or `stats.json` of the test it cites in the
  ledger. **Major** on a mismatch.
- The cards equal the ledger's `supported` entries, in `rank` order. The killed appendix plus the
  cards account for every ledger entry. **Minor** if they do not add up.

---

## Output note

Group findings by class (unregistered numbers, stale artifacts, failed recomputations, citation
failures, provenance) rather than listing them gene by gene. A single finding that says "these
seven counts do not reproduce, here they are" is more actionable than seven findings.
