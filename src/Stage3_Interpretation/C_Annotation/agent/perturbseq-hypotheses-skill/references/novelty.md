# Novelty — what counts, and how to check it

Novelty is the axis that goes wrong most often, and **the database verdict is unreliable in both
directions.** Absence from OmniPath does not make something new. Presence in OmniPath does not
make it old.

---

## 1. The database is a prior, not a verdict

A knowledge-base lookup is a cheap first pass over a large candidate list. It is never the claim,
and it never settles a call on its own. Both error directions are real and both have shipped:

**Absent from the database, but not new.** Interaction databases are incomplete and biased toward
well-curated signalling. A gene can be absent simply because nobody curated it. The correction
that established this, verbatim, after a first pass leaned on the database alone:

> in order for it to be a novel hypothesis, it needs to "not be in OmniPath" but also not reported
> in literature. the first 5-ish are all expected based on literature that may not be fully
> represented in OmniPath. Do some more filtering based on this, and more reading/LLM queries, and
> come up with some actually interesting+surprising+novel hypotheses

**Present in the database, but still new.** This is the direction that quietly loses real
findings, because a `CONFIRMED` path result looks like a reason to stop looking. An edge can be in
OmniPath and the hypothesis still be novel when:

- The curated edge comes from **a different cell type or tissue**, and nobody has shown it here.
  OmniPath is largely cell-type agnostic; your screen is not.
- The curated edge is **generic** (a kinase phosphorylates a substrate) while the claim is
  **specific** (that kinase sets the level of *this* program in *this* state).
- The **direction or sign** is the new part. The database says A acts on B; the screen says which
  way, and under which condition, and that the sign flips between programs.
- The **scale** is the new part — the gene is the top regulator of many programs at once, which
  no pairwise edge in a database can say.
- The edge exists but rests on **one low-quality source** (a single high-throughput screen, a
  propagated GO term from one unquantified mass-spec detection). Check what the edge is actually
  built on before letting it kill a hypothesis.
- The **combination** is new even where every edge is old: a branch point, a reciprocal pair, a
  shared hub. The parts are curated; the wiring is not.

So: run the database pass to triage, then **make the call from the literature, in this cell type,
for this specific claim.** A `CONFIRMED` path result demotes a candidate's priority; it does not
remove it, and a demoted candidate with a strong cell-type-specific claim goes back in.

## 1b. A gene in a results table is an observation, not a published hypothesis

A prior screen listing "gene X moves program Y" in a supplementary table, or even in a figure,
does **not** make a hypothesis about X non-novel. Novelty is judged on the **hypothesis** — the
mechanism, and the experiment that would test it — not on whether the observation has been
tabulated before. Kill or demote for prior art only when the mechanism itself has been proposed
and tested (a functional follow-up) in this cell type. Otherwise keep the hypothesis, and cite
the prior observation in "What is already known" as support ("also listed as a hit in …").
(Standing ruling from Jesse, 2026-09-24.)

## 2. The bar is a conjunction

> "New" means all of: absent from the signaling knowledge base; no prior report on a targeted
> literature search; surprising (the gene's known function does not predict it); and
> mechanistically plausible.

Read with §1: the first clause is the weakest of the four and the only computed one. Where the
database and the literature disagree, **the literature wins**, and where the literature is silent
on this cell type, the database's opinion about other cell types does not substitute.

---

## 3. The computed part — signed-path reconciliation

For a program, take its significant TF regulators that are also program members (top ~15% by cNMF
loading) as **effectors**; drop generic/basal TFs. For every other significant regulator, search
the interaction graph (OmniPath) for a **directed, signed** path to an effector. The path's net
sign — the product of stimulation `+1` / inhibition `−1` edges — times the effector's own role
must match the regulator's observed driver/brake role.

| Path result | Meaning |
|---|---|
| `CONFIRMED` | a signed path exists whose predicted role matches the observed one |
| `CONTRADICTION` | signed paths exist but every one predicts the opposite role |
| `PATH_UNSIGNED` | a directed path exists, but no fully-signed one |
| `NOVEL` | no directed path to any effector |

These four labels are **triage input for the Literature agent and the Reviser**, recorded in the
Literature agent's notes. They are not the literature verdict, they never appear in the report,
and the report carries no visual flag for them.

This design deliberately demotes essential-gene pleiotropy: a spliceosome or proteasome subunit
lands `NOVEL` because it has no coherent signalling path to the program's TFs, not because it is
a discovery. Hence the standing caveat, verbatim from the code that used it:

> novelty rests on the WebSearch check, not the verdict

Reference implementation (optional): an `examples/` directory, where present, holds one. It is
absent from sterile runs; the procedure above is complete without it.

A `CONTRADICTION` is worth passing on explicitly. The Literature agent checks whether the
curated sign rests on solid evidence in a comparable cell type; the Reviser then decides whether
the screen sign or the literature sign is more likely wrong and acts (sharpen, reframe or kill).
If the candidate survives, its card says in plain words which way the screen points and what the
literature reports, under "What is already known".

---

## 4. The two QC filters every novel call must pass

Both are cheap, both come from a direct instruction, and together they removed 18 of ~33
candidates in the run that introduced them:

> Review your hypotheses and remove those where either of the 2 criteria raise suspicions

1. **Expression in this system.** A gene below the floor and detected in very few cells is not
   expressed, so the "off-class discovery" is an artifact. Compute it from the screen's own count
   matrix; never infer non-expression from absence in a targeted panel. **State the unit in the
   constant's name** — CPM and CP10K differ by 100×, and one run implemented a documented
   `0.5 CPM` floor as `0.5 CP10K`, moving 70 of 192 candidates' on-target verdicts.
2. **CRISPRi cis-confound.** A bidirectional or near-overlapping promoter whose neighbour is
   expressed is a more likely source of the signal. Measure the neighbours as responses
   (`scripts/check_cis_neighbours.py`). This has manufactured exactly the signature a hypothesis
   rested on, more than once.

Add a third where per-guide data exists: **guide-vs-guide concordance.** A signal carried by one
guide is a guide effect until shown otherwise.

---

## 5. Run blind. Do not pre-load biology.

**No known pathway, gene list, or mechanism is handed to the skill as an input.** The point of
the method is to find out what it recovers *without* prior information. Handing it "these three
genes are the known axis" tells you nothing about whether it could have found them.

So `study.yaml` carries **no `prior_art` gene list.** Prior art is something the skill
**derives**, through the Literature agent, from literature, databases and its own searches — and that derivation is
part of what is being evaluated. If the skill cannot rediscover a well-established axis from the
literature, that is a finding about the skill. Do not paper over it by supplying the answer.

### What still belongs in `study.yaml`

The line is **design facts in, biological conclusions out.**

| In — properties of the experiment | Out — conclusions about biology |
|---|---|
| what the library targets, and **how those targets were chosen** | which genes or pathways are known to matter |
| the conditions, timepoints, cell line | which programs are protective vs risk |
| the sign convention, verified against the data | a curated list of "expected" regulators |
| preprocessing state and known artifacts | a predecessor paper's gene lists |

The selection criterion is the one that looks like biology but is not. If 240 of 300 targets
were picked because they scored in a prior differentiation screen, then "gene X affects
differentiation" is the *inclusion rule*, not a discovery — and a report that presents it as one
is circular. Record the selection criterion, and bar claims that restate it. That is a statement
about the experiment, not about the literature.

`novelty_embargo` survives on the same logic: it names topics where the *field* is contested, so
a novelty claim would be indefensible whatever the screen shows. The form of the instruction that
introduced it: *it is debated whether <drug class> acts through <this cell type> or through <another
tissue>, so bar that topic from novelty claims.* That is a constraint on what can be claimed, not a
hint about what is true.

### The answer key

Scoring against known answers is a separate step in a separate session, after the report is
frozen. See `blind-evaluation.md`. It never appears in the report.

---

## 6. The adversarial pass

The Literature agent's brief is in `briefs/literature.md`. It returns a verdict from a closed set,
`known_in_this_cell_type / known_elsewhere / novel / contradicted`, plus citations with what each
one shows. **The verdict is input to the Reviser, not a label for the report.** The Reviser must
act on it (`briefs/reviser.md`), and the report carries only the resulting claim.

### 6.1 Let it demote our own favourites

In the best prior report, two of six headline hypotheses were downgraded — one to "textbook
pharmacology, not novel", one because the flagship uncharacterized gene had been characterized
the previous year. That is the pass working.

In another project, two hypotheses died on "already published in 2016" *after* they had been
written. That is the pass being skipped.

### 6.2 When the literature explains it, find the remainder or record a positive control

Where the mechanism is not new, the Reviser locates precisely what *is*: usually the **cell-type
context**, the **sign**, or the **scale** (one regulator moving many programs). That remainder
becomes the claim. If there is no remainder, the screen re-deriving known biology from
perturbation data alone is a positive control. It goes in `explore/positive_controls.tsv` and
gets one sentence in Methods ("the screen re-derives N known relationships, listed in the
appendix"), not a card in the top N.

### 6.3 Mark corrections to the team's beliefs; do not silently apply them

Where the pass overturns something the team already believes, say so visibly and say you did not
edit the source:

> **I did not edit the overview** — these are proposed changes for you to accept or reject.

This applies to beliefs recorded in `study.yaml` or a team document, never to the run's own
hypotheses, which the Reviser rewrites directly.

### 6.4 The rejects go in the appendix

Killed candidates, with a one-line reason and the key reference, go in the killed-hypotheses
appendix. Not in the main report. A loop that never kills anything attractive is not filtering;
the validator's iteration-health check flags it.

---

## 7. Citations

- Inline and hyperlinked at the claim: `(Itoh et al., *Genes Dev* 1999,
  [PMID 9887101](https://pubmed.ncbi.nlm.nih.gov/9887101/))`.
- **Verify every one before it ships** — `scripts/verify_citations.py` checks DOIs against
  Crossref, PMIDs against E-utilities, and NCT numbers against the ClinicalTrials.gov v2 API, and
  compares title / first author / journal / year. In one prior audit 7 of 16 PMIDs were defective;
  in the report that ran this check, 72 of 72 passed.
- The Generator and the Literature agent query the literature directly — verification by a downstream reviewer is a
  backstop, not the primary mechanism.
- Check the genome build before quoting any genomic distance. An hg19 coordinate quoted as GRCh38
  turned 13.7 kb into 27.9 kb in a shipped draft.
