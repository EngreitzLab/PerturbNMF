# Report structure and per-hypothesis templates

Two linked reports. This was asked for explicitly:

> Not one report. **An index report with the ranked table, where every hypothesis title is a
> hyperlink into a second, deeper report** that carries the per-hypothesis research dive.

Both are `/analysis-report` Quarto documents, so the YAML, the `<details>` legends, the lint gate
and the `numbers.json` traceability all come from there.

---

## 1. The index report — `<study>_hypothesis_index.qmd`

```
## Headline figure               one graphical abstract + full <details> legend
## Summary                       3–5 cross-cutting bullets; nothing a single table already says
## Goals                         the study's questions, 2–4 sentences, from study.yaml
## Procedure                     ← do not omit. See §1.1
## Datasets                      every input: assay, system, source/citation, access date
## <finding-titled sections>     Q:/A: slides for the cross-cutting results
## The ranked shortlist          one table per evidence class. See §1.2
## What did not make it          one row per gene in the starting set. See §1.3
## Top picks for a real experiment   3, re-ranked. See §1.4
## Limitations
## Open questions / next steps   [Q for <name>] where it is aimed at a person
## Methods → Regenerate          env + pixi.lock hash + exact commands
```

### 1.1 Procedure

Asked for directly: *"State the procedure up front — 'how would you actually go and select drug
targets' deserves an explicit method section: the lanes, why they are separate, the gates, and
what each evidence type can and cannot decide."*

Four sub-parts:

1. **The evidence classes, and why they are separate** rather than one ranked list.
2. **What each evidence type can and cannot decide**, as a table. The screen decides direction and
   magnitude in this system. Human genetics decides relevance to the human trait. Neither decides
   the other.
3. **The gates**, as a funnel table with survivor counts at each step.
4. **The sign convention**, stated once, verbatim from `study.yaml`.

### 1.2 Candidate tables

One per evidence class. Before each, a bolded one-sentence funnel and a paragraph naming the sort
key:

> **Selection rule.** Of the 163 genes clearing the CRT gate, the 26 that (i) clear the
> cis-neighbour check, (ii) do not merely restate the library's inclusion rule, and (iii) fall
> into one of the six evidence classes.
>
> **Ordered by evidence class, then by effect size** — so the table reads as "what kind of
> hypothesis is this", not "what scored highest". `KD %` is reported as a covariate and is not a
> filter. *Prior studies* is the literature verdict, with the strongest citation linked.

Reference column set, adapt per study:

| Gene | Hypothesis | Programs moved (n; conditions) | max &#124;effect&#124; | KD % | Specificity | Class / locus | Prior studies |

- Every gene name is a markdown link into the deep dive: `[Leng1](..._deep_dive.html#leng1)`.
- `Hypothesis` is the one-sentence falsifiable claim, identical to the deep-dive slot.
- `Specificity` is **plain English**: `F>M`, `young>old`, `similar`, `flips`, not a coefficient.
- Wide tables need the `page-layout: full` + grid YAML and a `display:block; overflow-x:auto`
  style block, or the last column paints over the TOC.

### 1.3 What did not make it

One row per gene in the starting set: `gene | expressed? | moved a program? | cleared
cis-neighbour? | literature verdict | in shortlist? | why not`. Collapse under `<details>` if
long, but include it.

### 1.4 Top picks for a real experiment

Three, **deliberately breaking the table order**, re-ranked on experimental criteria rather than
evidence strength, with a named runner-up. The criteria, decoded from the best prior version:

cost of the experiment · size of the novelty gap · tractability/druggability · crispness of the
prediction · symmetric falsifiability ("either confirms or kills it") · named genes rather than
pathways.

```markdown
## Top picks for a real experiment

1. **N1 (two uncharacterized cluster members)** — cheapest discovery bet; the method is already
   validated by a third member the literature later confirmed, and these two have no literature
   in this pathway at all. High novelty, low cost, clear readout.
2. **N2 (a druggable kinase gating the top program)** — druggable node, direct disease
   relevance, grounded mechanism; the cross-program breadth is the new claim. A clean
   small-molecule experiment.
3. **N6 (p21–proteasome hinge)** — the crispest mechanistic prediction (single-gene epistasis
   across 6 programs); a straightforward co-KD / bortezomib rescue either confirms or kills it.

_N4 (USP8 → IFN tone) is a strong runner-up — grounded, testable, and it makes H3 concrete._
```

An unexplained sign paradox is **not** promoted here. Keep it in Open questions.

---

## 2. The deep dive — `<study>_hypothesis_deep_dive.qmd`

Sections grouped under an `#` per evidence class, `##` per hypothesis.

**The slot order is fixed and not negotiable.** It had to be asked for twice and needed a
dedicated rewriter script to retrofit:

> in deep dive, reorder the section. start with the hypothesis, and add a sentence explaining the
> novelty/biological rationale for more deeply investigating this hypothesis/gene

```markdown
## <Gene, program, or group>

**Hypothesis.** <one falsifiable sentence. A claim, not a topic.>

**Why investigate.** <one sentence: the novelty and the biological rationale.>

![](plots/<...>.png)

- **Screen.** <programs moved, signed effects with numbers, direction described in words.>
- **Specificity.** <plain English first: "stronger effect in males than females (1.4×)"; then the
  statistic and whether it reproduces on the other axis.>
- **Programs.** <what the moved programs are, top-loading genes, and which annotation source said
  so.>
- **Neighbours.** <top ± perturbation neighbours, and what they place this gene near.>
- **Prior studies.** <the literature verdict, with hyperlinked PMIDs/DOIs. What is already known,
  and what the screen adds on top.>
- **On target.** <cis-neighbour result: which neighbours were measured and how far they moved.>
- **Test.** <named reagent, named readout.> <Observation> = supported.
- **Confidence.** **<label>** — <which half is grounded and which is not.>
```

### 2.1 The `Test` slot

One sentence plus an explicit success clause. The clause is what makes it falsifiable, and it is
the template's hardest-working device:

> `Phenocopy = real member.` · `Dose-dependent anticorrelation = supported.` ·
> `Epistasis (p21 loss reverses the proteasome effect) = supported.` ·
> `Coordinated collapse = supported.` · `Opposite co-movement = supported.`

Name the reagent (`SB431542`, `bortezomib`, `TEER/permeability`, long-read RNA-seq) and say what
result would kill it. Where an analysis on data already in hand could answer it, say that, and
check first that the assay can support it.

### 2.2 The `Confidence` slot — closed vocabulary

Define once in the preamble; never invent a sixth label; never use the words "honest" or
"genuine".

| Label | Meaning |
|---|---|
| `correlation-only` | the screen shows co-movement; no independent mechanism |
| `partly grounded` | one arm has literature support, the other is correlation-derived |
| `grounded` | outside literature backs the mechanism |
| `grounded and clean` | grounded, **and** the prediction is a single crisp test |
| `dataset-property` | the claim is about the screen itself (a module collapsing, a timing skew, a negative) and there is no external literature to ground it against |

**`grounded` means the literature BACKS it, not that it is absent from it.** A real report
glossed `grounded and clean` as *"the mechanism is outside literature and the prediction is one
experiment"* — which reads as *absent from* the literature, the opposite of the definition — and
then applied it to four hypotheses whose own Stage-4 record said "Strongest citation: none
applicable". `grounded` was never used at all, so a four-level vocabulary silently became three,
with the top label covering both the best-supported controls and the claims with no prior art to
check. If there is nothing to ground against, the label is `dataset-property`, not the top one.

Always followed by a clause saying **which half** is which, and naming the artifact class that
could kill it plus the check to run:

> **Confidence.** **partly grounded** — the TGF-β/EndMT arm is established; the restraint arm is
> correlation-derived and could be a mirror artifact. Check mirror-fraction before believing it.

---

## 3. Two devices used throughout both reports

### 3.1 Reviewer option

Every judgement call becomes an auditable blockquote naming the alternative and why it lost.
Asked for directly (*"didn't give me the reviewer options"*), and it caught three of the 24 Majors
in the CAD review — including one where the stated rationale was itself invented.

```markdown
> **Reviewer option — rank on the shrunken coefficient, not on *t*.** The reason is the absolute
> scale, not a confound: b̂/τ is interpretable in τ units (and hence in units of a landmark
> target's effect) where *t* is not. It is **not** true that this removes an n_treated confound, as an earlier draft claimed —
> Spearman(n_treated, *t*) = +0.07 against +0.22 for b̂. The two orderings agree at ρ = 0.98, so no
> candidate ordering turns on the choice.
```

Rules: state the alternative concretely; say what would change if it were chosen; if the answer is
"nothing", say that too. If a previous draft's rationale was wrong, correct it here — this is the
one place a correction belongs in the body.

### 3.2 Worked examples (prose, longer form)

For the 3–5 candidates carrying the most weight, a prose entry alongside the table row:

```markdown
### TGFBR1 — the clearest hypothesis this class adds

*Internal.* <the screen evidence, with numbers, and what corroborates it inside the screen —
paralogues, complex partners, neighbours.>

*Why this is a result and not a tautology.* <only when a circularity objection exists — e.g. the
program is scored against a gene set that contains the gene's own pathway. Say what independent
annotation source rescues it.>

*External and caveat.* <literature, clinical precedent, and the honest liability.>
```

Include one **deliberate negative-control entry** — a candidate that looks good on the headline
metric and fails the class's own test. It is the cheapest way to show the filter works.

---

## 4. Preamble block, both reports

```markdown
_Dataset: <assay, system, K, conditions>. Programs are cNMF factors; <N> targets.
Sign convention: positive effect = **brake** (knockdown raises the program), negative = **driver**
(knockdown lowers it). Citations are hyperlinked PMIDs/DOIs inline.
"Correlation-only" means the screen shows co-movement with no independent mechanism; "grounded"
means outside literature backs the mechanism._
```

The sign convention line is load-bearing. It is most of why the best prior report is readable by
someone who has never seen the screen.
