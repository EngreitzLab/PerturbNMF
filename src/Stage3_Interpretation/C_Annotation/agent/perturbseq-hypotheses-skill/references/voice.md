# Voice — how to write a hypothesis report

Read this before writing a sentence. Three of the recorded corrections on prior versions of this
report were about writing, and one of them required rewriting two whole reports after delivery.

The reader is **a scientist in the lab or a collaborating lab who was not in the analysis
session.** They know the field. They do not know this project.

---

## 1. The stack

1. **`~/.claude/skills/analysis-report/lander_voice.md` — all five rules, applied to every
   sentence.** Not cited: applied. Lead with the result. Put the key number in the sentence.
   Define a term at first use and never coin a private one. Plain words, active voice, one idea
   per sentence. Numbers and technical terms stay formal.
2. **`analysis-report`'s stand-alone and worked-examples mandates.** Name and cite every input
   dataset concretely. Ground every aggregate in named individual instances.
3. Audience **Tier B** — adjacent-field collaborator — unless `study.yaml` says otherwise.

### Do not use `/grug-voice` for these reports

It was tried and rejected:

> No, don't use /grug-voice. You don't need to replace "Perturb-seq" with "knock-down". You
> overdid it. It just needs to be interpretable to other scientists in my lab and collaborating
> labs. For example, instead of saying "i took the Perturb-seq dataset and compared it to the AF
> lists", you should explain that you're talking about "<cell type> genome-wide Perturb-seq (in
> <cell line>) and compared it to lists of protein-protein interactions predicted with AlphaFold 2
> (from papers X, Y, and Z)".

---

## 2. The rule that resolves the apparent contradiction

`lander_voice.md` says *"Not a general audience — a peer."* The standing correction says the
report must be *"interpretable to OUTSIDE HUMAN READER/COLLABORATOR."* These are not in conflict.
Conflating them produced bad drafts in both directions: one jargon-dense, one dumbed down.

> **The fix is naming things concretely, not simplifying the vocabulary.**

Keep every domain term a working scientist knows — Perturb-seq, CRISPRi, cell-type names, cNMF
program, log2FC. Replace every **internal** reference with what it actually is, its
assay/system, and its source.

### Corollary A — do not lay-explain the biology; do lay-explain the statistics

A collaborator knows what a cNMF program is. They do not know what *your* correlation coefficient
means, and they should not have to. Every specificity, interaction, reproducibility or contrast
statistic is stated as **direction and magnitude in plain words first**, with the number in
support.

### Corollary B — no LLM-internal terminology

The recorded example of what this sounds like when it goes wrong, verbatim:

> not evidence of direct contact — not a sign-rationalization, this pair has an actual mined
> correlation edge

Words like *rationalization*, *mined edge*, *verdict object*, *contract*, *proposer/refuter* are
internal machinery. They do not appear in the report.

---

## 3. The translation table

| Do not write | Write |
|---|---|
| "timepoint-shared (r = −0.09, at the null); stimulus-specific, consistent across timepoints (r = +0.36)" | "Stronger effect in stimulated than resting cells (1.7×). No clear difference between early and late timepoints." |
| "I compared the Perturb-seq data to the AF lists" | "genome-wide CRISPRi Perturb-seq in <cell line>, compared against AlphaFold2-predicted protein–protein interactions from <named source, with citation>" |
| "significantly lower" | "lower (median 0.74 vs 0.89, p = 2×10⁻³¹)" |
| "the <X> axis", "dual-benefit", "the <Y>-leak axis" | the described term, reused verbatim each time |
| "P12 responds strongly" | "the hypoxia-response program rises on VHL knockdown (+5.3 SD)" (glossary title, never the number) |
| "the effect replicates" | "the same direction in both stimulated replicates (14/15 programs)" |
| "this gene is interesting" | the specific reason: what it does, what moved, what is unknown |

**Every statistic gets a plain-English gloss.** If you cannot say in words which group is stronger
and by how much, you do not yet understand the statistic well enough to report it.

---

## 4. Banned outright

- **Mannered prose.** *"Do not use mannered prose."*
- **Narrating your own process.** *"Do not explain your cycles of thinking. Just accurately report
  the final results."* *"I don't need a huge log of your decision-making, unresolved blah blah,
  history of edges added/removed."* State the final claim, never how it was reached. A caveat that
  still applies goes in **Caveats that apply to every hypothesis** or in the card's Confidence
  line — never as a section about corrections, a figure caption or a subtitle.
- **Process vocabulary** outside Methods. `lint_hypothesis_report.py` fails the report on words
  phrases such as *admission gate, funnel, blind run, leaked term, predecessor study, answer key,
  held-out key, evidence class, selection rule, in round, round 1, ledger, reviser, falsifier,
  sub-agent, literature verdict, kill test, survived* (the script holds the full list). If a sentence needs one of them, it is about the
  process and should be cut.
- **The words "honest" and "genuine".** Verbatim: *"Be scientifically accurate but do not use the
  words 'honest' or 'genuine'."* The confidence field is called `Confidence`, not `Honesty`.
- **Hype**: novel, powerful, robust, striking, elegant, seamless, cutting-edge. If a result is
  strong, the number shows it. (`render_report.py` lints for these.)
- **Throat-clearing**: "It is worth noting that", "Importantly", "Interestingly", "In this
  section we".
- **Coined shorthand** without an inline definition at first use — and once defined, reuse the
  exact phrase.
- **Narrated arithmetic.** Never write "N of M candidates have X" from memory. Recompute it from
  the table and register it. One prior draft carried five independent miscounts in one pass.
- **Hedging clouds**: somewhat, rather, fairly, essentially, may possibly. If a hedge carries real
  meaning, spell the caveat out instead.

---

## 5. Required

- **The sign convention in the preamble**, and `*driver*` / `*brake*` (or the study's equivalent)
  italicised throughout. This single definition is most of why the best prior report is readable
  by a non-specialist.
- **Hypothesis card titles state the claim as a finding, in ≤ 8 words.** "VHL loss raises the
  hypoxia program", not "VHL and hypoxia". A topic label is not a title.
- **Programs are named by their glossary title, never by number.** "the hypoxia-response
  program", not "P12" or "program 12". The lint fails bare program numbers and other internal
  codes (`H07`, `C6`).
- **Define before use.** Every term in a claim is either one a working scientist knows or is
  defined in that sentence. If a claim runs against a prevailing model, state the model in one
  plain sentence first.
- **Inline hyperlinked citations** (PMID / DOI / NCT) at the claim, not only in a reference list.
- **The boring-true half separated from the interesting-uncertain half** in every "What is already
  known": *"So '<the broad pattern>' is correct and expected. The new part is <the specific
  remainder: the sign, the cell type, the scale>."*
- **The artifact class that could kill the finding, named, with the check to run:** *"could be a
  mirror artifact — check mirror-fraction before believing it."*
- **Where evidence supports nothing specific, say so.** *"admit/flag when there just isn't enough
  evidence to propose anything specific."*

---

## 6. Titles

**Program titles** (the glossary):

- 2–4 words. Six only when needed to separate similar programs.
- Never the word "program" inside a program title.
- **Globally unique and coordinated.** Not three programs all called "Interferon Response" —
  disambiguate by a distinguishing gene, pathway, or concept: "Interferon Response — IRF7".
- Where several annotation sources exist, carry all of them and mark each one. Prefer the
  LLM/GPE-derived label over bare GO enrichment when the LLM had cell-type context — but show
  both, labelled.

**Hypothesis card titles**: the claim in ≤ 8 words, stated as a finding (§5).

---

## 7. Before/after, from real drafts

| Draft | Revised |
|---|---|
| "timepoint-shared (r=-0.09, at the null); stimulus-specific, consistent across timepoints (r=+0.36). Net effect on the four nominal stimulus-up programs (<program codes>) +0.14 — nominal only; no program survives correction (<finding id>), so this is not a reversal claim." | "Stronger effect in stimulated than resting cells (1.7×); no clear difference between early and late timepoints. The four programs that weakly trend up with stimulus move by +0.14 overall, which is not statistically significant once corrected — this is not a reversal result." |
| "In this analysis we sought to investigate the relationship between knockdown efficiency and evidence quality, finding a notable association." | "Knockdown efficiency does not predict whether a gene's downstream signature reproduces (r = +0.08, n = <targets>); requiring a minimum knockdown discards 11 genes that replicate strongly below 25% knockdown." |
| "KEAP1 is an uncharacterized NRF2 regulator and a novel candidate." | "KEAP1 is an established repressor of NRF2 that binds its Neh2 domain (Itoh et al., *Genes Dev* 1999, [PMID 9887101](https://pubmed.ncbi.nlm.nih.gov/9887101/)). The screen placed it with the antioxidant-response genes by co-perturbation alone — that is method validation, not a discovery. The uncharacterized members of that cluster are the remaining candidates." |
