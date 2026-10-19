# Report template — the hypotheses are the report

The reader is a PI or collaborator who wants to know **what the screen suggests, and what to do
next**, in five minutes. They do not want to know how the hypotheses were produced. Everything
about the process lives in `ledger/`, `traces/` and one short Methods paragraph at the end.

Rendered via `/engreitzlab-report` → `/analysis-report` (Quarto → one self-contained HTML), then
linted by `scripts/lint_hypothesis_report.py`.

---

## Files

| File | What |
|---|---|
| `report/<study>_top<N>_hypotheses.qmd` | The report. One file; the ranked table links to per-hypothesis sections further down. |
| `report/<study>_killed_hypotheses.qmd` | Appendix: one line per killed or shelved candidate, with its reason and key citation, built from `report/killed_extract.tsv`. Linked once from Methods. Not required reading. |
| `traces/` | Code, figures and stats behind every claim. Linked from each hypothesis card. |

## Outline of the main report

```
# Top <N> hypotheses from <screen, in plain words>
<one-line subtitle from study.yaml: system, assay, and the sign convention ("driver"/"brake")>

## Three to test first
   3 numbered items. Each: the claim in one sentence, why investigate in one sentence (shortened
   from the card), the
   experiment, and what each outcome would mean. Link to the full card.

## All <N>, ranked
   One table. Columns: # | Hypothesis (linked title) | Claim, one sentence | Confidence | Key evidence, one phrase
   Nothing else: no gate counts, no class labels, no audit columns.

## Overview figure
   The top N regulators across the programs they move (plot_overview_map). One legend.

## The programs
   Collapsible glossary: program title → what it likely is → top 10 genes. Only the programs
   the N hypotheses mention.

## <Hypothesis 1 title> {#h1}
## <Hypothesis 2 title> {#h2}
   … one card each, in rank order (template below)

## Caveats
   Caveats that apply to every hypothesis, from study.yaml :: caveats and explore/README.md.
   Bullets. Five at most.

## Methods
   One paragraph: data, significance test, how hypotheses were generated, tested against the data,
   checked against the literature, and revised or dropped (M candidates → N survivors, counted
   from report/killed_extract.tsv; the rest are in the appendix, linked). Positive controls get
   one sentence here. Environment line. Regenerate command.
```

## The hypothesis card

```
## <Title: the claim in ≤ 8 words, stated as a finding> {#hK}

**Claim.** One or two sentences, mechanistic, with every term already defined or defined here.

<cartoon: the mechanism — known above, proposed below (`cartoons.md`) — with the one-line legend
*Top row: what is already established. Bottom row: what this screen proposes.*>

**Why investigate.** 2–3 sentences, from the ledger's `why_investigate`: if this is true, what is
the story, and what would it mean for this system and the study's questions? Not a restatement of
the claim, and not generic importance.

**Confidence: <low | medium | high>.** One sentence on what it rests on and what could still
break it.

<headline figure: 1–2 panels, with a full Nature-style legend in <details>>

**What the screen shows.**
- 2–4 bullets with the numbers, each ending in a [trace](traces/…) link.
- Include the counter-example test that it passed.
- Name what is specific to this cell type (or say it is not known).

**What is already known.**
- 2–4 bullets: what the literature establishes, in which cell type, each with an inline
  hyperlinked citation. Then say plainly what is new here.

**Experiment.** The design. *If true:* … *If false:* …

<details><summary>More evidence</summary>
Perturbation neighbours, per-guide plot, knockdown, cell counts, other tests run.
</details>
```

## Rules the lint enforces

`scripts/lint_hypothesis_report.py --n-hypotheses <N> [--allow SYMBOL …]` fails the report if:

1. The title does not start with `Top <N> hypotheses`.
2. The first `##` section is not the top picks, or the second is not the ranked table.
3. A process phrase appears anywhere outside `## Methods`, the preamble included. The list is
   phrase-level, for example: *admission gate, answer key, held-out key, predecessor study,
   evidence class, selection rule, ledger, reviser, falsifier, sub-agent, the skill, literature
   verdict, blind run, in round, round 1, kill test, survived, funnel, uncontaminated*. The script
   holds the full list.
4. A hypothesis card is missing Claim, Why investigate, Confidence, What the screen shows, What is
   already known, Experiment, or the if-true / if-false pair.
5. A card has no figure: an image, or a code chunk with a figure label or caption.
6. Internal codes (`X1`, `C6`, `H07`, `P36`) appear in prose. Use titles and gene names.
   Programs are referred to by their glossary title, never by number alone. Gene symbols with the
   same shape (`H19`, `C3`) are on a default allow list; add others with `--allow`.
7. The number of cards does not equal N, the number of `supported` entries in the ledger.

## Writing rules

`voice.md` applies. On top of it:

- **State the hypothesis, not its history.** Never "originally we proposed X; the literature
  showed Y, so it was reframed." Write the final claim. The history is in the ledger.
- **Define before use.** If a claim runs against a prevailing model, say what the model is in
  one plain sentence first: "Active mTORC1 blocks autophagy, so losing an mTORC1 subunit usually
  raises autophagy genes." Then say what the screen shows.
- **Integrate, don't list.** "What is already known" and "What the screen shows" should combine
  into one coherent story in the Claim: what is established, what the screen adds, why that
  matters.
- **Confidence is about the claim, not the gene.** A strong effect with an untested mechanism is
  `medium` at best.
- **No numbers without a unit or a comparison.** "+0.9 log2FC on the hypoxia program
  (safe-targeting median 0.02)".

## Figures

Two per card, doing different jobs: the **cartoon** says what the claim means, the **headline
figure** shows the data it rests on. Neither stands in for the other. Cartoon structure,
vocabulary and the check to run are in `cartoons.md`.

Each card's headline figure is chosen from its traces, usually:

- **Effects across programs** for the hypothesis members plus their complex partners, with
  significance marked (`plot_effects_across_programs`).
- or **per-guide agreement** when the claim rests on a surprise (`plot_guide_concordance`).
- plus **program top genes** when the program's identity carries the claim.

House style: `EngreitzLabAgents/style/house_style.yaml`; vector PDF + PNG + CSV of plotted values.
Four failure modes a real review caught:

- On-plot labels must be built from the same values as the marks.
- A legend must describe the encoding the code implements.
- A masked cell must not render as a colour the palette gives meaning to.
- If colour is the only key, the colours must differ in luminance.

## Render toolchain (verified 2026-09-23)

1. Quarto is not on `PATH`: `PATH="$(dirname "$(mise which quarto)"):$PATH"`.
2. `render_report.py` from `analysis-report` is stdlib-only; system `python3` runs it.
3. Put the house-style commit sha, the `analysis-report` sha and `quarto --version` on the
   Environment line.
4. The final render must run with the Bash sandbox disabled (deno_kv SASS cache). Lint-only runs
   sandboxed.
