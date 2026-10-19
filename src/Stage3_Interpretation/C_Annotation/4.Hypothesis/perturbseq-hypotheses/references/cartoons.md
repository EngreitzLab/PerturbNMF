# Cartoons — one per surviving hypothesis

A hypothesis card carries a headline figure showing *the data*. That figure cannot show what the
claim **means**, and a reader outside the field will not assemble it from effect sizes. So every
`supported` hypothesis also gets a cartoon: a hand-drawn schematic of the mechanism, drawn in two
rows — **what is already known on top, what is proposed below**.

The Writer draws them, after the claims are final and before the report is written. One SVG per
hypothesis, `report/cartoons/<id>_<the gene the card is about, lower case>.svg`, recorded as
`cartoon` on the ledger entry. The final validator requires the file to exist.

**It goes directly under the card's Claim paragraph**, before the evidence — the reader meets the
mechanism, then the data — with a one-line legend: *Top row: what is already established. Bottom
row: what this screen proposes.* Not above the headline figure, and not at the foot of the card.

## The rule that makes a cartoon worth drawing

**The contrast has to be visible as geometry, not only as words.** If the claim is "acts two steps
earlier", draw the same axis in both rows and put the two events at different places on it. If the
claim is "the opposite sign to the literature", draw an activating arrow above and a brake below.
If the claim is "these two perturbations differ", draw them side by side with the matched quantity
tied between them.

A cartoon that would read the same with the gene names swapped is not finished.

## Draw, do not explain

The picture carries the claim. The card's prose is immediately above it and must not be restated
underneath it. **Text on a cartoon earns its place only by labelling geometry** — what a box is,
what an arrow does, "every guide agrees", "not measured here", "untested".

- Two or three short lines per row in the left column. One clause per line. No paragraphs.
- No prose blocks inside a panel. If a sub-panel needs three sentences to make sense, the
  geometry is wrong — redraw it.
- Two versions of the same set were compared on a real report and the sparse one won: the
  explainer-heavy panels duplicated the card and read worse.
- Cutting text is exactly when a box ends up oversized for its contents and a label ends up
  orphaned, so re-render and look again after trimming.

The one thing never cut for brevity is an honesty qualifier — *untested*, *not measured here*,
*published elsewhere*. Shorten it to a label; do not drop it.

## Structure

```
Known:        <2–4 short lines, left column>      │  panel: the established mechanism
Hypothesis:   <2–4 short lines, left column>      │  panel: the proposed mechanism
```

- 1400 × 900 viewBox, white background, two light panels (`#f7f7f8`) on the right, text column on
  the left. Split a panel with a dashed vertical rule when the row shows two conditions or two
  literatures side by side.
- One idea per row. If the top row needs three known facts, the claim is doing too much.
- Sub-panel headers bold black; everything explanatory in grey at 12.5 px.

## Vocabulary — keep it identical across a report's cartoons

| Meaning | How it is drawn |
|---|---|
| Program, cell state, stage | rounded rect, green fill `#d8f0e3` / stroke `#5fbf96` / text `#1c6b4b` |
| The gene the hypothesis is about | rounded rect, purple `#e3e0fb` / `#8b7fe8` / `#3b2f9e` |
| Complex or known machinery | rounded rect, grey `#eceff1` / `#b0b6bb` / `#37474f` |
| Where the effect lands — the day, the state that changes | rounded rect, orange `#fbe0d8` / `#e59277` / `#8c3a1e` |
| Outside this experiment, or untested | same shape, dashed stroke, muted grey fill `#f4f6f7` |
| Promotes / produces | thin arrow, `#8a9299`, arrowhead |
| Restrains | line ending in a perpendicular T-bar, same colour as the restraining gene |
| Blocked step | red `#c0392b` cross on the arrow |
| Rises / falls | short vertical arrow, `#b5651d` up, `#c0392b` down |
| Membrane | band `#fbeecb` / `#e0c98a`, with `extracellular` and `cytoplasm` labelled — only when the mechanism is actually at the membrane |

## What goes on the cartoon, and what does not

- **Qualitative support, yes:** "every guide agrees", "still raised at the endpoint". These are
  what the picture is claiming.
- **Coefficients and q-values, no.** They live in the card's bullets and the figure legend. A
  number on a cartoon invites the reader to treat a drawing as a measurement.
- **The caveat that could overturn the claim, yes** — in a dashed box at the foot of the
  hypothesis panel. If a second perturbation reproduces the whole signature, or the claim rests on
  one endpoint with no trajectory, the cartoon says so. A schematic is the easiest place in a
  report to overstate, so it carries its own limit.
- **Never draw a mechanism the data cannot separate as though it were settled.** Two live routes
  are drawn as a fork, labelled untested, with the experiment that would separate them.

## Worked example (textbook biology, not from any screen)

Claim: *VHL restrains the hypoxia program at normal oxygen through degradation, not through low
transcription.*

- **Known row:** oxygen → EGLN1 hydroxylates HIF-α → VHL binds → HIF-α degraded. A linear chain,
  grey machinery boxes, arrows.
- **Hypothesis row:** two halves split by a dashed rule. Left, `VHL present`: HIF-α degraded, the
  hypoxia program (green, dashed = off). Right, `VHL knocked down`: HIF-α stable, hypoxia program
  green and solid with an up arrow, annotated "EGLN1 knockdown does the same, HIF1A knockdown
  reverses it" — the counter-example, which is the whole reason to believe it.
- **Foot caveat:** what a constitutive knockdown cannot separate from a normoxic-sensing claim.

## Drawing and checking

Write the SVG by hand. Do not generate it from a plotting library: the point is deliberate
geometry, and box-and-arrow renderers produce software diagrams.

Then **look at it**, every time. Text overflowing a box, a label sitting on top of an arrow and a
T-bar landing inside a box are all invisible in the source and obvious on sight:

```bash
qlmanage -t -s 1400 -o . report/cartoons/<file>.svg   # square thumbnail, quick check
```

or open the file in the browser pane at 1420 × 920 for a full-size view. Fix collisions before the
file is committed.

SVG is the deliverable: Illustrator opens it directly, so panels stay editable for the manuscript.
Render a PNG alongside it for the HTML report if the report embeds rather than links.
