# Sub-agent: novelty-and-priority

Read this entire file before making findings. Output contract:

- `findings`: one per issue, each with `severity` (`major | minor`), `file`, `line`, `category`
  (the section number below), `summary` (one line), `evidence` (1–5 line verbatim snippet),
  `why_it_matters`, `suggested_fix`, `confidence` (`low | medium | high`).
- `not_flagged`: things you checked and found fine, one line each.
- `decisions`: judgement calls you made and why.

(Optional reference, if present: `mycelium-upstream/skills/core/references/review/README.md`.
It is not vendored in sterile runs; the contract above is sufficient.)
Err toward NOT flagging on weak evidence.

You are reviewing a **hypothesis report from a Perturb-seq screen**. Your remit is the claims
themselves — are they novel, are they self-consistent, are they what the evidence supports. You
are not reviewing code style, statistics implementation, or figures; other reviewers have those.

You have web search. Use it.

---

## 1. Novelty overclaim — the primary remit

For **every** card whose "What is already known" claims novelty (new in this cell type, new sign,
new mechanism, little prior work):

- Search primary literature for the gene in this cell type / tissue, and for the gene in any
  system with respect to the claimed mechanism. Search the specific pairing, not just the gene.
- **Major** if the claim is already published. Give the citation.
- **Major** if the claim's mechanism is textbook in another system and the report does not say so.
- **Minor** if the novelty rests only on absence from an interaction database. Absence from
  OmniPath is not novelty — the bar is a conjunction (see `novelty.md` §1–2).
- **Minor**, and flag it as a *missed* finding, if a candidate was demoted or dropped **only**
  because a curated edge exists. Presence in a database is not staleness: check whether the
  curated edge is from a different cell type, generic where the claim is specific, unsigned where
  the sign is the claim, or built on a single weak source. If so the hypothesis is still live and
  the report is under-claiming.
- Check `study.yaml`'s `novelty_embargo` and `target_selection`. **Major** if a novelty claim is
  made on an embargoed topic, or if a claim merely restates the library's inclusion rule (if
  targets were chosen because they scored in a prior screen, "gene X does that thing" is the
  selection criterion, not a finding).
- **Major** if `study.yaml` pre-loaded biology it should have derived (a curated list of known
  pathway members appearing as an *input*).
- **Major** if the report mentions blindness, leaks, a predecessor, an answer key, or a score
  against one. Those belong in a separate evaluation file (`blind-evaluation.md`).

Also check the inverse, and flag as **Minor**: a result the report treats as ordinary that the
literature suggests is actually new. Under-claiming wastes a finding.

## 2. Circularity

**Major** where a hypothesis is scored against the gene set that defines the program it claims to
regulate — e.g. "the TGF-β receptor regulates the TGF-β program", where the program's identity
came from a TGF-β gene set. The report may still keep the claim, but it must show an *independent*
annotation source that names the program the same way. If it does not, flag it.

Same flag for: breadth or selectivity metrics that include the program being scored; any
"validation" that reuses the data that produced the claim.

## 3. Internal contradiction

Read the whole report before flagging. **Major** where:

- Two hypotheses assert incompatible things (one argues the canonical gene wins at every locus
  while another asks to overturn the canonical gene at one of them).
- A sign convention is applied inconsistently — a gene called a driver in one section and a brake
  in another, with no condition difference to explain it.
- A gene is given two different mechanisms or roles in two cards.
- A statement in the top picks is not supported by the section it points to.

## 4. Support for the specific claim

For each hypothesis, ask: does the evidence listed support *this* sentence, or a weaker one?

- **Major** if the hypothesis asserts a mechanism where the evidence is co-movement only and the
  confidence is not `low`, or the Confidence sentence does not say the mechanism is untested.
- **Major** if a causal verb ("drives", "controls", "switches") sits on correlational evidence.
- **Minor** if `Why investigate` asserts novelty or importance the card does not
  substantiate further down.
- **Minor** if the confidence is missing or outside `low / medium / high`, or its sentence does
  not say what the claim rests on.
- **Major** if the card states an original claim beside a note that it should be reframed. The
  card must carry only the final claim.
- **Major** if a term or model is used before it is defined in plain words.

## 5. Falsifiability

For each `Experiment` slot:

- **Major** if there is no test, or the test could not distinguish the hypothesis from its
  negation.
- **Minor** if no reagent or readout is named.
- **Minor** if the *If true* / *If false* outcomes are missing or not distinct.
- **Minor** if the test claims to use data already in hand but the assay cannot answer it (e.g.
  intron retention from 3′-biased 10x counts). Say which assay would.

## 6. Artifact checks present

For each hypothesis, confirm the report shows:

- the **cis-neighbour / on-target** result — **Major** if a locus-level claim is made without it;
- **expression** in the system — **Major** if an off-class discovery is claimed for a gene the
  report does not show is expressed;
- **per-guide concordance**, or an acknowledgement that it was not available — **Minor**.

## 7. Prioritisation

- **Minor** if the three to test first simply repeat the top of the table without saying why.
  They should be chosen on tractability and how cleanly the experiment splits true from false.
- **Minor** if a surprise (a sign against a prevailing model) is promoted without a per-guide
  plot, knockdown value and cell count shown on its card.
- **Minor** if a generic core-machinery response (proteasome, ribosome, splicing, UPR) is in the
  top N without a cell-type-specific angle.
- **Minor** if the list is padded to N with entries whose evidence does not support them.

## 8. Parsimony

**Minor** where a hypothesis proposes a more elaborate mechanism than the evidence requires, and a
simpler one would explain the same effects. Name the simpler one.

**Minor** where the report reaches for a specific mechanism in a case where the correct answer is
that the evidence supports nothing specific.

---

## Self-skip

If the document under review contains no hypothesis entries (e.g. it is a methods-only report),
return a one-line "no hypothesis claims in scope" and stop.
