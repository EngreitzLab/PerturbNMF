# Evidence classes — tags for spreading candidates

These classes are **tags**, nothing more. The Generator gives each candidate one or more of them
(ledger field `tags`) so the candidate set spans different kinds of hypothesis. The Ranker reads
`tags` to avoid a top N that is all one kind. They are never report sections, never table
columns, and never named in the report: the report is ordered by rank.

## Why tags and not a score

A prior project spent four days computing "interesting" as a composite score. Every version died:
ranking on hit counts measured statistical power, count-based stratum axes measured stratum
calibration, and a bespoke replication statistic duplicated a better-calibrated test that already
shipped with the run. What survived was a declared shortlist, and its docstring gave the reason,
paraphrased: the targets passing the test were dominated by canonical regulators of the dominant
pathway, so ranking purely by breadth would return textbook genes and bury the ones only the
screen could nominate. Spreading across kinds of evidence, declared openly, is auditable in a way
a composite score is not.

This matches the standing instruction: *"I care less about the exact metrics for ranking, and more
about having a large set of interesting, well-supported hypotheses that we can explore further."*

So v2 has no composite score. Survival is decided by falsification tests and the literature
check; ordering is the Ranker's judgement; tags only keep the candidate pool and the final list
diverse.

---

## The default eight tags

Configure extra tags in `study.yaml :: special_interest`; add study-specific tags freely.

### `control` — the gene's top program is the one its biology predicts

Framework validation. If a screen cannot recover the biology everyone already knows, nothing else
it says is believable (e.g. VHL knockdown raising a hypoxia-response program).

A surviving `control` candidate is usually a **positive control**, not a top-N hypothesis: it goes
in `explore/positive_controls.tsv` and gets one sentence in Methods (`novelty.md` §6.2). It earns
a card only if the Reviser finds a cell-type-specific remainder.

### `understudied` — large effect, little or no literature in this system

What the screen uniquely adds. An understudied gene is worth more than a canonical one because
the screen is the only evidence that exists.

Look for: a large, reproducible effect with low curated and topic-restricted literature counts.
Carry the ambiguous-symbol flag: symbol collisions between a gene and a similarly named paralog
(`<GENE>` vs `<GENE>L`) inflate and deflate literature counts in both directions.

**Perturbation neighbours matter most here.** For a gene with no tissue literature, "its effect
vector correlates +0.70 with `<receptor>`, +0.69 with `<adaptor>` and +0.63 with `<kinase>`, three
members of one signalling pathway" is the single most informative result available.

### `disease_locus` — a GWAS or Mendelian-locus gene with human support

The translational hook. Human genetics is an **annotation on the screen, not a second discovery
axis**: small weight, never a filter.

Look for: locus membership from the study's own variant set, plus whatever human evidence exists
(Genebass, Open Targets, FinnGen, gnomAD constraint, IMPC). **Every locus claim needs the
cis-neighbour check first.** CRISPRi at one promoter silences its neighbours, and that has
manufactured exactly the signature a locus hypothesis rested on.

### `mirror` — a weak effect supported by anti-correlation with a known partner

A gene whose own evidence is marginal but whose effect vector anti-correlates cleanly with a
well-understood partner is a real hypothesis with a built-in control.

### `contrast` — strong human genetics, no screen response

The informative negative. It is not a hypothesis on its own: it is a statement about power or
about which cell type carries the genetic effect. Record it in `explore/`; if it bears on every
hypothesis, it becomes one line in the report's caveats.

### `cross_program` — branch point, shared hub, or sign flip across programs

Only a genome-wide screen with a program decomposition can see this, so it is where the method's
unique signal lives. Three shapes:

- **Branch point**: one regulator with opposite signs on two programs. One receptor, two jobs.
- **Shared hub**: one gene that is the top regulator of many programs at once. The new part is
  usually the *scale*, not the mechanism.
- **Reciprocal pair**: two genes acting as the two ends of one hinge across a program family.

A prior report ordered these, in its words, *"roughly by how much they use the new cross-program
signal (branch points, shared hubs, sign-flips) and how testable they are."* In v2 that is an
input to the Ranker's interest rating, not a separate ordering.

Where a study's whole point is cross-program structure rather than per-gene nomination, the
Generator weights `cross_program` candidates more heavily. The report is still ordered by rank.

### `unexplained_member` — inside a coherent group, and the group's function does not explain it

Where a RegulatorGroupAnnotator run exists for the screen, it groups regulators by their effect
profiles and annotates each group the way a program is annotated: a label, the function its
members share, and a role for every member — `core_explained`, `consistent` or `unexplained`. An
**`unexplained` member moves the programs the way a coherent complex does while having no evident
link to what that complex does**, and the annotator is required to leave it standing as a
hypothesis rather than explain it away, carrying its own `hypothesis` and `what_would_test_it`.

This is a hypothesis the other axes do not generate. The gene need not have a large effect, thin
literature or a locus: its claim comes from the company it keeps. Rules:

- **Every `unexplained` member is a candidate seed**, worked through rather than sampled from.
  The entry names the group, the group's shared function, and what about this member that
  function fails to cover.
- **The annotator's hypothesis is a prior, not a result.** One blinded model call wrote it from
  evidence; nothing has tried to kill it. Restating it is not a candidate — the Generator's claim
  has to be its own, and the Falsifier tests it like any other.
- **The eligibility gate still applies** (`data-hygiene.md` §2). The annotator excludes members
  whose membership a neighbouring promoter could explain, which is complementary to
  `regulator_eligibility.tsv` and not a substitute for it; a member it merely *flagged* for a
  promoter caveat is `flag` at best.

### `complex_function` — the group is coherent, and what the complex is doing is the claim

The other half of the same source. Where a group's `coherence` is `strong` and its members are a
known complex, the interesting object may be the complex rather than any one gene: what it does
**in this system**, read through the programs it moves, can be new even when every member is
individually well studied.

Look for: a group whose shared function is established elsewhere but whose effect signature here
is not what that function predicts; a complex moving a program family nobody has connected it to;
or a complex whose members split across programs — which is a `cross_program` claim about a
complex, and takes both tags.

The bar is the same as everywhere else: restating the complex's textbook job is not a hypothesis.
What it is doing *here* is.

---

## How many

The Generator aims for a spread: a few candidates per tag where the data supports it, never
padding a tag. The report size is `loop.n_hypotheses`. The Ranker never promotes a weaker entry
only for diversity (`briefs/ranker.md`).
