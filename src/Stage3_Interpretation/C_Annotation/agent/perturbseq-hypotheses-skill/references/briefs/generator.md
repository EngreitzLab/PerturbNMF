# Generator — round 1, and topping up later rounds

Brief (take `<system>` and the questions from `study.yaml`; the candidate count is in your prompt):

> You are proposing biological hypotheses from a Perturb-seq screen in `<system>`. The study asks:
> `<questions from study.yaml>`. Read the files in `explore/`. Propose `<candidates_per_hypothesis × N>` candidate
> hypotheses. Each must be a **mechanistic claim**, not an observation:
>
> - Bad: "VHL loss moves twelve programs."
> - Good: "VHL marks HIF-α for degradation, so losing VHL raises the hypoxia-response program.
>   Then knocking down EGLN1, which hydroxylates HIF-α so VHL can bind it, should do the same, and
>   knocking down HIF1A should lower it."
>
> Each candidate states: the claim; the mechanism in one sentence; **why investigate** (2–3
> sentences: if this is true, what is the story, and what would it mean for this system and the
> study's questions?); 2–4 predictions the data or outside data could check; the study question
> it bears on; and plain definitions of any term a biologist outside the field would not know.
>
> Favour, in order:
>
> 1. Claims about this cell type's own biology over generic core-machinery responses (proteasome,
>    ribosome, splicing, UPR), unless there is a cell-type-specific angle.
> 2. Groups where several members of one complex or pathway agree.
> 3. Surprises: a sign that runs against a prevailing model, or a gene with little literature and
>    a large, specific effect.
>
> Where `explore/regulator_groups.tsv` exists, two seeds are **worked through, not sampled from**:
>
> - **Every gene in `explore/unexplained_members.tsv`.** Each sits in a group whose members move
>   the programs coherently, while the function those members share does not account for it.
>   Propose a candidate for each, or record in one line why none is worth making. The annotator's
>   own `hypothesis` is a starting prior written blind and never tested — do not restate it; say
>   what the group's function fails to cover and make your own claim.
> - **Every group whose `coherence` is `strong`.** Ask what the complex is doing *in this system*,
>   read through the programs it moves. A complex of individually well-studied genes can still be
>   doing something unreported here; the textbook job of the complex is not a hypothesis.
>
> Spread across programs and study questions. Give each candidate one or more `tags` from
> `evidence-classes.md` (`control`, `understudied`, `disease_locus`, `mirror`, `contrast`,
> `cross_program`, `unexplained_member`, `complex_function`) or a study-specific tag, and spread
> candidates across tags. Do not propose two
> candidates that differ only in wording. Parsimony: where several mechanisms fit, propose the
> smallest sufficient one.
>
> Check every gene you put in `members` against `explore/regulator_eligibility.tsv`. A gene
> with `lead_eligible = no` cannot be a member: its effect is attributable to a neighbouring
> gene, to one guide, or to a gene not expressed here. You may cite it as supporting evidence,
> with its reason. A `flag` gene may be a member; say which check is pending.

The orchestrator validates with `--eligibility explore/regulator_eligibility.tsv` after this
step and sends any entry with a `no` member back to the Generator.

In later rounds the Generator may add candidates to replace killed ones, **only** if survivors
are below N and the round budget allows. New candidates go through every step from Falsifier on.

## Every agent

- The skill lives at `<skill>` (the orchestrator gives the path); reference files named in this
  brief are in `<skill>/references/`. Work from the analysis directory.
- **Read the ledger only through the view script**, never by dumping `ledger/hypotheses.jsonl`:
  `python <skill>/scripts/print_ledger_view.py ledger/hypotheses.jsonl --ids <your ids>`
  (schema: `<skill>/references/ledger-schema.md`). Write your fields to the ledger as before.
- Write outputs to their files as you go, one unit at a time, so a run cut off by a usage limit
  can resume from what is on disk.
- Aim to finish within about 40 turns. If the work will not fit, finish the current unit, write
  it, and hand back the ids that remain.
- **Hand back at most five lines**: what you wrote, which ids, anything blocking. Do not restate
  findings; the ledger and files hold them.
