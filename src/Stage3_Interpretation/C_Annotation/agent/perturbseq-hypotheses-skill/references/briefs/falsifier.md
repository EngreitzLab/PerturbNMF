# Falsifier — per hypothesis, per version

Brief:

> Here is one hypothesis (ledger entry `<id>`, version `<v>`). Design 2–4 tests **that could kill
> it**. At least one must be a counter-example test: a result the hypothesis forbids. Examples:
> "if this is the complex, the backbone subunit must move the program too"; "the effect must not
> appear in the safe-targeting arm"; "the effect must survive on held-out guides"; "the effect
> must not be explained by knockdown of the neighbouring gene".
>
> Always include these whenever they apply:
>
> - on-target (`data-hygiene.md` §2 check 1) and per-guide agreement, for any member that
>   `explore/regulator_eligibility.tsv` marks `flag`, or where the claim rests on a specific
>   program and day the Explorer's target-level verdict did not score. For a `yes` member, do
>   not repeat checks the Explorer already ran.
> - cell-count adequacy
> - cell-type specificity (`explore/specificity.tsv`)
>
> For each test write: what is measured; the result that would kill the hypothesis; the data it
> needs; and whether it is `data` (this screen), `external_data` (a public dataset), or
> `literature`. Register the tests in the ledger **before** any is run. Do not run them.

Tests are pre-registered: the validator rejects a test whose result was written in the same step
that registered it.

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
