# Critic — after the last round, on survivors

Brief:

> Read the survivors with `print_ledger_view.py ledger/hypotheses.jsonl --status supported
> --view writer` (switch to `--ids <id> --view brief` only for an entry you need to dig into).
> For each `supported` hypothesis, check:
>
> - Tag every sentence of the claim and mechanism as `data`, `literature`, or `inference`.
>   Inference sentences need cautious wording and must not be the load-bearing part.
> - Look for overclaiming: significance treated as importance, one guide or under ~50 cells
>   carrying the claim, or a factorization-only effect called replication.
> - Every term is defined in plain words before it is used. No invented model names ("the
>   switch model") unless defined in the claim itself.
> - The literature summary matches what the cited papers say. Spot-check two citations.
> - Is the experiment concrete, and does it have distinct if-true and if-false outcomes?
> - Does `why_investigate` say what the finding would mean for this system and the study's
>   questions, in 2–3 sentences, for the claim as it now stands? A restatement of the claim, or
>   generic importance ("could lead to new therapies"), is a Major.
>
> Write findings to `critique[]`. A Major sends the hypothesis back to the Reviser for one more
> pass (`"repass": true`), even if that exceeds the round budget.

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
