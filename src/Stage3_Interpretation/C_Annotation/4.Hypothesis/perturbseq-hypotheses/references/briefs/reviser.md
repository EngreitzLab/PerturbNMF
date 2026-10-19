# Reviser — per hypothesis, end of every round

This is the step v1 lacked. Brief:

> Here is hypothesis `<id>` with its test results and literature verdict. You must choose exactly
> one action and write it to `history[]`:
>
> - `survived`: every test on the current version has a result, a counter-example test on data
>   returned `supports` (`inconclusive` does not count), and a literature check made at or after
>   the last version bump does not show it is already known in this cell type. Only now may status
>   become `supported`.
> Whatever the action, if the claim changes, rewrite `why_investigate` to match the new claim.
>
> - `sharpened`: narrow the claim to what the evidence supports. The new version needs new
>   predictions and new tests.
> - `reframed`: the literature or data points to a different mechanism. Write the new claim; the
>   old one goes in `previous_claim`.
> - `merged`: this and another entry are one hypothesis. Keep the stronger one; this one points to
>   it.
> - `split`: two separable claims. Create the second entry.
> - `killed`: a kill condition was met, or the mechanism is already proposed *and tested* in this
>   cell type with nothing new left. A gene appearing in another screen's hit table does not count
>   (`novelty.md` §1b). Write a one-line `kill_reason`.
> - `shelved`: the revision budget is spent and it is still unresolved.
>
> Write the event as `{"round": r, "action": …, "actor": "reviser", "reason": …}`, adding
> `previous_claim` for `sharpened` / `reframed` and setting `status` to match: `survived` →
> `supported`; `killed` → `killed` + `kill_reason`; `merged` → `merged` + `merged_into`;
> `shelved` → `shelved`; `sharpened`, `reframed`, `split` → `candidate`.
>
> **Adding a verdict without changing the hypothesis is not an action.** If the literature says
> "reframe", you reframe; the final document carries the reframed claim, never the original plus a
> verdict.
>
> When known biology explains part of the result, say what is left that is new (the cell-type
> context, the sign, the scale) and make *that* the claim. If nothing is left, kill it. A screen
> re-deriving known biology is a positive control: note it in `explore/positive_controls.tsv`,
> not in the top N.

A pass triggered by a Critic Major is one extra Reviser event with `"repass": true`, allowed
beyond `max_rounds`. The validator judges survival on the **last** Reviser event, so a repass that
sharpens a survivor sends it back to `candidate` with new tests.

A user mark from the human steer (`agent-roles.md`, last section) is binding: `user_kill` → action `killed`; `user_push`
→ the candidate is not killed or shelved this round unless a kill condition is met.

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
