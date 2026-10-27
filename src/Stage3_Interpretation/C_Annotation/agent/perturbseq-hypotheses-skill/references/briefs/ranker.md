# Ranker — on survivors

Read the survivors with `print_ledger_view.py ledger/hypotheses.jsonl --status supported
--view writer`. Score each survivor 1–3 on **interest** (judged mainly from `why_investigate`: bears on the study
questions; cell-type-specific; would change what someone does next), **support** (tests passed, internal replication, specificity), and
**tractability** (the killing experiment is cheap and clear). Then order them by pairwise
comparison, using the scores as tiebreaks. The ranking is a judgement, not evidence: never quote
the scores as findings. Pick the top 3 for an experiment separately, on tractability and how
clean the if-true/if-false split is.

Use `tags` to keep the top N diverse: when two entries are close, prefer the one whose tag, program
or study question is not already represented. Never promote a weaker entry only for diversity.

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
