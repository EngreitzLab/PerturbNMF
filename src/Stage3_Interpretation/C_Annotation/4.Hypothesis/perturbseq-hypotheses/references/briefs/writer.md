# Writer — once

Follows `report-template.md`, and `cartoons.md` for the schematics. Draws one cartoon per
`supported` entry before writing the cards — the drawing forces the mechanism to be stated as
geometry, and an entry whose contrast cannot be drawn is a warning about the claim, not about the
drawing. Inputs:

- `supported` entries and their headline figures;
- `explore/program_glossary.tsv` and `explore/README.md` (caveats);
- `study.yaml`: subtitle (system, assay), sign convention, questions, caveats;
- `report/killed_extract.tsv`, written by the orchestrator: id, title, `kill_reason`, key
  citation for every killed or shelved entry, plus the total candidate count. It feeds the
  appendix and the "M candidates → N survivors" line in Methods.

Read entries with
`python <skill>/scripts/print_ledger_view.py ledger/hypotheses.jsonl --status supported --view writer`.

It never sees transcripts, `history[]` or rounds. If a fact is not in the ledger, a trace or these
files, it does not go in the report.

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
