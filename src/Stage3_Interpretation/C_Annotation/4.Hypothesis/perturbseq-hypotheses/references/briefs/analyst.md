# Analyst — per hypothesis (all its open tests), or a batch of at most six tests

Brief:

> Run the tests you were given (`<id>`: `<test ids>`). Each spec is in the ledger. For each test,
> write to `traces/<id>/<test_id>/`: the code, one figure (PDF + PNG + the CSV of plotted
> values), `stats.json`, and `result.md`. In `result.md`, give the verdict
> (`supports / refutes / inconclusive`) and one sentence with the numbers. Then write the
> result into the test's ledger fields before starting the next test.
>
> **Resume first.** If `traces/<id>/<test_id>/result.md` already exists, the test is done: copy
> its verdict into the ledger if the ledger lacks it, and move on. Never redo a finished test.
>
> **Use the shared loaders.** `src/screen_io.py` (the Explorer's adapted loaders; what each
> returns is under "Loaders" in `explore/README.md`) and `src/plot_hypothesis_evidence.py`
> already fix the significance column, effect column, sign and technical programs. Import them;
> do not write new loaders or re-derive those choices. If a test needs data they do not load,
> add a loader to your test's own code, not to `src/screen_io.py`.
>
> Read only the inputs listed in `inputs/MANIFEST.txt` (and public datasets the test names).
> Do not browse other directories on Oak or elsewhere.
>
> Then look at your own figure once it is final: does it show what `result.md` says? Do not
> reopen figures you have already checked. Judge the test by its pre-registered kill condition,
> not by whether the hypothesis would survive. Do not reword the hypothesis.

**Reruns** (dispatched separately, `SKILL.md` step 3.3): a fresh Analyst that did not run the
original reruns the counter-example test an entry could survive on, from the spec alone, into
`traces/<id>/<test_id>/rerun/`, without reading the original trace. A mismatch with the original
verdict makes the test's `result` `inconclusive`; record both summaries.

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
