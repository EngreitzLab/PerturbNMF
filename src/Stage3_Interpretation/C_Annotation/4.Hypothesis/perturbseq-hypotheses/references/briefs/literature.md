# Literature — per hypothesis, whenever its claim is new or has changed

Brief:

> Search the primary literature (PubMed / Europe PMC / bioRxiv / web) for **this claim**, not just
> the gene names. Answer:
>
> 1. Is the mechanism known in this cell type?
> 2. Is it known in another cell type?
> 3. Is there direct evidence against it?
> 4. What is the strongest paper for and the strongest paper against?
> 5. What would a specialist say is missing?
>
> Return a verdict (`known_in_this_cell_type / known_elsewhere / novel / contradicted`) and up to
> five citations. Each citation needs a resolvable PMID or DOI plus one sentence on what that
> paper actually shows. Default to "possibly known" when unsure. Write your notes to
> `literature/<id>_r<round>.md`. Suggest how the claim could be reframed, but do not edit it.

Rules from `novelty.md` §1–2 apply: database presence or absence is triage, not a verdict. The
predecessor study on the same screen is **allowed** in the search. It is prior art like any other
paper.

**Only entries that need a check.** The orchestrator sends you only entries with no literature
check at or after their last version bump; an unchanged claim keeps its earlier check.

**Search economically.** Triage from search-result titles and snippets first, then fetch full
records for at most about eight papers per hypothesis. Write `literature/<id>_r<round>.md` and the
ledger entry as soon as each hypothesis is done, before starting the next; never re-fetch a
paper whose record you already hold.

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
