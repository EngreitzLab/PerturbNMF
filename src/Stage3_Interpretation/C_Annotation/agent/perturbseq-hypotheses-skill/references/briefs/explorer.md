# Explorer — runs once, before round 1

Applies `data-hygiene.md` in full and writes compact, plain-text summaries the Generator can read
in one go. Every file is a small TSV with a one-line header comment saying what it holds.

- `explore/program_glossary.tsv`: one row per program. Columns: short title (2–4 words, see
  `voice.md` §6), what it likely represents in one plain sentence, top 15 genes by loading, top 15
  by abundance with junk filtered, the top 3 enrichment terms, and `technical: yes/no`. **The
  titles come from the data**: loadings plus enrichment. Do not copy them from a predecessor's
  program table.
- `explore/regulator_summaries.tsv`: one row per regulator that passes the `data-hygiene.md` §2
  filter. Columns: programs moved (signed effect and q), breadth, excess pleiotropy, top ±5
  perturbation neighbours, cells, knockdown %, cis-neighbour flag, per-guide agreement if
  available.
- `explore/regulator_eligibility.tsv`: the cis-neighbour, per-guide and expression verdicts for
  **every** admitted regulator, and the `lead_eligible` call they give (`data-hygiene.md` §2).
  All three checks run here, over the whole admitted set, before the Generator starts. Leaving
  them to the loop spends whole rounds killing hypotheses the checks would have excluded.
- `explore/co_effect_modules.tsv`: clusters of regulators whose effect vectors correlate. Record
  whether members share a complex (CORUM / STRING physical), since complex-level agreement is
  internal replication.
- `explore/regulator_groups.tsv` and `explore/unexplained_members.tsv` — **only where a
  RegulatorGroupAnnotator run exists for this screen** (`study.yaml :: regulator_group_annotation`;
  the annotator lives in PerturbNMF's `Stage3_Interpretation/C_Annotation/RegulatorGroupAnnotator`,
  its per-group `answer.json` files under the dispatch directory). Do not run the annotator from
  this skill; read its output, or record that there is none.
  - `regulator_groups.tsv`: one row per group — id, label, the shared function in one line,
    `coherence` (`strong` / `partial` / `weak` / `none`), the members with their role
    (`core_explained` / `consistent` / `unexplained`), stability, and any promoter caveat the
    annotator flagged.
  - `unexplained_members.tsv`: one row per `unexplained` member — gene, its group, that group's
    label and shared function, the annotator's own `hypothesis` and `what_would_test_it`, and the
    member's effect summary. Both feed the `unexplained_member` and `complex_function` tags in
    `evidence-classes.md`. Carry the annotator's hypothesis as a **prior**: it is one blinded
    model call, never falsified.
- `explore/specificity.tsv`: for each regulator, whether a public screen in another cell type
  (for example a Replogle genome-wide K562 or RPE1 Perturb-seq) shows the same regulator moving
  a similar gene set. Values: `cell_type_specific / shared / not_tested`. If no external screen is
  reachable, write the file with `not_tested` everywhere and say so in its header.
- `src/screen_io.py`: adapt `<skill>/scripts/screen_io_template.py` so every loader returns the
  tidy table its docstring names, with the significance column, effect column, sign and technical
  programs decided once. Copy `<skill>/scripts/plot_hypothesis_evidence.py` into `src/` and point
  its loaders at `screen_io`. Run `python src/screen_io.py --check`. Every Analyst imports these
  instead of writing its own loaders.
- `explore/README.md`, **one page at most** (every Analyst reads it): which significance column,
  which effect column, which program ranking carried each identity, the caveats the Writer must
  repeat, and a "Loaders" section holding the `--check` output.

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
