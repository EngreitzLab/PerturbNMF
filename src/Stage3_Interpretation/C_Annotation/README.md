# C_Annotation — agent-based gene-program annotation

Each cNMF program gets its own folder with its tables and a plan; one Claude agent per program works through the
plan, calling **skills** (one per data source) for gene information, interactions and literature, and writes the
program's label and layered interpretation. Every step is ticked in the plan only after host-side checks pass.
The programs are then read together (screen-wide ledgers and regulator synthesis) and rendered as a viewer.

```
C_Annotation/
  0.Prepare/     prepare_programs.py (program folders), build_confounder_screens.py, refresh_string.py (re-run
                 STRING in existing folders), program_dir.py (plan.md),
                 program_inputs.py (h5mu / table loaders)
  agent/         skills, Claude format: <name>-skill/{SKILL.md, scripts/, references/}
  1.Annotate/    run_program_agent.py (one agent per program), sandbox.py (Bubblewrap sandbox per agent),
                 validate_program.py (the step checks)
  2.Synthesis/   build_ledgers.py, synthesize_regulators.py, program_viewer.py (the viewer), publish_viewer.py,
                 html_report.py, program_site.py (helpers the viewer imports)
  api/           Claude API / session backends, budget ledger, run records, prompt loader, .env reader
  prompts/       prompts of the pipeline's own LLM calls (regulator synthesis)
  status.py      program x step status of a run
```

## Run order

| # | Step | Script (runner next to it) | Writes |
|---|---|---|---|
| 1 | Program folders | `0.Prepare/prepare_programs.py` | `<run>/Programs/P<k>/` tables, empty `literature.jsonl`, `plan.md` |
| 2 | Confounder screens (no LLM) | `0.Prepare/build_confounder_screens.py` | `confounder_screens.json` |
| 3 | Annotation agents | `1.Annotate/run_program_agent.py` | per program: gene info, interactions, literature, label, interpretation |
| 4 | Status | `status.py` (with its recheck option) | `<run>/Annotation_status.tsv` |
| 5 | Programs together | `2.Synthesis/build_ledgers.py`, `synthesize_regulators.py` | `<run>/Synthesis/` |
| 6 | Viewer (default output) | `2.Synthesis/program_viewer.py`, then `publish_viewer.py` to put it on a site | `<run>/Report/program_viewer.html`; `<site>/<dataset>/viewer.html` + root index |
| — | STRING refresh (existing runs) | `0.Prepare/refresh_string.py` | `string_sig.tsv`, `string_genes.json` (+ `regulator_context.md` / evidence where not yet interpreted) |
| — | Hypotheses (optional) | `agent/perturbseq-hypotheses-skill` (Claude Code skill) | its own run folder |

`<run>` is the annotation run folder, e.g. `<out_dir>/<run_name>/Interpretation/Annotation`. SLURM runners sit next
to the scripts; set `PIPELINE_ROOT` and the placeholders before submitting.

## A program folder

```
Programs/P10/
  plan.md                 the steps; [x] done, [ ] to do, [!] failed (with why), [-] skipped (optional)
  program.json            id, organism, cell type, conditions, usage per condition, counts
  genes_top300.tsv        rank, gene, loading, distinctive_score
  regulators_sig.tsv      every significant knockdown: regulator, condition, log2fc, adj_p, role
  go_sig.tsv              every significant GO term
  string_sig.tsv          every significant STRING term (+ string_genes.json: PPI enrichment, partners, and the
                          regulator network: STRING links among the regulators, typed experimental / curated /
                          predicted, with the regulators' PPI enrichment)
  confounder_screens.json
  gene_info.jsonl         one row per gene: MyGene / NCBI / UniProt
  interactions.jsonl      one row per tested regulator x gene pair (found or not); empty without regulators
  literature.jsonl        one row per paper: candidate -> read (finding, stance, context) or irrelevant
  papers/                 downloaded PDFs
  regulator_context.md    regulator biology (C2), interactions with the genes (C3), specificity across programs (C4),
                          STRING links among the regulators (C5)
  label.json, driver_genes.json (optional), evidence.md + evidence.json, interpretation.json
  _runs/<step>.json       run record of every step (who, when, output hashes, check errors)
  _runs/agent.json        the agent session (cost, turns, plan before / after), agent_transcript.jsonl
```

`plan.md` is changed only through `0.Prepare/program_dir.py` (`done`, `skip`, `show`). `done` runs the step's
checks (`1.Annotate/validate_program.py`) first and refuses to tick a step whose outputs do not pass — e.g. an
interpretation naming a gene, PMID or evidence id that is not in the program's evidence.

## Skills (`agent/`)

One folder per source, in Claude skill format (`SKILL.md` with `name` / `description`, `scripts/`, optional
`references/`). Each script reads one JSON object on stdin and writes one JSON object on stdout (`ok: true`, or
`ok: false` with `error.code`) and can append its results straight into a program file (`write_to`). Every skill is
self-contained: shared helpers are vendored into `scripts/_lib/` (`agent/_shared/sync_skill_lib.py` keeps the
copies in sync; its check mode reports stale ones).

| Skill | Source |
|---|---|
| program-annotator-skill | the orchestrator: how to work a program's plan; label / interpretation instructions and schemas |
| mygene-skill, ncbi-gene-skill, uniprot-skill | gene cards |
| string-skill, omnipath-skill | enrichment / partners, regulator-gene interactions |
| pubmed-skill, openalex-skill, crossref-skill, nature-skill, arxiv-skill | paper search |
| gene-function-papers-skill, generif-papers-skill | a gene's own function / GeneRIF papers |
| paper-fetch-skill | download, read and record a paper |
| wikipedia-skill | background only (never cited) |
| driver-gene-paperqa-skill | optional: paper-qa verdict per gene (OpenAI) |
| perturbseq-hypotheses-skill | optional, screen-wide hypothesis generation |

`run_program_agent.py` links the skills into `<run>/.claude/skills/`; the agents (Claude Agent SDK, cwd = run
folder) and an interactive Claude Code session opened in the run folder ("annotate P10 with the
program-annotator-skill") see the same skills. Keys go in `C_Annotation/.env` (`ANTHROPIC_API_KEY`,
`OPENALEX_API_KEY`, `OPENAI_API_KEY` for paper-qa, optional `NCBI_API_KEY`).

## Sandbox

Each agent runs inside a Bubblewrap sandbox (`1.Annotate/sandbox.py`, on by default; needs `bwrap` on PATH or
`BWRAP=/path/to/bwrap`). The SDK is pointed at a per-program wrapper of the Claude CLI, so the agent and everything it
runs see only: the OS, the Python env, `C_Annotation/{agent, 0.Prepare, 1.Annotate, api}`, the run's skills and
`Programs/meta.json`, its own program folder (the only writable path), and `regulators_sig.tsv` + `label.json` of the
other programs. `C_Annotation/.env`, other runs, other programs' outputs and the real HOME are invisible; keys reach
the agent as environment variables from an allowlist; the network stays on for the skills. The wrapper and its spec
live in `<run>/.sandbox/P<k>/`, outside what the agent can see. Before any model call a canary checks all of this
inside the sandbox and skips the program if anything leaks (`_runs/agent.json` records the result).
`1.Annotate/sandbox.py` runs the canary by hand for one program folder (its check mode).
