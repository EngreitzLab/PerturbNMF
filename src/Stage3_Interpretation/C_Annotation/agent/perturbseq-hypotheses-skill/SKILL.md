---
name: perturbseq-hypotheses
description: >
  Turn a finished Perturb-seq / PerturbNMF screen (cNMF programs, per-perturbation effect tables,
  NTC null, regulator correlation matrix, enrichment tables) into a short, curated report of the
  top N (default 10) researched biological hypotheses. Each is a mechanistic claim that survived
  falsification tests on the data, a literature check and at least one round of revision, and each
  comes with plots, integrated literature and a concrete experiment. It runs as an iterative
  multi-agent loop (explore → generate → design falsification tests → analyse → literature →
  revise or kill → critique → rank → write) over a shared hypothesis ledger. The report leads with
  the hypotheses and carries no process narrative. Trigger phrases: "prioritize the hypotheses from
  this screen", "top 10 hypotheses", "most interesting hypotheses", "what are the best candidates
  from this Perturb-seq data", "nominate targets from the screen", "hypothesis report",
  "/perturbseq-hypotheses". Do NOT trigger for: a general analysis report (/engreitzlab-report),
  reviewing code or an analysis (/engreitzlab-review), program wiring diagrams
  (GeneProgramExplorer), or annotating a single program.
---

# Perturb-seq hypotheses

Produce the **top N hypotheses a screen supports**, where each one has been proposed, tested
against the data, checked against the literature, and revised or killed until what is left is
worth an experiment. The deliverable is a list of hypotheses, not an account of how they were
found.

**Parameters** (in `study.yaml :: loop`, or given by the user):

- `n_hypotheses`: default **10**
- `effort`: `low` / `medium` / `high`, default `medium`. **Ask the user at the start, beside
  `study.yaml`.** It sets `candidates_per_hypothesis` unless that is given explicitly:

  | `effort` | `candidates_per_hypothesis` | Candidates for N = 10 | Use when |
  |---|---|---|---|
  | `low` | 2 | 20 | a quick first look |
  | `medium` | 3 | 30 | the default |
  | `high` | 9 | 90 | a screen worth a full run. Roughly 3× the agents and time of `medium` |

  About one candidate in ten survives the loop, so `medium` often ships fewer than N. Say so
  when asking.
- `max_rounds`: default 3
- `human_steer_after_round_1`: default false

> **Teaching examples in this skill must never name findings from a screen it will be run on.**
> An earlier version used a real lab screen as its worked example and contaminated a blind run
> on that screen through its own required reading. Use textbook biology. `setup_sterile_run.sh`
> enforces this with a held-out-terms check.

---

## How it works

A small team of sub-agents shares one state file, `ledger/hypotheses.jsonl`. Roles, briefs and
what each role may see are in `references/agent-roles.md`. The schema is in
`references/ledger-schema.md`.

```
Explorer ─► Generator ─► ┌──────────────── round r (≤ max_rounds) ────────────────┐
  (once)     (2–9N)      │ Falsifier → Analyst → Literature → Reviser              │
                         │   register     run      check the     survive / sharpen │
                         │   kill tests   tests    claim         reframe / merge / │
                         │                                       split / kill      │
                         └──── repeat on candidates still open ────────────────────┘
                                     │
                          Critic ─► Ranker ─► Writer ─► lint + review gate
                                                (cards + a cartoon each)
```

The orchestrator (you, in the main session) does five things:

1. Dispatches each role with the short prompt, batch sizes, concurrency caps and models in
   `agent-roles.md` "Dispatching". The sub-agent reads its own brief from `references/briefs/`.
   Never paste a brief, transcripts or your own summaries into a prompt; the ledger is the only
   shared memory.
2. After every step, snapshots the ledger to `ledger/snapshots/r<round>_<step>.jsonl`.
3. Runs the validator after every step and does not continue until it passes:
   `python scripts/validate_hypothesis_ledger.py ledger/hypotheses.jsonl --round <r> --max-rounds <loop.max_rounds> [--mid-round] --snapshot-dir ledger/snapshots --eligibility explore/regulator_eligibility.tsv`
   (the eligibility check also catches a Reviser reframe that adds an ineligible member)
4. After every step, overwrites `ledger/RUN_STATE.md`: round, step just finished, ids still open,
   units dispatched but not finished, next step. A fresh session (after a usage limit, a crash or
   `/compact`) resumes from this file and the files on disk (`agent-roles.md` "Resuming").
5. Decides when to stop.

**Keep your own context small**: it is re-read on every turn of a run that takes hundreds. Read
only this file, `agent-roles.md` and `ledger-schema.md`; each brief names the other references its
role needs, so do not read them yourself. Look at the ledger through
`scripts/print_ledger_view.py`, never by dumping it. Run only the steps below: no extra filters,
checks or passes.

### Steps

0. **Set up.**
   - For a blind or evaluation run, build a sterile directory first: `scripts/setup_sterile_run.sh`
     (see "Clean runs" below).
   - Otherwise work in `analysis/<name>/` of the analysis repo, with this layout: `inputs/`,
     `explore/`, `ledger/`, `traces/`, `literature/`, `report/`, `src/`. `inputs/MANIFEST.txt`
     lists every input the run may read, one absolute path per line (in a sterile run the setup
     script writes it, as names under `inputs/`). The Explorer and Analyst read only what it lists.
   - Give the analysis a pixi environment (`pandas`, `numpy`, `matplotlib`, `scipy`, plus
     `anndata`/`mudata` if per-cell files are used). List both `linux-64` (e.g. an HPC cluster) and
     `osx-arm64` as platforms if it will run in both places.
   - Copy `study_template.yaml` to `study.yaml`, fill it from the repo's README / spec, and **show
     it to the user before anything runs**, with the `effort` question (low / medium / high,
     with the candidate counts and relative cost above). The rule is design facts in, biological conclusions
     out (`references/novelty.md` §5).
1. **Explore** (once). The Explorer applies `references/data-hygiene.md` and writes the
   `explore/` summaries: program glossary, regulator summaries, co-effect modules, cell-type
   specificity, and the eligibility table. The eligibility table records the cis-neighbour,
   per-guide and expression checks for **every** admitted regulator. Where a
   RegulatorGroupAnnotator run exists for the screen, the Explorer also reads its per-group
   annotations into `explore/regulator_groups.tsv` and `explore/unexplained_members.tsv`. The
   Generator reads only these files, so they must be good. The Explorer also writes the shared
   loaders every Analyst imports (`src/screen_io.py`, from `scripts/screen_io_template.py`).
2. **Generate.** About `candidates_per_hypothesis × N` mechanistic candidates. They must be
   claims, not observations, and favour this cell type's biology over generic core machinery.
   Where regulator-group annotations exist, every `unexplained` member and every `strong`-coherence
   group is worked through as a seed (`evidence-classes.md`).
   No member may be `lead_eligible = no`. Validate with
   `--round 0 --eligibility explore/regulator_eligibility.tsv` and send failures back.
3. **Round loop**, for every candidate still open:
   1. The **Falsifier** registers 2–4 kill tests, at least one of them a counter-example. Snapshot.
   2. The **Analyst** runs each test into `traces/<id>/<test>/` (code, figure, stats, one-line
      result), one agent per entry. Snapshot.
   3. **Rerun.** For each entry that could survive this round (every current-version test has a
      result and a data counter-example test returned `supports`), a fresh Analyst reruns that
      counter-example test from its spec (`loop.analyst_reruns` runs in all). A mismatch makes
      it `inconclusive`. Nothing else is rerun. Snapshot.
   4. **Literature** searches for the claim itself, not the gene names, for entries with no
      literature check since their last version bump. Snapshot.
   5. The **Reviser** takes exactly one action per candidate. A verdict alone is not an action.
      A reframed claim replaces the old one. Snapshot, then validate without `--mid-round`.
   6. If `human_steer_after_round_1` is set, stop after round 1's Falsifier step for the user's
      kill / push marks, recorded as `user_kill` / `user_push` events (`agent-roles.md`, last
      section).
   The orchestrator runs only the steps listed here. Do not add extra filters (e.g. a separate
   "prior-art check" that kills entries): prior art enters through the Literature agent and the
   Reviser, under `references/novelty.md` §1b.
4. **Stop** when the first of these happens:
   - `supported` reaches N and no open candidate outranks the weakest survivor on interest;
   - the survivor list is unchanged across two rounds;
   - `max_rounds` is reached, at which point open candidates are `shelved`.

   If survivors fall short of N, the Generator may top up once. Otherwise report what survived:
   **never pad to N**.
5. **Critique.** The Critic checks every survivor for overclaiming, undefined terms and a vague
   experiment. A Major sends it back to the Reviser for one more pass, recorded with
   `"repass": true` (one allowed beyond `max_rounds`).
6. **Rank.** Interest, support and tractability (1–3 each), then pairwise ordering, using the
   entries' `tags` to keep the top N diverse. Pick the top 3 for an experiment separately.
7. **Draw a cartoon per survivor**, following `references/cartoons.md`: a hand-written SVG in two
   rows, known mechanism above and proposed mechanism below, with the claim's contrast visible as
   geometry. Saved as `report/cartoons/<id>_<member>.svg` and recorded as `cartoon` on the entry.
   Look at every one rendered before moving on.
8. **Write** to `references/report-template.md`. The Writer sees `supported` entries, their
   headline figures, the program glossary, `study.yaml` (subtitle, sign convention, questions,
   caveats) and a killed-entries extract (id, title, `kill_reason`, key citation) that the
   orchestrator writes to `report/killed_extract.tsv`. The extract feeds the appendix and the
   "M candidates → N survivors" line in Methods.
9. **Verify.**
   - `validate_hypothesis_ledger.py ledger/hypotheses.jsonl --final --round <last round> --max-rounds <loop.max_rounds> --n-hypotheses N --analysis-dir .`
   - `lint_hypothesis_report.py report/<file>.qmd --n-hypotheses <number of supported entries in the ledger>`.
     Gene symbols that look like internal codes (`C3`, `F8`) beyond the default list go in
     `--allow`.
   - `verify_citations.py`
   - `/engreitzlab-review` scoped to the report, traces and figures, plus the two skill reviewers
     (`references/novelty-and-priority.md`, `references/claim-traceability.md`), each dispatched
     as a `general-purpose` agent with the checklist path.
   - `render_report.py --strict` from `analysis-report`.
   - Fix everything, then show the user. Fixes go to Sonnet agents, one report section each,
     editing lines in place (`agent-roles.md` "Dispatching").
10. **Record** (non-blind runs only): see "Recording" below.

---

## What makes a good hypothesis here

- **A mechanism with a prediction.** "Loss of X raises program Y because X normally does Z, so
  knocking down X's partner W should do the same" — not "X moves 16 programs".
- **About this cell type.** A generic stress response of the proteasome is a positive control,
  not a top-N hypothesis, unless it has a cell-type-specific angle. Positive controls go in
  `explore/positive_controls.tsv` and are mentioned once in Methods.
- **Internally replicated where possible**: complex partners agree; guides agree; the effect is
  absent from control guides; it is specific to this cell type compared with another screen.
- **Surprises get a sanity check before promotion**: per-guide plot, knockdown, cis neighbours,
  cell count. A strange knockdown value (negative or >100%) means the claim is about the locus,
  not the protein.
- **What is new is said precisely.** When known biology explains part of it, the claim becomes
  the remainder (the cell type, the sign, the scale). If nothing remains, it dies.

## Clean runs

For any run whose output will be scored against known answers, or that must be blind to prior
work on the same screen:

1. Write the held-out terms (answer-key genes, the predecessor paper's title, named pathways) to a
   file **outside** any run directory.
2. Write an inputs manifest: only the data files. No prior analyses, no predecessor program labels.
   If program labels are needed, the Explorer derives them.
3. Run `scripts/setup_sterile_run.sh --run-dir ~/ph-sterile/<run> --inputs <manifest> --terms <file>`.
   It builds a fresh directory outside `~/Claude` with its own git history, vendors a scrubbed
   copy of this skill (no `examples/`), writes settings that turn off auto-memory and deny reads
   of `~/Claude/projects` and `~/.claude/projects`, and fails if any held-out term appears in the
   text the run will read.
4. Launch `claude` from that directory in a new terminal and run the printed context audit first.
5. **Scoring happens elsewhere, afterwards**, in a separate session that reads the frozen report
   and the key. See `references/blind-evaluation.md`. Nothing about blindness, leaks or scores ever
   appears in the report.

The literature agent **may** find and use prior papers on the same screen. The goal is new
hypotheses beyond them, and blocking prior art would only make the novelty calls worse.

## Recording (non-blind runs)

If the repo is a mycelium living repo: `.living/` decisions and findings, `ANALYSIS_MANIFEST.md`,
one pointer line in the EngreitzLabVault project note (paths only, never results). If not, add a
project-type pointer note by hand via `/vault-ingest`: `sensitivity: private` for unpublished
data, point at the `.qmd` and figure files rather than `.html`. Then register the repo in the
mycelium registry.

---

## Quick reference

| Need | Where |
|---|---|
| Roles, context boundaries, dispatch (batch sizes, concurrency, models, resume) | `references/agent-roles.md` |
| Each role's brief (the sub-agent reads its own) | `references/briefs/<role>.md` |
| Looking at the ledger without dumping it | `scripts/print_ledger_view.py` |
| Ledger fields and rules | `references/ledger-schema.md` |
| Reading the run correctly (significance column, aliases, kill-checks, program identity) | `references/data-hygiene.md` |
| Report outline and card | `references/report-template.md` |
| Mechanism cartoons: structure, vocabulary, what may not go on one | `references/cartoons.md` |
| What counts as novel; the literature rules | `references/novelty.md` |
| Voice | `references/voice.md` — do **not** use `/grug-voice` |
| Useful tags for spreading candidates across kinds of hypothesis | `references/evidence-classes.md` |
| What went wrong before | `references/failure-modes.md` |
| Scoring a blind run | `references/blind-evaluation.md` |
| Reference code | `scripts/` (see `scripts/README.md`); adapt into `src/`, commit the copy |
| Why it is built this way | `DESIGN.md` |
