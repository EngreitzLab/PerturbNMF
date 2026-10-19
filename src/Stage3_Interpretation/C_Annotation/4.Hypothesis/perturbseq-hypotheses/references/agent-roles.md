# Agent roles — who does what, and what each one is allowed to see

The loop is a small team of sub-agents that talk **only through the ledger**
(`ledger/hypotheses.jsonl`, schema in `ledger-schema.md`) and the files it points to. No agent reads
another agent's transcript. The orchestrator (the main session) dispatches agents, runs
`scripts/validate_hypothesis_ledger.py` after every step, and decides when to stop.

Why this shape: see `DESIGN.md` §2. In short, it borrows the Kosmos world model (shared structured
state, every claim linked to code or a paper), POPPER's falsification tests, and co-scientist's
evolve-or-die step. It also fixes v1's two biggest failures: literature became a label instead of a
filter, and the report narrated the process.

| Role | Sees | Never sees | Writes |
|---|---|---|---|
| Explorer | files in `inputs/MANIFEST.txt`, `study.yaml` | literature | `explore/` |
| Generator | `explore/` summaries, `study.yaml` questions, `evidence-classes.md`, ledger (after round 1) | literature, traces | new ledger entries, with `tags` |
| Falsifier | its ledger entries, `explore/` file list | other hypotheses' results | `tests[]` on that entry |
| Analyst | its tests' specs, `src/screen_io.py`, files in `inputs/MANIFEST.txt`, external datasets | literature verdicts, other hypotheses' results | `traces/<id>/<test>/`, test `result` |
| Literature | its entries' claims + predictions | data plots, traces | `literature/<id>_r<n>.md`, `literature[]` |
| Reviser | its entries: tests, results, literature | other hypotheses (except to merge) | new claim version, `history[]` |
| Critic | surviving entries + their headline figures | transcripts | `critique[]` on each entry |
| Ranker | surviving entries (incl. `tags`) | transcripts | `ratings`, `rank`, `top_pick` |
| Writer | `supported` entries, their headline figures, `explore/program_glossary.tsv`, `study.yaml`, `report/killed_extract.tsv` | transcripts, `history[]`, killed entries beyond the extract | `report/`, `report/cartoons/` |

Each role's brief is its own file in `briefs/` (`briefs/explorer.md` … `briefs/writer.md`). The
sub-agent reads its brief itself; the orchestrator never copies a brief into a prompt.

## Dispatching

**The prompt is short.** Everything else is in the brief and the ledger:

```
You are the <Role> in a perturbseq-hypotheses run. Skill: <skill path>. Analysis dir: <path>.
Read <skill path>/references/briefs/<role>.md and follow it.
Round <r>. Your entries: <ids>. [Tests: <test ids>.] [N = <n>, candidates = <m>.]
```

Do not paste summaries of earlier steps, findings, or file contents. The ledger is the only memory.

**Units and batch sizes.** Agent cost grows with turns × context, so several short agents cost
less than one long one. Size each so it finishes in about 40 turns.

| Role | One agent per | Concurrent | Model |
|---|---|---|---|
| Explorer, Generator | the run | 1 | opus |
| Falsifier, Reviser | ≤ 6 entries | ≤ 5 | opus |
| Literature | ≤ 6 entries that need a check (no check since the last version bump) | ≤ 4 | opus |
| Analyst | 1 entry (all its open tests), or ≤ 6 tests | **≤ 4** | opus |
| Analyst rerun | ≤ 6 tests it did not run first | ≤ 4 | opus |
| Critic, Ranker, Writer | the run | 1 | opus |
| Reviewers (step 9) | as `/engreitzlab-review` dispatches them | — | opus |
| Mechanical: applying listed review fixes, citation verification, lint or figure-craft fixes, path fixes | one report section or one fix list | ≤ 3 | **sonnet** |

The concurrency caps matter: a usage limit that stops ten parallel Analysts at once loses all
of their unfinished work. Judgement roles stay on Opus; only work whose instructions fully
determine the result goes to Sonnet (set `model: "sonnet"` on the Agent call).

**Hand-backs are at most five lines** (what was written, which ids, anything blocking). If an
agent returns more, do not echo it into later prompts.

**Resuming.** Before re-dispatching after an interruption, check what is on disk: an Analyst test
is done when `traces/<id>/<test>/result.md` exists, a Literature check when
`literature/<id>_r<round>.md` exists. Dispatch only the unfinished units.

**Fix agents edit by section.** An agent applying review findings to the report greps for the
lines it changes and edits them in place; it does not re-read the whole report after each fix.
Split long fix lists by report section, one agent each.

## Optional: human steer after round 1

If `study.yaml :: loop.human_steer_after_round_1` is `true`, stop after the round-1 Falsifier step
and show the user one line per candidate: title, claim, the test that could kill it. The user
marks each as kill, push, or leave. Record each mark as a `history[]` event
`{"round": 1, "action": "user_kill" | "user_push", "actor": "user", "reason": …}`; leave needs no
event. User events are not Reviser actions: every candidate still needs its own Reviser action at
the end of round 1, and a `user_kill` makes that action `killed`. Then continue.
