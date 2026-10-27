# Ledger schema — `ledger/hypotheses.jsonl`

One JSON object per line, one line per hypothesis, rewritten in place after every step. After
each step the orchestrator copies the whole file to `ledger/snapshots/r<round>_<step>.jsonl`
(e.g. `r2_falsifier.jsonl`). The validator uses the snapshots to prove that tests were registered
before they were run.

`scripts/validate_hypothesis_ledger.py` enforces everything marked **(checked)**.

```json
{
  "id": "H07",
  "version": 2,
  "status": "supported",
  "title": "VHL restrains the hypoxia program",
  "claim": "VHL marks HIF-α for degradation in these cells, so losing VHL raises the hypoxia-response program; the prolyl hydroxylase EGLN1 acts upstream and should phenocopy it.",
  "mechanism": "One sentence.",
  "why_investigate": "If true, these cells keep the hypoxia program switched off at normal oxygen through active degradation rather than low transcription, which makes the program a sensitive readout of oxygen and of any drug that blocks the hydroxylase. For a study asking what sets the cells' metabolic state, it would name the switch that gates the move to glycolysis.",
  "members": ["VHL"],
  "tags": ["metabolic_state"],
  "programs": ["Hypoxia response — CA9"],
  "study_question": "Which regulators set the cells' metabolic state?",
  "definitions": {"prolyl hydroxylase": "an oxygen-dependent enzyme that tags HIF-α so VHL can bind it"},
  "predictions": [
    {"id": "P1", "statement": "EGLN1 knockdown also raises the hypoxia program; HIF1A knockdown lowers it."}
  ],
  "tests": [
    {
      "id": "T1",
      "prediction": "P1",
      "kind": "data",
      "counterexample": true,
      "measures": "log2FC of the hypoxia program for EGLN1 and HIF1A",
      "kill_if": "EGLN1 knockdown lowers the program, or HIF1A knockdown raises it",
      "registered_round": 2,
      "hypothesis_version": 2,
      "result": "supports",
      "result_round": 2,
      "summary": "EGLN1 +0.9 (q=0.002); HIF1A −0.6 (q=0.03).",
      "trace": "traces/H07/T1/",
      "headline_figure": "traces/H07/T1/figure.png"
    }
  ],
  "cartoon": "report/cartoons/H07_vhl.svg",
  "literature": [
    {
      "round": 2,
      "verdict": "known_elsewhere",
      "summary": "The VHL–HIF axis is textbook; its effect on this program in this cell type is not reported.",
      "citations": [{"id": "PMID:10353251", "says": "VHL targets HIF-α for oxygen-dependent proteolysis"}],
      "notes": "literature/H07_r2.md"
    }
  ],
  "history": [
    {"round": 1, "action": "created", "actor": "generator"},
    {"round": 1, "action": "user_push", "actor": "user", "reason": "hypoxia is central to this system"},
    {"round": 1, "action": "sharpened", "actor": "reviser", "reason": "the axis is textbook; the new part is which program it controls here", "previous_claim": "VHL loss moves twelve programs."},
    {"round": 2, "action": "survived", "actor": "reviser", "reason": "T1 counter-example supports; literature r2 finds no report in this cell type"}
  ],
  "critique": [{"severity": "minor", "issue": "define 'hypoxia-response program' by its top genes", "resolved": true}],
  "ratings": {"interest": 3, "support": 2, "tractability": 3},
  "rank": 4,
  "top_pick": false,
  "confidence": "medium",
  "experiment": {
    "design": "CRISPRi of VHL, EGLN1 and HIF1A; hypoxia program score plus HIF-1α western, at 21% and 1% oxygen.",
    "if_true": "VHL and EGLN1 raise the program at 21% oxygen, HIF1A lowers it, and the differences shrink at 1%",
    "if_false": "only VHL moves it: a VHL-specific effect, not the HIF axis"
  },
  "kill_reason": null,
  "merged_into": null
}
```

Enumerations used above: `status` ∈ `candidate | supported | killed | merged | shelved`; test
`kind` ∈ `data | external_data | literature`; `result` ∈ `supports | refutes | inconclusive |
not_run`; literature `verdict` ∈ `known_in_this_cell_type | known_elsewhere | novel |
contradicted`; `confidence` ∈ `low | medium | high`; critique `severity` ∈ `major | minor`.

`tags` is a list, assigned by the Generator: names from `evidence-classes.md` (`control`,
`understudied`, `disease_locus`, `mirror`, `contrast`, `cross_program`) plus any study-specific
tag (the example's `metabolic_state`). The Ranker uses them to keep the top N diverse. Not checked; never shown in the report.

`history[]` events: `actor` ∈ `generator | reviser | user`. Reviser actions are `survived |
sharpened | reframed | merged | split | killed | shelved`; user marks are `user_kill | user_push`
(human steer only). A Critic-Major pass is a Reviser event with `"repass": true`.

## Rules

1. **Status matches the last Reviser action.** `survived` → `supported`; `killed` → `killed` with
   `kill_reason`; `merged` → `merged` with `merged_into`; `shelved` → `shelved`; `sharpened`,
   `reframed`, `split` → `candidate`. `revised` is not a status. **(checked)**
2. **Every round ends in an action.** For every entry still `candidate` at the start of round *r*,
   `history[]` has exactly one Reviser action with `round == r` (plus at most one `repass`). User
   events do not count. A round that only adds literature or results is rejected. **(checked)**
3. **`survived` must be earned**, judged on the **last** Reviser action. Every test registered
   against the current version has a result other than `not_run`; at least one
   `counterexample: true` test with `kind` `data` or `external_data` returned `supports`
   (`inconclusive` does not count); the literature has an entry with `round` at or after the
   round of the last version bump; and the latest verdict is not `known_in_this_cell_type`.
   **(checked)**
4. **Tests are pre-registered.** Every test with a result must appear, without a result, in an
   earlier snapshot. **(checked with `--snapshot-dir`)**
5. **Versions count revisions.** `version` = 1 + the number of `sharpened` and `reframed` events,
   and each of those events carries `previous_claim`. The new version needs at least one test with
   `hypothesis_version` equal to it before it can survive. **(checked)**
6. **Writer-eligible** (`--final`): `status == supported`; `why_investigate` has at least 25 words;
   `confidence`, `experiment.design`,
   `if_true` and `if_false` are set; at least one data test has a `headline_figure` that exists;
   `cartoon` is set and its file exists
   on disk; no unresolved Major critique. **(checked)**
7. **Iteration health** (`--final`): the share of entries that were ever sharpened, reframed,
   merged, split, killed or shelved. A warning is printed below 30%, meaning the loop was
   annotating rather than iterating. **(checked, warning)**
8. **Budget.** At most `--max-rounds` Reviser actions per entry (pass `loop.max_rounds`), plus
   one `repass`. **(checked)**
