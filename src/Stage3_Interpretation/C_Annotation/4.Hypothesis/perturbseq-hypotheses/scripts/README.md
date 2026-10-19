# Reference implementations — adapt, don't run blind

These scripts encode decisions that were expensive to learn. They do **not** encode any
particular run's file layout, and no PerturbNMF run has quite the same one.

**How to use them:** read the script, copy it into the analysis directory, adapt the I/O to this
run, and commit the adapted copy there. The analysis repo's own `pixi` environment runs it — this
skill deliberately ships no environment, because a number produced by a different environment
from the rest of the analysis is a number you cannot defend.

Dependencies: `pandas`, `numpy`, `matplotlib`, `requests`. `verify_citations.py` needs network.
The validator, lint and leak check are stdlib-only and run as-is (they are checks, not analysis).

| Script | What it encodes |
|---|---|
| `build_candidate_table.py` | The Explorer's regulator filter (was v1's admission gate): the run's own test, ≥2 conditions, thresholded against the matched NTC arm — plus the promoter-suffix collapse and its fail-loud guards |
| `check_cis_neighbours.py` | The on-target check. No locus-level claim ships without this |
| `perturbation_neighbours.py` | Effect-vector correlation → ± top-k neighbours. The highest-value axis for an understudied gene |
| `verify_citations.py` | Every DOI / PMID / NCT in a report, resolved and compared against what the report says |
| `plot_hypothesis_evidence.py` | The Analyst's per-hypothesis plots: effects across programs, program top genes, perturbation neighbours, per-guide agreement, the overview map. `--self-test DIR` runs on synthetic data |
| `screen_io_template.py` | The Explorer adapts it into `src/screen_io.py`: one loader per tidy table (effects, null arm, guides, loadings, correlation), with the significance column, effect column and technical programs decided once. `--check` prints what each returns. Every Analyst imports it |
| `print_ledger_view.py` | **Run as-is.** A compact, role-shaped view of the ledger (`--ids`, `--status`, `--view brief/writer/full`). Agents read this instead of dumping the ledger, which grows to ~1 MB |
| `validate_hypothesis_ledger.py` | **Run as-is**, with `--round` (also with `--final`) and `--max-rounds` from `study.yaml :: loop.max_rounds`. Enforces the ledger rules: every round changes every open hypothesis, status matches the last Reviser action, survival is earned, tests are pre-registered |
| `lint_hypothesis_report.py` | **Run as-is.** Fails a report that narrates process (preamble included), buries the hypotheses, or ships an incomplete card. `--n-hypotheses` = supported entries; `--allow` for gene symbols shaped like internal codes |
| `check_for_leaked_terms.py` | **Run as-is.** Fails if a held-out term appears in text a blind run will read |
| `setup_sterile_run.sh` | **Run as-is.** Builds a sterile directory for a blind run and prints the launch command and context audit |

What none of them do, on purpose: compute a composite score, gate on knockdown efficiency, or
invent a null. See `../references/failure-modes.md`.
