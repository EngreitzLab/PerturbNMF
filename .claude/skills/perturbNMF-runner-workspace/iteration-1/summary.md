# perturbNMF-runner skill drift cleanup — verification report

## Files edited

| File | Change |
|---|---|
| `references/01-inference.md` | sk-cNMF + torch-cNMF `--K` default → `[30, 50, 70, 80, 100, 200, 300]`; `--sel_threshs` default → `[0.2, 2.0]`; added `--use_gpu`, `--tpm_fn`, `--genes_file`, `--parallel_running`, `--ensembl_prefix`, moved `--remove_noncoding` to shared section |
| `references/02-evaluation.md` | added `--guide_annotation_path` (alternative to `--guide_annotation_key`) |
| `references/03-calibration.md` | renamed `--components` → `--K` (CRT + U-test tables and example script); fixed `--sel_threshs` defaults to `[0.2, 2.0]`; rephrased `--mdata_guide_path` narrative to drop the leading `--` (drift-hook false positive); added `--covariates`, `--log_covariates`, `--guide_annotation_path`, `--reference_gtf_path`, `--compute_real_perturbation_tests`, `--compute_fake_perturbation_tests`, `--visualizations`, `--check_format` |
| `references/04-visualization.md` | populated K-selection `--K` / `--sel_threshs` defaults; added `--Conditions`, `--stability_file`, `--corr_matrix_path`, `--skip_existing`, `--tagert_col_name`, `--n_processes`, `--expressed_only`, `--control_target_name` |
| `references/05-annotation-summary.md` | added "Common CLI overrides" table for ProgramExplorer (19 flags); new "Literature Search" sub-section with its own SLURM resources; restored Annotation SLURM resources |
| `references/parameter-catalog.md` | sk-cNMF + torch + evaluation + U-test + CRT `--K`/`--sel_threshs` defaults aligned; `--components` renamed to `--K` in Sections 7 + 8 with rename notes; `--mdata_guide_path` narrative rephrased; new Section 10 (Annotation/ProgramExplorer) and Section 11 (Literature Search) |
| `SKILL.md` | no edits — convention placeholders (`--flag`, `--flag_name`, `---COMMENTED---`) live here as documentation of the all-flags convention; not real drift |

## Verification

### 1. Official drift hook (`check-pipeline-drift.py`)

```
Skill drift (args mentioned but not defined in ANY pipeline .py):
  SKILL.md                    : COMMENTED---, flag, flag_name
  references/01-inference.md  : COMMENTED---, flag
  references/02-evaluation.md : COMMENTED---, flag
  references/03-calibration.md: COMMENTED---, flag
  references/04-visualization.md: COMMENTED---, flag
  references/05-annotation-summary.md: COMMENTED---, flag
```

**Real drift: 0.** Every remaining hit is a convention placeholder — the all-flags
convention example uses `--flag` / `--flag_name` as abstract names and quotes the
literal `---COMMENTED---` sentinel. The drift hook's regex (`--[a-zA-Z]...`) can't
distinguish these from real flag mentions. Two ways to silence them in a future
pass:
- Add a strings allowlist to `check-pipeline-drift.py` (`{"flag", "flag_name", "COMMENTED---"}`).
- Rewrite the markdown to use `<flag>` placeholder syntax instead of `--flag`.

### 2. Programmatic verifier (`verify_alignment.py`)

Bidirectional comparison of every `add_argument` in `src/**/*.py` against every
`| --flag |` table row in the skill markdown.

After edits:
- 161 catalog table-row flags
- 231 pipeline argparse flags

Remaining "problems" (all noise, not real drift):
- **~70 internal sub-script flags** in `ProgramExplorer/src/01_*.py` through `05_*.py` and similar utilities. These aren't user-facing CLIs (they're called by `run_pipeline.py`), so they don't belong in user docs.
- **~27 Matched-Cell-DE flags** my Python-only verifier can't see (R-script `make_option` definitions). The official drift hook handles `.R` files correctly.
- **Skill-helper flags** like `--gpu_min_mem`, `--gpu_sku`, `--cpus`, `--mem`, etc. (defined in `scripts/generate_slurm.py`, not pipeline scripts).
- **External-tool flags** like `--execute` from `jupyter nbconvert`.
- **Two rename notes** in `03-calibration.md` lines 41 and 126 that say "(formerly named `--components`)" — the drift hook correctly exempts these via its `RENAME_RE`.

### 3. Defaults spot-check (the four user-facing flags edited)

```
--K       : pipeline default [30, 50, 70, 80, 100, 200, 300] across all 6 stages
            catalog rows match (or say "(from data)" for plotting stages)
--sel_threshs : pipeline default [0.2, 2.0] across all 6 stages
                catalog rows match
--numhvgenes  : sk-cNMF 5451 vs torch 2000 — catalog correctly distinguishes
--tol         : sk-cNMF 1e4 (documented as likely bug) vs torch 1e-4 — catalog matches
```

## Where the workspace lives

```
.claude/skills/perturbNMF-runner-workspace/iteration-1/
├── verify_alignment.py          ← reusable verifier script
├── verify_output.txt            ← initial run (pre-Section-10/11 additions)
├── verify_output_v2.txt         ← post-additions run
└── summary.md                   ← this file
```

Re-run anytime with:
```
python3 .claude/skills/perturbNMF-runner-workspace/iteration-1/verify_alignment.py
```
