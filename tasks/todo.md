# RegulatorGroupAnnotator — worklog

Plan: `~/.claude/plans/we-just-updated-the-temporal-spring.md` (approved 2026-09-25).

Objective: annotate groups of co-regulating CRISPRi targets (similar effect profiles across programs)
with the V3 recipe (deterministic evidence → tool-less `claude -p` → gates → citation pass → viewer),
sharing code with ProgramAnnotatorV3 through `C_Annotation/annotator_core/`.

Constraints / decisions
- Shared core + two apps; literature same as V3 (gated PMID pool + citation pass, no live search).
- Grouping must not be a single global correlation cut (weak-effect groups are as coherent).
- Promoter-confounded members are flagged and excluded from annotation.
- No new heavy deps: the group clustering is scipy only (SNN similarity + average linkage +
  bootstrap stability) instead of igraph/leidenalg, which the NMF_Benchmarking env lacks.
- Group ids are integers so dispatch dirs stay `<arm>_p<N>` and every core script works unchanged
  (`p` = prompt unit).
- Test data: CC-Perturb-seq k50 (D0–D3 CRT, 298 targets), `annotation-prompt-benchmark/data/cc_k50`.

## Step 0 — annotator_core refactor
- [x] baseline V3 outputs (prompts, viewer, screens, citation prompts) in scratchpad
- [x] move dispatch + PMID gates + citation pass to `annotator_core/`; `--subject` for citation prompts
- [x] shared `answer_io`, `http_cache`, `gene_coordinates`, `viewer_common`, `gate_rules`
- [x] V3 byte-identical: prompts, viewer, screens, citation prompts, gate output (CC run + broken answers)
- [x] V3 README / skill ref paths updated; committed (0f8a655)

## Steps 1–6 — RegulatorGroupAnnotator
- [x] build_regulator_effect_matrix.py (noise model from p-values)
- [x] define_regulator_groups.py (disattenuated r, permutation gate, SNN, bootstrap, calibration, rescue)
- [x] measure_neighbour_knockdown.py + build_promoter_confound_screen.py (assembly check, shared-locus tie-break)
- [x] build_group_evidence.py (signature w/ V3 labels, STRING vs screened background, complexes, pool)
- [x] build_group_prompts.py + validate_group_answers.py
- [x] build_group_viewer.py (subagent; 0 gate failures on 5 groups)
- [x] tests: 15 pass (synthetic grouping, promoter geometry, gate)
- [x] real run CC k50: calibration table; 5 groups dispatched, all pass gate
- [x] citation pass on groups 5, 15: gate 0 problems, 7 PMIDs verified, 0 defects; viewer rebuilt with support
- [x] docs: README, 05-annotation-summary.md, SKILL.md, CLAUDE.md mermaid; drift check clean

## Evidence
- Calibration (CC k50, 192 co-complex pairs): SNN k5/0.7 recall 0.20 @ precision 0.64 vs GPE global
  cut 0.5 recall 0.10 @ 0.63; weak-tier 0.08 vs 0.02. Disattenuation: precision 0.64 vs 0.49 raw.
- Promoter screen: 4 excluded (C5orf22→DROSHA, CFAP119→RNF40, RIBC1→SMC1A, GOLT1B→RECQL), 16 flagged;
  22/26 measurable neighbours knocked down. CC guide names are hg19 vs hg38 coordinates → TSS fallback.

## Review
- Built and verified end to end on CC k50 (5 groups annotated, 2 with citation pass). V3 unchanged.
- Open: first-pass pool is thin (co-mention only) → members like TAOK1 under-called until the
  citation pass; gate cannot catch coherence talk without keywords; drift-check script is
  hard-coded to the Oak path (no-op locally).

---

# Annotation viewer — Olga's feedback (#perturbnmf-paper, 2026-09-25)

Decisions (Jesse): comparators stay in the benchmark viewers only, not in the package; drop
family/distinguisher from the page; no separate citation list — citation-pass PMIDs stay next to
their genes; regenerate the two posted viewers.

- [x] family/distinguisher header + distinguisher-evidence line removed (both viewers)
- [x] distinctive genes: same 30 the prompt showed, uniqueness score defined, rank + n on hover
- [x] modules defined on the page; "competing readings" -> "Alternative program annotations"
      (group viewer: "Alternative annotations"); JSON key unchanged
- [x] package viewer: dead comparator + citation-list code removed; unused .cmp CSS dropped
- [x] benchmark viewer: comparator boxes name + link + describe their source; citations card removed
- [x] regenerated annotation_viewer_{cc_k50,telohaec_k60}_v3.html; checked in browser, no console errors
- [x] 17 tests pass; package viewer builds on teloHAEC and CC
