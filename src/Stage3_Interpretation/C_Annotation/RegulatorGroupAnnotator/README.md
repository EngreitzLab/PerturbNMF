# RegulatorGroupAnnotator — groups of co-regulating perturbations, annotated like programs

Finds groups of perturbed genes whose CRISPRi knockdowns shift the gene programs the same way, then
annotates each group with the ProgramAnnotatorV3 recipe:

1. deterministic evidence;
2. one blinded, tool-less `claude -p` call per group, with citations selected from a pool;
3. gates;
4. the shared citation pass;
5. a self-contained HTML viewer.

Each group gets:
- a label;
- the function its members share;
- why that function shows up in this system, read through the programs it moves;
- a role for every member. Members with no evident link to the others are flagged as
  hypotheses, not explained away.

Members whose membership a neighbouring promoter could explain are excluded before annotation.

The dispatch, PMID gates and citation pass are the same code as ProgramAnnotatorV3, in
`../annotator_core/`. A fix there lands in both annotators.

## How groups are defined

A single global cut on effect-profile correlation (GeneProgramExplorer: 1 − r at 0.5) favours
strong regulators. Noise attenuates correlation, so equally related weak regulators fall below the
cut. `define_regulator_groups.py` uses four steps instead:

1. **Noise correction.** Each regulator gets a reliability: the share of its profile's variance
   that is signal, with the standard error taken from its p-values. Correlations are
   disattenuated: r* = r / √(rel₁·rel₂).
2. **Pair significance.** Each pair is tested against a program-permutation null, with BH
   correction across pairs. This is a significance gate, not a magnitude cut.
3. **Local consistency.** Regulators are clustered on shared nearest neighbours of r*, so a weak
   regulator is judged against its own neighbours.
4. **Stability.** Bootstrapping over programs gives each member a stability. Core members are at
   ≥ 0.3; the rest are peripheral.

Curated complexes (CORUM / ComplexPortal / SIGNOR via OmniPath) are used in two ways:
- **Calibration** (`--calibrate`). Tune `--neighbors` / `--snn-cut` on co-complex pair recovery.
  This is partly circular, so it is recorded in the output.
- **Rescue.** A complex partner of ≥ 2 core members that correlates significantly with the group
  centroid is added with role `rescued`.

On the CC-Perturb-seq k50 screen (238 eligible regulators, 192 co-complex pairs), at the same
precision (≈0.64), recall of co-complex pairs was:

| Method | Recall, all pairs | Recall, weak-tier pairs |
|---|---|---|
| GeneProgramExplorer global cut | 0.10 | 0.02 |
| Noise-corrected SNN (k=5, cut 0.7, the defaults) | 0.20 | 0.08 |

The noise correction alone raised precision at matched recall (0.64 vs 0.49).

## Promoter confounds

CRISPRi represses a window around the guide, so a guide for gene A can silence gene B next to it.
This is most often a divergent (bidirectional) pair. On the CC screen, 22 of the 26 measurable
promoter neighbours of grouped regulators were knocked down by the target's guides.

- `measure_neighbour_knockdown.py` measures, per condition, whether each neighbour is knocked down
  in the target's cells compared with non-targeting cells.
- `build_promoter_confound_screen.py` decides what to do with each member:
  - **Excluded:** the neighbour explains the group (it is a member, a complex subunit or a STRING
    partner of members), or the neighbour is knocked down and the target is not.
  - **Flagged:** otherwise.
  - **Shared locus:** two members that share a promoter are one locus. The better-supported one is
    kept.

Guide positions are used only when they sit at the target's TSS in the coordinate file. The CC
library names carry hg19 positions and the coordinate file is hg38, so the screen fell back to
TSSs.

## Requirements

- Python 3.10+ with `pandas`, `numpy`, `scipy`, `h5py`. `../pixi.toml` has the environment:
  `pixi run -m ../pixi.toml python ...`.
- The Claude Code CLI (`claude`), logged in (same as ProgramAnnotatorV3).
- Network for the evidence and citation steps: `string-db.org`, `mygene.info`,
  `www.ncbi.nlm.nih.gov` (PubTator3), `eutils.ncbi.nlm.nih.gov`, `omnipathdb.org` (once), plus the
  citation-pass hosts.
- **Run the network steps outside any sandbox.** Behind the Claude Code sandbox proxy, Python reads
  of large STRING responses come back truncated. They are retried, not cached, but most attempts
  fail.

## Inputs

These are the same data directory as ProgramAnnotatorV3, plus the h5mu.

| File | What | Source |
|---|---|---|
| regulators | `program_id, [condition,] target_gene, log2_fc, p_value, adj_pval, significant` | the V3 regulator table. **Needs the raw p-value column** for the noise model |
| targets | TSV with `target_name`: every perturbed gene | the guide library (the enrichment background) |
| gene coordinates | `name chrom start end strand gene_type`, no header | the alignment GTF (same file as V3) |
| h5mu | expression (`mod/rna`) + guide assignment (`mod/cNMF/obsm/guide_assignment`, `uns/guide_names`, `uns/guide_targets`) | the cNMF inference output. Optional: without it, promoter confounds are judged on distance alone |
| complexes | OmniPath complexes TSV | downloaded on first use to the path you give |
| program annotations (optional) | a ProgramAnnotatorV3 dispatch directory | labels for the programs in each group's effect signature |

## Run

```bash
cd src/Stage3_Interpretation/C_Annotation/RegulatorGroupAnnotator/scripts
CORE=../../annotator_core; export PYTHON=python
D=path/to/annotation_inputs; G=path/to/regulator_groups; C=my_group_config.json

# 1. effect profiles + noise model
$PYTHON build_regulator_effect_matrix.py --regulators $D/regulators_by_condition.csv \
    --conditions D0 D1 D2 D3 --output-dir $G
# 2. groups (add --calibrate once to see the complex-recovery grid, grouping_calibration.tsv)
$PYTHON define_regulator_groups.py --matrix-dir $G --complexes $D/omnipath_complexes.tsv --output-dir $G
# 3. promoter confounds
$PYTHON measure_neighbour_knockdown.py --groups $G/regulator_groups.json \
    --gene-coordinates $D/gene_coordinates.tsv --h5mu cNMF.h5mu --condition-key day --output $G/knockdown.tsv
$PYTHON build_promoter_confound_screen.py --groups $G/regulator_groups.json \
    --gene-coordinates $D/gene_coordinates.tsv --knockdown $G/knockdown.tsv \
    --complexes $D/omnipath_complexes.tsv --string-cache $G/cache --output $G/promoter_confounds.json
# 4. evidence (network; cached)
$PYTHON build_group_evidence.py --groups-dir $G --targets $D/targets.tsv \
    --complexes $D/omnipath_complexes.tsv --cache-dir $G/cache \
    --program-annotations path/to/v3/dispatch --program-arm v3 --output-dir $G
# 5. prompts, blinded dispatch (read dispatch_groups/DISPATCH_STATUS, not the exit code), gate
$PYTHON build_group_prompts.py --config $C --output group_batch.json
$PYTHON $CORE/split_prompts_for_blinded_dispatch.py --batch group_batch.json --arm rg --dispatch-root dispatch_groups
nohup bash $CORE/dispatch_until_complete.sh dispatch_groups "rg_p*" 4 > dispatch_groups.log 2>&1 &
$PYTHON validate_group_answers.py --dispatch dispatch_groups --arm rg
$PYTHON $CORE/verify_cited_pmids.py --dispatch dispatch_groups --arm rg
# 6. citation pass (shared; --subject regulator_group changes only the wording)
$PYTHON $CORE/build_citation_candidates.py --dispatch dispatch_groups --arm rg \
    --enrichment $G/string_enrichment_groups.csv --ncbi-context $G/group_context.json \
    --cache-dir $G/cache --output-dir $G/citation_candidates
$PYTHON $CORE/flag_retracted_pmids.py --candidates-dir $G/citation_candidates --output $G/excluded_candidate_pmids.json
$PYTHON $CORE/build_citation_prompts.py --candidates $G/citation_candidates --dispatch-root dispatch_group_citations \
    --arm rgcite --subject regulator_group --cell-system "<cell_system>" --excluded-pmids $G/excluded_candidate_pmids.json
nohup bash $CORE/dispatch_until_complete.sh dispatch_group_citations "rgcite_p*" 4 > citations.log 2>&1 &
$PYTHON $CORE/validate_citation_answers.py --dispatch dispatch_group_citations --arm rgcite
$PYTHON $CORE/verify_cited_pmids.py --dispatch dispatch_group_citations --arm rgcite --answer-key claims
# 7. viewer
$PYTHON build_group_viewer.py --config $C --dispatch dispatch_groups --arm rg \
    --citations dispatch_group_citations/rgcite --program-viewer annotation_viewer.html --output group_viewer.html
```

Copy `configs/example_config.json` for `$C`. Its settings match the ProgramAnnotatorV3 config of
the same screen.

## Scripts

| Script | Does |
|---|---|
| `build_regulator_effect_matrix.py` | long regulator table → regulator × program[×condition] log2FC, standard error, reliability, strength tier |
| `define_regulator_groups.py` | noise-corrected SNN grouping, bootstrap stability, CORUM calibration grid, complex rescue |
| `measure_neighbour_knockdown.py` | knockdown of each target's promoter neighbours (and of the target) from the h5mu, per condition |
| `build_promoter_confound_screen.py` | exclude / flag / clear per member; shared-locus tie-break |
| `build_group_evidence.py` | effect signature (with V3 program labels), STRING, enrichment vs screened targets, complexes, reference pool; files for the citation pass |
| `build_group_prompts.py` | one blinded prompt per group (`--groups` to select) |
| `validate_group_answers.py` | annotation gate (label rules shared with V3 via `annotator_core/gate_rules.py`) |
| `build_group_viewer.py` | the HTML viewer |

Tests: `cd .. && pixi run -m ../pixi.toml pytest ../tests -q`. They cover:
- grouping on a synthetic screen, where weak and strong blocks must both be recovered;
- promoter-neighbour geometry;
- the group gate.
