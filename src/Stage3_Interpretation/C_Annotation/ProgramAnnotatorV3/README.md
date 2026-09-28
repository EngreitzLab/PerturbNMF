# ProgramAnnotatorV3 — blinded gene-program annotation, citation pass, HTML viewer

Annotates cNMF gene programs from a Perturb-seq screen, one blinded LLM call per program, then
makes the labels distinguishable as a set, attaches a citation to every gene and regulator the
label rests on (discovery paper first), and renders everything as one self-contained HTML file.

Design and rules: `.claude/skills/perturbNMF-runner/references/05-annotation-summary.md`
("Blinded v3 annotation", "Cross-program label disambiguation", "Citation pass",
"Annotation viewer").

## Requirements

- Python 3.10+ with `pandas`, `numpy`, `scipy` (no other packages; HTTP uses the stdlib).
- The Claude Code CLI (`claude`), logged in. Every LLM call is a minimal `claude -p`: no tools
  (`--tools ""`), no Claude Code context (`--safe-mode --strict-mcp-config
  --disable-slash-commands`, neutral working directory), prompt on stdin, answer and cost back as
  JSON (logged to `usage.jsonl`); see `annotator_core/answer_one_prompt.py`. Run the dispatch
  scripts outside any sandbox (keychain + network). `PYTHON` and `ANNOTATOR_MODEL` (default
  `sonnet`) are read from the environment.
- `ANNOTATOR_SPECIES` (environment, NCBI taxon id): `9606` human (default) or `10090` mouse. It
  sets the species for STRING, mygene.info, UniProt and QuickGO lookups, and maps the
  human-curated complexes and confounder marker sets to mouse symbol case. Run the STRING
  enrichment input with the same `--species`.
- Network for the retrieval steps, no API keys: `www.ncbi.nlm.nih.gov` (PubTator3),
  `eutils.ncbi.nlm.nih.gov`, `www.ebi.ac.uk` (Europe PMC, QuickGO), `mygene.info`,
  `rest.uniprot.org`, `string-db.org`.

## Inputs (one directory, named in the config as `data_dir`)

| File | Columns / format | How to make it |
|---|---|---|
| gene loading | `Name, Score, program_id, UniquenessScore` — top 300 genes per program | ProgramExplorer `01_genes_to_string_enrichment.py extract` (computes `UniquenessScore`) |
| regulators | `program_id, target_gene, log2_fc, significant, adj_pval` (+ `condition` for multi-condition screens; config key `regulators_by_condition`, else `regulators`) | your U-test / CRT results. **log2FC sign: negative = knockdown lowers the program.** Drop non-targeting controls |
| STRING enrichment | ProgramExplorer filtered CSV (`program_id, category, term, description, fdr, inputGenes`) | `01_genes_to_string_enrichment.py enrich --species 9606` |
| literature context | ProgramExplorer JSON (`gene_summaries`, `evidence_snippets`, `regulator_validation`) | `02_fetch_ncbi_data.py --top-loading 20 --top-unique 10 --regulator-file <one row per program×target> --keyword "<your system>"` |
| gene coordinates | TSV, no header: `name chrom start end strand gene_type` | from the GTF used for alignment |
| targets | TSV with a `target_name` column: every perturbed gene | the guide library |
| program activity (optional) | `program_id, condition, mean_score` | per-condition mean program score |
| confounder screens | JSON | `build_confounder_screens.py` (below) |
| excluded PMIDs | JSON `{"retracted": [...], "unresolved": [...]}` | `flag_retracted_pmids.py` (below) |
| TF-motif enrichment (optional; config key `motif_enrichment`) | TSV `program, element_type (promoter/enhancer), tf, fdr, enrichment, significant` (+ Stage 2 extras). `program` may be `12` or `K10_12` | Stage 2 motif enrichment (`Stage2_Evaluation/A_Metrics/src/motif_enrichment.py`) |
| candidate TFs (optional; config key `candidate_tfs`) | TSV `program, element_type, tf, tf_gene_symbol, fdr, tf_program_loading_rank, knockdown_log2fc, knockdown_fdr, evidence_tier` | Stage 2 `nominate_candidate_tfs.py` |

Copy `configs/example_config.json` and fill it in. For a single-condition screen, drop
`conditions`, `program_activity` and `regulators_by_condition` and give `regulators`.

**TF motifs are optional.** With `motif_enrichment` (and optionally `candidate_tfs`) the user
message gains section E2 (between the screens, E, and the reference pool, F): one line saying how
the motifs were tested and that motif enrichment is correlative, then per element type the top 5
significant motif families (Stage 2 `motif_family`, e.g. KLF-SP; the motif name for older tables), one
line each: up to 3 motifs with enrichment and FDR, then up to 6 expressed candidate TFs of the family
(tiers `motif+regulator` > `motif+expressed_in_program` > `motif+expressed`). The system prompt is unchanged. The viewer shows
the same selection per program, with sequence logos when the config key `motif_logos` names the Stage 2
`{K}_motif_logos.json` (only the logos of shown motifs are embedded). An optional
`motif_enrichment_label` names the source on the viewer page.
Leave the keys out (or empty) and the prompts are byte-identical to a build without them.
With a Stage 2 `--motif_source both` table (`motif_source` column: FIMO and Fi-NeMo) each
element type x source gets its own block, FIMO first (labelled FIMO/MotifCompendium or FIMO/HOCOMOCO
from the motif names); the two are never pooled. The how-to-read line follows the Stage 2 test: t-test on the
top `n_top` genes (default) or `correlation` (r > 0). It is read from the run's
`{K}_motif_enrichment_config.yml` next to the table, or set with config keys `motif_method` /
`motif_n_top`.

## Multi-condition screens

A screen read out separately in several **conditions** is annotated per condition. A condition
is whatever the screen varies: timepoints, stimuli, doses, donors, genotypes, age × sex cohorts.

| Config key | Meaning |
|---|---|
| `conditions` | `[{"label", "description"}]`; labels match the `condition` column of the regulator and activity tables |
| `settings.condition_variable` | what the conditions vary, in plain words (e.g. `"timepoint"`, `"cytokine stimulus"`, `"donor"`); default `"condition"` |
| `settings.condition_design` | `"unordered"` (default) or `"ordered"` (timepoints, a dose series) |

The prompt shows each regulator per condition plus a cross-condition log2FC profile, and
activity per condition with its peak. The answer fills a `condition_dependence` slot: where the
program is active, where its regulators act, and whether each effect is condition-specific,
constitutive or switches sign. Only an `"ordered"` design adds ordering language (contiguous
blocks of conditions; a regulator acting after the peak cannot be its trigger).

`condition_composition` confounder (optional): when conditions differ in which cells are
present (e.g. timepoints of a differentiation), pass `--activity-by-condition` and
`--condition-markers` (canonical marker genes of each condition's cells;
`configs/example_condition_markers.json`) to `build_confounder_screens.py`. The prompt then asks
whether the program just reads out which cells are present.

Older spellings still work, with a warning: a condition's `stage` (now `description`),
`condition_design` `"time_course"` / `"groups"` (now `"ordered"` / `"unordered"`),
`--condition-markers`, formerly `--stage-markers`. Answers with the older `temporal_window` /
`group_dependence` slot are still read by the gate, the collision pass and the viewer.

**Check the program index first.** cNMF spectra files can number programs 1..K while regulator
tables number them 0..K-1. Confirm on one program (its top genes in both) before building prompts.

## Run

```bash
cd src/Stage3_Interpretation/C_Annotation/ProgramAnnotatorV3/scripts
CORE=../../annotator_core   # dispatch, PMID gates and citation pass, shared with RegulatorGroupAnnotator
export PYTHON=python
D=path/to/annotation_inputs; C=my_config.json

# 1. deterministic screens (add --activity-by-condition + --condition-markers for the condition_composition screen)
$PYTHON build_confounder_screens.py --gene-loading $D/gene_loading_top300_with_uniqueness.csv \
    --gene-coordinates $D/gene_coordinates.tsv --targets $D/targets.tsv \
    --regulators $D/regulators.csv --output $D/confounder_screens.json
# 2. drop retracted / unresolvable papers from the reference pool
$PYTHON $CORE/flag_retracted_pmids.py --ncbi-context $D/ncbi_context.json --output $D/excluded_pool_pmids.json
# 3. prompts, one directory per program
$PYTHON build_annotation_prompts.py --config $C --output batch_request.json
$PYTHON $CORE/split_prompts_for_blinded_dispatch.py --batch batch_request.json --arm v3 --dispatch-root dispatch
# 4. answer them (detached; read dispatch/DISPATCH_STATUS, not the exit code)
nohup bash $CORE/dispatch_until_complete.sh dispatch "v3_p*" 4 > dispatch.log 2>&1 &
# 5. gates — move failing answers to answer.rejected.<n>.json and re-run step 4 for them
$PYTHON validate_annotation_answers.py --dispatch dispatch --arm v3 --programs 0-49 --write-problems
bash $CORE/repair_rejected_answers.sh dispatch 4 "v3_p*"   # short fix call per failure; then re-run the gate
$PYTHON $CORE/verify_cited_pmids.py --dispatch dispatch --arm v3
# 6. cross-program label collisions
$PYTHON resolve_label_collisions.py detect --dispatch dispatch --arm v3 \
    --gene-loading $D/gene_loading_top300_with_uniqueness.csv --out-root dispatch_collisions \
    --cell-system "<cell_system from the config>"
bash $CORE/run_blinded_annotations.sh dispatch_collisions 4 "v3_group*"
$PYTHON resolve_label_collisions.py apply --dispatch dispatch --arm v3 --out-root dispatch_collisions
# 7. citation pass: discovery-first candidates, screened, selected, gated
$PYTHON $CORE/build_citation_candidates.py --dispatch dispatch --arm v3 \
    --enrichment $D/string_enrichment_filtered.csv --ncbi-context $D/ncbi_context.json \
    --excluded-pmids $D/excluded_pool_pmids.json --cache-dir $D/citation_cache \
    --output-dir $D/citation_candidates           # rerun once or twice: failures are not cached
$PYTHON $CORE/flag_retracted_pmids.py --candidates-dir $D/citation_candidates --output $D/excluded_candidate_pmids.json
$PYTHON $CORE/build_citation_prompts.py --candidates $D/citation_candidates --dispatch-root dispatch_citations \
    --arm cite --cell-system "<cell_system>" --excluded-pmids $D/excluded_candidate_pmids.json
nohup bash $CORE/dispatch_until_complete.sh dispatch_citations "cite_p*" 4 > citations.log 2>&1 &
$PYTHON $CORE/validate_citation_answers.py --dispatch dispatch_citations --arm cite
$PYTHON $CORE/verify_cited_pmids.py --dispatch dispatch_citations --arm cite --answer-key claims
# 8. viewer
$PYTHON build_annotation_viewer.py --config $C --dispatch dispatch --arm v3 \
    --citations dispatch_citations/cite --output annotation_viewer.html
```

`settings.effect_label` (default `"log2FC"`) names what the regulator table's `log2_fc` column
holds. The viewer uses it for the volcano x-axis, the effect table headers and every tooltip. Set
it when the column holds another statistic, e.g. `"Calibrated t-statistic"`. Keep it
short: it is the axis title. The prompts use it too, wherever they name the regulator effect
(system rules, regulator and motif-candidate lines; the answer schema keeps its `log2fc` fields), so
set it only with a blinded comparison or a fresh run. With the default the prompts are byte-identical.

## What to expect

- About 3 minutes per program per LLM call on Sonnet; citation retrieval about 8 minutes per
  program cold (Europe PMC reference lists dominate), under a minute when cached.
- Roughly 1 program in 5 fails a gate once (paraphrased quote, a gene named that the prompt
  never showed, coherence talk in the summary). The repair pass fixes it (rejected answer kept as
  `answer.rejected.<n>.json`; after 2 repairs delete `answer.json` for a full re-dispatch).
- `claude -p` exits nonzero with an empty stderr when usage-limited; the dispatcher sleeps and
  retries. It can also return a truncated or prose-prefixed answer — the completeness check
  rejects those and the next pass retries them.
- Validate any prompt change with a blinded pairwise judge on the full program set. For cost,
  the Claude Code harness, not the prompt, dominated: trimming prompt text saved little.

## Scripts

In `scripts/` (program-specific):

| Script | Does |
|---|---|
| `build_confounder_screens.py` | positional, cell-cycle, stress-set, symbol-family, cis-target and (optional) condition-composition screens |
| `build_annotation_prompts.py` | the v3 prompt per program (single- or multi-condition) |
| `validate_annotation_answers.py` | annotation gate |
| `resolve_label_collisions.py` | cross-program label disambiguation |
| `build_annotation_viewer.py` | the HTML viewer |

In `../annotator_core/` (shared with `RegulatorGroupAnnotator`, so a fix lands in both):

| Script / module | Does |
|---|---|
| `split_prompts_for_blinded_dispatch.py` | one isolated directory per program |
| `run_blinded_annotations.sh`, `dispatch_until_complete.sh`, `check_answer_complete.py` | blinded `claude -p` dispatch with completeness check and usage-limit retry |
| `verify_cited_pmids.py`, `flag_retracted_pmids.py` | PMID existence / retraction checks |
| `build_citation_candidates.py`, `build_citation_prompts.py`, `validate_citation_answers.py` | citation pass (`--subject program`, the default) |
| `answer_io.py`, `gene_coordinates.py`, `viewer_common.py` | answer loading, coordinate loading, viewer helpers + stylesheet |
| `conditions.py`, `species.py` | multi-condition config reading (with deprecated aliases); `ANNOTATOR_SPECIES` |
