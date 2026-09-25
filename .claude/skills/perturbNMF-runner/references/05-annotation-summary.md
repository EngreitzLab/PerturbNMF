# Annotation & Excel Summary Stages

> **All-flags convention (mandatory for every generated `.sh`):** When invoking `generate_slurm.py`, list active flags first, then `---COMMENTED---`, then every remaining flag for this stage from `references/parameter-catalog.md` (annotation / excel-summary parameter sections) with a sensible default/example value. The generator emits unused flags as `#     --flag value` lines below the python command so the user can toggle them later. See `SKILL.md` Step 5.

---

## Annotation (Stage 3d)

**Conda**: `progexplorer`

LLM-driven gene program annotation. Runs the PerturbNMF Annotation pipeline which extracts top genes per program, queries STRING for protein interactions, mines literature, builds prompts, and submits to an LLM for annotation.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Path to pipeline config YAML (see `src/Stage3_Interpretation/C_Annotation/configs/pipeline_config.yaml` for template) |

The config YAML specifies: input spectra file, output directory, LLM model, STRING parameters, and literature mining settings. Any of the YAML keys can also be overridden on the command line — see `references/parameter-catalog.md` (Annotation section) for the full list.

### Common CLI overrides (skip the YAML for one-offs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gene-loading` | (from YAML) | Path to gene loading matrix CSV |
| `--celltype-enrichment` | None | Raw cell-type enrichment CSV (auto-summarized if provided) |
| `--output-dir` | (from YAML) | Output directory for all results |
| `--regulator-file` | None | SCEPTRE regulator results CSV |
| `--topics` | all | Comma-separated topic IDs to process (e.g. `2,6,33`) |
| `--species` | `10090` | NCBI taxonomy ID (`10090` = mouse, `9606` = human) |
| `--keyword` | (from YAML) | PubMed search keyword for tissue/cell type |
| `--annotation-role` | (from YAML) | Specialist role used in the LLM prompt header |
| `--annotation-context` | (from YAML) | Dataset/cell-type description for the prompt header |
| `--top-positive-regulators` | (from YAML) | Number of positive regulators per program |
| `--top-negative-regulators` | (from YAML) | Number of negative regulators per program |
| `--regulator-significance-threshold` | (from YAML) | Adjusted p-value cutoff when regulator file lacks a `significant` column |
| `--start-from` | None | Resume from a step: `string_enrichment`, `literature_fetch`, `batch_prepare`, `batch_submit`, `parse_results`, `html_report` |
| `--stop-after` | None | Stop after a step (same choices as `--start-from`) |
| `--restart-from` | None | Re-run from the specified step (overwrites later state) |
| `--gcs-prefix` | None | GCS prefix for batch results (for resuming at `parse_results`) |
| `--no-resume` | off | Disable resume/caching; re-query all APIs |
| `--wait` | off | Wait for LLM batch completion (default: submit and exit; resume later) |
| `--force-restart` | off | Ignore existing state and restart pipeline (overwrites prior output) |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 32G
- Time: 1-2h (depends on number of programs and LLM response time)

---

## Blinded v3 annotation (Stage 3d — recommended path)

`src/Stage3_Interpretation/C_Annotation/ProgramAnnotatorV3/` — see its `README.md` for input
formats and the full command sequence. Use it instead of the ProgramExplorer LLM step when you
want annotations that can be defended program by program. It runs locally (or on an
interactive node) with the Claude Code CLI; it is not a SLURM stage.

What is different from the ProgramExplorer prompt:
- **Confounders first.** Deterministic screens (positional clusters, cell cycle, heat shock /
  ISR / UPR / interferon sets, ribosomal and other symbol families, CRISPRi cis-targets,
  regulator counts; plus `stage_composition` for a time course) are computed before any LLM
  sees the program, and the model must clear each one citing a number.
- **Layered reading:** upstream trigger / co-regulation mechanism / cellular output (plus
  `temporal_window` for multi-condition screens), not one forced category.
- **Evidence:** top 30 genes in detail plus all program genes ranked, 30 distinctive genes, every
  significant regulator split by sign (per condition, in experimental order, with a
  cross-condition log2FC profile), STRING enrichment, gene summaries, a PMID reference pool.
- **Label rules:** no quality words; a distinguisher is a process, pathway, compartment or state
  term, never a bare gene; several processes as a comma-separated list.
- **Structural blinding:** one tool-less `claude -p` per prompt, prompt on stdin. Never use
  subagents as annotators, and never put reference labels in the evidence.

Gates (deterministic; a failing answer is kept as `answer.rejected.<n>.json` and re-dispatched):
- the answer must PARSE as JSON — `claude -p` can return a truncated answer and exit 0;
- every gene and PMID named must appear in the prompt; label <= 6 words, no banned words; no
  coherence talk in the brief summary; every confounder assessed; the temporal window's peak
  matches the data;
- every cited PMID exists and is not retracted (retraction NOTICES are caught too).

Operational landmines:
- `claude -p` needs keychain and network access: run it outside any sandbox.
- When usage-limited, `claude -p` exits nonzero with an EMPTY stderr. `dispatch_until_complete.sh`
  rides through it; judge the run by `<dispatch>/DISPATCH_STATUS`, never by the exit code.
- cNMF spectra files can index programs 1..K while regulator tables index 0..K-1 — verify the
  mapping on one program before building any prompt.
- `02_fetch_ncbi_data.py --keyword` defaults to an endothelial keyword; set it for your system.

## Cross-program label disambiguation (Stage 3d, second pass)

**Run this after annotation, always.** Every program is annotated by its own independent call
(deliberately — it keeps annotations comparable and blind), so no annotator can know another
program already took its label. Distinguishability is a property of the whole set and can only be
fixed once the set exists. Left unfixed, a run with dozens of programs reliably produces pairs like:

```
Program A   Vascular endothelial adhesion identity
Program B   Mature vascular endothelial identity
```

Both are defensible; neither tells you which program it is.

### What the annotation prompt must ask for

| Field | What it holds |
|---|---|
| `label_family` | the broad theme a sibling could plausibly share (`Angiogenesis`, `Cell cycle`, `UPR`) — the plainest standard name, so two siblings produce the SAME string |
| `label_distinguisher` | what separates THIS program from a sibling — a cell-process, pathway, compartment, phase or state term. Never a bare or parenthesised gene symbol, unless the gene IS the accepted name of the process ("KLF2 flow response") |
| `label_distinguisher_evidence` | the genes the distinguisher rests on, preferably distinctive genes |

### Detect, then resolve

1. **Detect — deterministic, free.** Group programs by normalised `label_family` and by
   near-duplicate labels (word-token Jaccard >= 0.6). One prompt per colliding group.
2. **Resolve — one blinded call per group.** The prompt sees only the colliding programs' labels,
   families, distinguishers, slot claims, top and distinctive genes.
3. **Apply**, keeping the original as `label_before_disambiguation`.

### House rules for the rewrite

```
Angiogenesis - Tip cell        Angiogenesis - Stalk cell     <- best: a named sub-state or process
Angiogenesis 1                 Angiogenesis 2                <- last resort only
Angiogenesis - APLN            Angiogenesis - ESM1           <- NOT allowed: one gene looks arbitrary
```

- **Do no harm.** A label already specific and distinct from every other in the group is kept
  verbatim; only labels that actually collide are rewritten.
- **Standardise the shared part first**, then distinguish within it.
- **Distinguish by process, not by gene.** No gene tag on a named sub-state ("Cell cycle - G2M",
  not "Cell cycle - G2M (CDC20)"). Prefer the established name of a process.
- **A bare trailing number is allowed** when nothing in the evidence separates two programs;
  record it (`used_bare_number: true`) — a high rate means k is too large for the data.
- **Conservatism beats specificity.** A duller label that is true beats a sharper one that is not.
- <= 6 words; never "program", "process", "regulation of", or a quality word ("grab bag",
  "incoherent", "heterogeneous", "mixed", "unclear", "miscellaneous"). Several processes are a
  comma-separated list, never slashes.
- A program that does not belong to the family is labelled from its own genes and flagged
  `belongs_to_family: false`.

```bash
python resolve_label_collisions.py detect --dispatch <dir> --arm v3 \
    --gene-loading <loading.csv> --out-root <groups_dir> --cell-system "<cell system>"
bash run_blinded_annotations.sh <groups_dir> 4 "v3_group*"
python resolve_label_collisions.py apply --dispatch <dir> --arm v3 --out-root <groups_dir>
```

---

## Citation pass (Stage 3d, third pass — required)

**Every gene the label rests on, and every key regulator hypothesis, gets a citable source — or
an explicit "none found".** Run it after disambiguation, always.

Why a separate pass: the annotation prompt's literature pool is one PubTator search per program
(30 genes OR'd, capped at 25 papers), so only a small minority of label genes arrive with a
citable sentence, while most sit in a significant enrichment term of their own program that
nobody links to them. And support for "gene X belongs to process Y" can only be searched for
once Y — the label — exists.

**Claims:** every gene in `label_evidence.genes`, every regulator in `label_evidence.regulators`,
and every `regulators[]` hypothesis with `high` or `medium` confidence.

### Candidates, per claim — DISCOVERY FIRST (deterministic retrieval, cached)

The citation a reader expects is the study that **discovered** the gene's role in the labelled
process (first identification, the defining loss/gain-of-function, the first mechanism) — not a
recent paper that restates it. Relevance-ranked search (PubTator, PubMed "best match") returns
restatements almost exclusively. So candidates come from channels that surface originals, the
way citation-graph and research tools do (Semantic Scholar / Connected Papers "prior works",
PaperQA2 citation traversal, scite citation contexts, citation counts). The backend is Europe
PMC (EMBL-EBI) — search sorted by citations or date, abstracts, publication types and
per-paper reference lists, with no API key:

| Channel | How | Why it finds originals |
|---|---|---|
| `most_cited` | Europe PMC `(TITLE_ABS:"SYMBOL" OR aliases) AND (TITLE_ABS:label words)`, non-review, `sort=CITED desc`; over-fetch 25, keep the 8 matching the most distinct label words | foundational papers are the most cited on their topic |
| `earliest` | same search, `sort=PUB_YEAR asc` | first reports that never became highly cited |
| `co_cited` | references (Europe PMC `/MED/{pmid}/references`) shared by >= 2 of the topic papers — the most-cited hits, the 3 most-cited reviews (their reference lists concentrate the originals) and the PubTator hits — that name the gene | the backbone every later paper cites |
| `curated` | UniProt FUNCTION evidence (ECO:0000269) and NCBI GeneRIFs matching the label words | curators attach the finding to the paper that reported it |
| `topic` | PubTator `GENE AND (label words)` | current context and the cell-system match |
| database | significant enrichment terms of THIS program containing the gene, with the PMID the gene's experimental GO annotation (QuickGO) or its gene summary ("[PubMed N]") cites | the database's own citation |

**Aliases are essential** — originals use old names (KDR = Flk-1, ETV2 = ER71/Etsrp, CDH5 =
VE-cadherin). Take MyGene aliases plus UniProt protein short/alternative names.

Each paper is shown with year, journal, total citations, co-citation count, channels and a
`[REVIEW]` flag (PubMed pubtype), oldest first, with its title/abstract sentences that name the
gene. Per-channel quotas (curated 4, co_cited 4, most_cited 4, earliest 2, topic 3) stop one
channel crowding out the rest. Retracted papers and retraction notices are dropped — screen the
candidate pools, not just the annotation pool (targeted retrieval pulls in ~20x more papers, and
retracted ones are among them).

### Selection — one tool-less call per program

Per claim, up to 2 supports, each with a `role`:
- `discovery` — the original primary study that established THIS relationship (gene <->
  labelled process); never a review, never the gene's first paper in an unrelated function;
- `context` — primary evidence in the matching cell system, or a database term with its PMID;
- `restatement` — the best available support when no original is on offer; never passed off as
  a discovery.
**"None" is valid and better than a stretched citation.** Literature picks carry a verbatim quote.

### Gate (deterministic — reject and re-dispatch on any failure)

1. every claim answered exactly once; a claim with no support has a `none_reason`
2. every chosen id exists under THAT claim; a literature PMID matches its id
3. the quote is verbatim from the offered sentence or title; an elision mark ("...") between
   verbatim pieces is fine, paraphrase is not
4. a database PMID is one the entry itself cites (its GO annotation, or "[PubMed N]" in its text)
5. every chosen PMID exists and is not retracted (esummary)
6. a `discovery` support is a literature candidate and not a review; a discovery >= 10 years
   newer than an offered primary paper with >= 3x its citations is a WARN (possible restatement)

An **id slip** (the model's `L3` actually quotes `L4`: PMID and quote match exactly one other
paper under the same claim) is resolved to that candidate and reported as a WARN. Expect roughly
1 program in 5 to need one retry.

**Caveat to state wherever citations are shown:** retrieval uses the label's own words, so a
citation shows "a paper links this gene to this process" — it cannot test whether the label is
right, and a `discovery` pick is the likeliest original on offer, not a guarantee. Entailment is
not checked mechanically.

```bash
python build_citation_candidates.py --dispatch <dir> --arm v3 --enrichment <string_filtered.csv> \
    [--enrichr <enrichr.tsv> ...] --ncbi-context <ncbi_context.json> --excluded-pmids <excluded.json> \
    --cache-dir <cache> --output-dir <candidates>    # no API keys; cached — rerun to fill transient gaps
python flag_retracted_pmids.py --candidates-dir <candidates> --output <excluded_candidates.json>
python build_citation_prompts.py --candidates <candidates> --dispatch-root <cite_dir> --arm cite \
    --cell-system "<cell system>" --excluded-pmids <excluded_candidates.json>
bash dispatch_until_complete.sh <cite_dir> "cite_p*" 4
python validate_citation_answers.py --dispatch <cite_dir> --arm cite
python verify_cited_pmids.py --dispatch <cite_dir> --arm cite --answer-key claims
```

---

## Annotation viewer (HTML)

`ProgramAnnotatorV3/scripts/build_annotation_viewer.py` — one self-contained HTML file (no CDN),
from the same config the prompt builder used:

```bash
python build_annotation_viewer.py --config <config.json> --dispatch <dir> --arm v3 \
    [--citations <cite_dir>/cite] --output annotation_viewer.html
```

- Left rail: every program, grouped by peak condition (multi-condition) or label family (single
  condition); positional / technical programs tagged. Full-text search, `#program-N` deep links,
  arrow keys, dark mode.
- Per program: label / family / distinguisher and brief summary; activity by condition and the
  temporal window; genes; the genes the label rests on with the support the citation pass chose
  (★ discovery, ◆ context, ◐ restatement, or none); regulator volcano plots (one per condition,
  shared axes, regulators named in the annotation labelled) with a table view; regulator
  hypotheses with their support; confounders; layered interpretation; modules; competing
  readings; QC (re-dispatches, validator warnings, collision-pass renames).

---

## Literature Search (optional companion to Annotation)

**Conda**: `progexplorer`

Mines PubMed/PubTator for evidence supporting the program annotations produced by the Annotation stage. Run after Annotation if you want literature citations attached to each program.

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--excel` | Input Excel with one row per program (typically the Annotation HTML report's source workbook) |
| `--output-dir` | Output directory for per-program literature pages |

### Common optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--programs` | all | Comma-separated program IDs (e.g. `2,6,33,34`) |
| `--interactions` | (built-in 17-verb list) | Comma-separated interaction verbs used to formulate queries |
| `--domain-keywords` | (built-in vascular set) | Comma-separated domain keywords for evidence scoring |
| `--max-papers` | `30` | Max papers per program |
| `--max-pubtator-results` | `50` | Max results per PubTator query |
| `--max-llm-queries` | `8` | Max LLM-generated queries per program |
| `--llm-provider` | `stanford` | One of `anthropic`, `stanford`, `openai`, `deepseek`, `gemini` |
| `--llm-model` | None (provider default) | LLM model name |
| `--llm-max-tokens` | `4096` | Max tokens for LLM output |
| `--semantic-check` | off | Enable LLM semantic verification (costs tokens) |
| `--resume` / `--no-resume` | resume on | Enable/disable resume/caching |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 32G
- Time: 1-3h (depends on `--max-papers` and LLM throughput)

---

## Excel Summarization (Stage 3e)

**Conda**: `NMF_Benchmarking`

Compiles Stage 1 (`.h5mu`) and Stage 2 (Evaluation) outputs into a single multi-sheet
Excel workbook for **one `(K, sel_thresh)` per job**. There is now a standalone CLI
wrapper — generate the `.sh` with `generate_slurm.py --stage excel-summary` like any
other stage (no notebook required). Submit one job per K to cover multiple K values.

**Script**: `src/Stage3_Interpretation/B_Summarization/Slurm_Version/cNMF_excel_summary.py`
**Source library**: `src/Stage3_Interpretation/B_Summarization/src/Compile_excel_sheet.py`
**Reference notebook (interactive equivalent)**: `src/Stage3_Interpretation/B_Summarization/JupterNote_Version/cNMF_compile_excel_table.ipynb`

### Required parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--out_dir` | Output root directory (contains `{run_name}/`) | `/path/to/project/Result` |
| `--run_name` | Run name identifier | `030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50` |
| `--K` | Number of components (single K) | `50` |

### Commonly set parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sel_thresh` | `0.2` | Density threshold (`0.2` → `0_2`, `2.0` → `2_0`) |
| `--Sample` | `D0 sample_D1 sample_D2 sample_D3` | Condition/sample labels (e.g. `D0 D1 D2 D3`, `WTC`) |
| `--categorical_key` | `sample` | obs column for sample/condition grouping (e.g. `batch`, `timepoint`) |
| `--perturbation_file_name` | `perturbation_association_results` | Perturbation file stem (e.g. `CRT`) |
| `--effect_size` | `log2FC` | Effect-size column (e.g. `approx_log2FC`) |
| `--gene_names_key` | `symbol` | var column with gene symbols |
| `--non_targeting_key` | `non-targeting` | Negative-control target label(s) |

See `references/parameter-catalog.md` (Section 12) for the full flag list, including
`--save_path`, `--mdata_path`, `--num_gene`, the `--prog_key`/`--data_key`/`--guide_targets_key`
keys, `--adjusted_pval_key`, and the per-sheet `--*_Term_key` / `--*_Genes_key` overrides.

### Key path conventions

Reads the per-K evaluation outputs at (paths derived from `--out_dir`/`--run_name`/`--K`/`--sel_thresh`):
```
{out_dir}/{run_name}/Evaluation/{K}_{thresh}/
├── {K}_GO_term_enrichment.txt
├── {K}_geneset_enrichment.txt
├── {K}_trait_enrichment.txt
├── {K}_{perturbation_file_name}_{Sample}.txt   (one per sample)
├── {K}_categorical_association_results.txt
└── {K}_Explained_Variance.txt
```
and the MuData at `{out_dir}/{run_name}/Inference/adata/cNMF_{K}_{thresh}.h5mu`. Where
`{thresh}` = `str(sel_thresh).replace('.', '_')`. Override the auto-derived input with
`--mdata_path` if your h5mu lives elsewhere. (The wrapper calls the individual
`Compile_*` helpers directly, so it does not depend on `load_simple_sheets()`'s hardcoded
`Evaluation/`; for a non-default eval dir, point `--mdata_path` and `--save_path`
explicitly and keep eval files under `Evaluation/` or symlink them.)

### Output

Per job, written to `--save_path` (default `{out_dir}/{run_name}/Interpretation/Summary_table/{K}_{thresh}/`):
- `cNMF_{K}_{thresh}.xlsx` — main multi-sheet workbook
- `Summary_{K}_{thresh}.tsv`, `Program_Loadings_{K}_{thresh}.tsv`, `Targets_Summary_{K}_{thresh}.tsv`
- Sidecars: `specificity_score_{Sample}.txt`, `corr_gene_matrix_{Sample}.txt(.gz)`, `kd_efficiency.txt`, `perturbation_merged_{Sample}(.._significant).txt`
- `config_{SLURM_JOB_ID}.yml`

### Output sheets

| Sheet | Description |
|-------|-------------|
| **Summary** | One row per program: top genes, enrichment highlights, perturbation hit counts, mean scores per condition |
| **Program Loadings** | Long-format gene loading scores with gene descriptions (via MyGene API) |
| **Targets Summary** | Per-target aggregated perturbation stats: expression, cell counts, significant programs, specificity, correlations, KD efficiency |
| **Sample Association** | Kruskal-Wallis + Dunn posthoc p-values per program |
| **Perturbation Association {n}** | Full perturbation results merged with specificity scores (chunked across sheets if >1M rows) |
| **significant regulators only {n}** | Perturbation Association filtered to adj_pval < 0.05, also carrying specificity scores (chunked) |
| **Trait Enrichment** | GWAS trait enrichment via Fisher exact test (Open Targets L2G) |
| **GO Term Enrichment** | GO Biological Process 2023 enrichment |
| **Geneset Enrichment** | Reactome 2022 pathway enrichment |

### SLURM resources

- Partition: `engreitz,owners`
- CPUs: 4
- Memory: 64G
- Time: 2h (MyGene API queries for gene annotations are the bottleneck)
