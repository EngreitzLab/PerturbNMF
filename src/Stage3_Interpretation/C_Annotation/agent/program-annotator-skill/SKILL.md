---
name: program-annotator-skill
description: Annotate one cNMF gene program of a PerturbNMF Perturb-seq screen by working through its Programs/P<k>/plan.md — gene information, regulator-gene interactions, literature, regulator context, label, optional driver genes and the layered interpretation — calling the data skills for each step and ticking each step only after its checks pass. Use whenever asked to annotate, label or interpret a program (e.g. "annotate P10").
---

# Program annotator

One program = one folder, `Programs/P<k>/` (relative to the run folder you are in). Its `plan.md` lists the steps;
`prepare` and `confounder_screens` are already done. Work the remaining steps in order and stop when
`scripts/plan.py show` reports that all required steps are done.

## Rules
- Write only inside `Programs/P<k>/` (PDFs in `Programs/P<k>/papers/`). Read other programs' folders only through
  the scripts here.
- Never edit `plan.md` by hand. Close a step with
  `python <this skill>/scripts/plan.py done --program_dir Programs/P<k> --step <id> --by agent`. It runs the step's
  checks; on failure it prints what is wrong; fix the outputs and run it again. Never work around a check.
- Use the data skills' scripts (JSON on stdin, JSON on stdout) with `write_to` so large results go to the program's
  files, not into the conversation.
- Evidence comes from the program's files and from papers you actually read. Nothing from memory is evidence.

## Files you start with
`program.json` (cell type, conditions), `genes_top300.tsv` (rank, gene, loading, distinctive_score),
`regulators_sig.tsv` (every significant knockdown: negative log2fc = activator of the program), `go_sig.tsv`,
`string_sig.tsv` + `string_genes.json`, `confounder_screens.json`, an empty `literature.jsonl`.

## Steps
1. **gene_info** — genes = top 20 of `genes_top300.tsv` + every regulator in `regulators_sig.tsv` (+ any gene you will
   discuss). Call mygene-skill, ncbi-gene-skill and uniprot-skill, each with `write_to: Programs/P<k>/gene_info.jsonl`.
2. **interactions** — omnipath-skill with the top 15-20 genes, the regulators as {symbol: {log2fc}},
   `gene_info: Programs/P<k>/gene_info.jsonl`, `write_to: Programs/P<k>/interactions.jsonl`. A program with no
   significant regulator has nothing to test: create an empty `interactions.jsonl` and close the step.
3. **literature** — see references/literature.md. Search (gene-function-papers, generif-papers, pubmed, openalex,
   crossref, nature) with `write_to: Programs/P<k>/literature.jsonl`; fetch and read the relevant papers with
   paper-fetch-skill and record each one's finding with `note`. Rows already in `literature.jsonl` with
   `status: candidate` are papers found earlier but never read: read those first. Aim for 8-15 read papers, including the strongest
   regulators and the top genes; mark off-topic hits `irrelevant`.
4. **regulator_context** — `echo '{"program_dir":"Programs/P<k>"}' | python scripts/build_regulator_context.py`
   (C2 what each regulator is, C3 its interactions with the program's genes, C4 its specificity across programs,
   C5 the STRING links among the regulators).
5. **label** — references/label.md; write `label.json` (references/schemas.md).
6. **driver_genes** (optional) — when asked, or when the label is uncertain: references/driver_genes.md (judge each
   top gene and regulator from the papers you read; no OpenAI). Otherwise
   `plan.py skip --step driver_genes --reason "not requested"`.
7. **interpretation** — references/interpretation.md: run `scripts/build_evidence.py`, read `evidence.md`, write
   `interpretation.json`.

Finish with a short summary: the label, the coherence, and the steps that needed fixes.

## References
- references/literature.md — how to search, read and record papers
- references/label.md, references/interpretation.md — the two writing tasks
- references/driver_genes.md — the optional driver-gene judgement from the papers read
- references/schemas.md — label.json, driver_genes.json, interpretation.json
