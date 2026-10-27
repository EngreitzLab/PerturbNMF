# Driver genes step (optional)

Question for each candidate gene: **does this gene's own literature show it carries out or controls the biology
named in the program's label?** The answer comes from papers about the gene that you actually read — the rows of
`literature.jsonl` with `status: read` and the gene in `genes` — not from memory. This is the agent version of the
step; driver-gene-paperqa-skill (OpenAI) is not needed.

## Candidates
The top 15 genes of `genes_top300.tsv`, the 8 most distinctive genes (highest `distinctive_score` outside the top 15)
and the 6 strongest regulators of `regulators_sig.tsv` (lowest adj_p). Skip gene families that carry no
gene-specific literature (e.g. unannotated ENSG IDs, MT-RNR) with verdict insufficient and reason "no gene-specific
literature".

## For each candidate
1. Collect its read papers from `literature.jsonl`. If it has none and is a top-5 gene or a top-3 regulator, run
   gene-function-papers-skill for it (with its aliases from `gene_info.jsonl`), read the 1-3 most relevant papers with
   paper-fetch-skill and record them with `note`, exactly as in the literature step.
2. Judge, in ANY cell type or species:
   - `supported`: the papers show the gene carries out or controls a core process named in the label;
   - `mixed`: some papers show it, others show the opposite;
   - `contradicted`: the papers show it does not;
   - `insufficient`: the papers do not address this biology (papers on unrelated topics are not "against").
   Do not answer insufficient only because the screen's cell type was not studied; record that in
   `cell_type_evidence` instead: `same` (the screen's cell type), `related`, `other`, or `none`.
3. Write one sentence of `reason` (what the papers show the gene does, by which assay and in which system), and
   `paper_findings`: for each read paper that supports the verdict, `{"pmid", "title", "finding"}` — the title and
   PMID exactly as in `literature.jsonl`, the finding in one sentence.

## Output
`driver_genes.json` (schema in schemas.md): `annotation` = the label's `program_name`; `genes` = every candidate
judged; `explained` = the genes with verdict `supported` (or `mixed`), strongest first. If fewer than 2 of the top 15
genes are explained, set `relabel_suggested: true` and say why in `note` — do not change the label in this step.
Then close the step with `plan.py done ... --step driver_genes`.
