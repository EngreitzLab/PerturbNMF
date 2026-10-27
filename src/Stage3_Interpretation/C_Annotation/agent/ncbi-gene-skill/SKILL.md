---
name: ncbi-gene-skill
description: Look up NCBI Gene records (gene id, official symbol and name, aliases, other designations, RefSeq summary, cytogenetic location, GeneRIF PMIDs) for human gene symbols. Use for the gene_info step of a program, or to get a gene's GeneRIF PMIDs for the literature step.
---

## Operating rules
- Use `scripts/ncbi_gene.py`; it follows NCBI's rate limit (set `NCBI_API_KEY` for 10 requests/s).
- For a program, pass `write_to` = `Programs/P<k>/gene_info.jsonl` (merged into each gene's row under `NCBI`).
- `generif_pmids` are the gene's GeneRIF papers, the starting set for generif-papers-skill.

## Input
One JSON object on stdin. Required: `genes`. Optional: `taxid` (9606), `top_generif` (10), `generif` (true),
`write_to`.

## Output
`ok`, `source`, `n_found`, `not_found`, and either `genes` ({symbol: record}) or `written_to`.

## Execution
```bash
echo '{"genes":["TFAM","MT-ND6"]}' | python scripts/ncbi_gene.py
```
