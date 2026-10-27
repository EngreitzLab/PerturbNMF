---
name: gene-function-papers-skill
description: Find a gene's own function papers in PubMed ((symbol OR aliases) AND function/role/regulation/mechanism in title or abstract), rank them (gene in title, then the screen's cell type, then primary research over reviews) and download open-access PDFs. Use in the literature step for each top program gene and each regulator, to learn what the gene is known to do.
---

## Operating rules
- Use `scripts/gene_function_papers.py`, one gene per call. Pass the gene's aliases from `gene_info.jsonl` (old symbols
  are common in older papers) and the program's cell type.
- For a program: `write_to` = `Programs/P<k>/literature.jsonl`, `pdf_dir` = `Programs/P<k>/papers`, `n_pdfs` 1-3. Papers
  are appended as candidates; read the ones that matter with paper-fetch-skill and record their findings.
- Papers on unrelated tissues are expected; their findings still describe the gene's molecular function, but say
  so in `context` when you note them.

## Input
One JSON object on stdin. Required: `gene`. Optional: `aliases`, `max_aliases` (3), `function_terms`, `cell_type`,
`context_terms`, `pool_size` (20), `n_papers` (5), `n_pdfs` (0), `pdf_dir`, `write_to`, `abstract_chars` (400).

## Output
`ok`, `query`, `pool`, `gene`, `n_papers`, `n_pdfs`, `papers` (pmid, doi, title, year, journal, study_type, abstract,
pdf_path, pdf_status), and with `write_to`: `added`, `already_present`.

## Execution
```bash
echo '{"gene":"LRPPRC","aliases":["LRP130"],"cell_type":"aortic endothelial cell","n_papers":3}' | python scripts/gene_function_papers.py
```
