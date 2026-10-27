---
name: generif-papers-skill
description: Fetch and rank a gene's GeneRIF papers — the references NCBI curators attached to statements of the gene's function — preferring papers in the screen's cell type, with optional open-access PDFs. Use in the literature step when a gene's function papers are sparse or when curated function statements are wanted.
---

## Operating rules
- Use `scripts/generif_papers.py`, one gene per call. The GeneRIF PMIDs come from `pmids`, or from the gene's NCBI
  block in `gene_info` (run ncbi-gene-skill into gene_info.jsonl first).
- For a program: `write_to` = `Programs/P<k>/literature.jsonl`, `pdf_dir` = `Programs/P<k>/papers` (candidates, as with
  every search skill).

## Input
One JSON object on stdin. Required: `gene` and either `pmids` or `gene_info`. Optional: `cell_type`, `context_terms`,
`n_papers` (5), `n_pdfs` (0), `pdf_dir`, `write_to`, `abstract_chars` (400).

## Output
`ok`, `gene`, `n_papers`, `n_pdfs`, `papers`, and with `write_to`: `added`, `already_present`. Failure `not_found` when
the gene has no GeneRIF PMIDs.

## Execution
```bash
echo '{"gene":"TFAM","gene_info":"Programs/P10/gene_info.jsonl","cell_type":"aortic endothelial cell"}' | python scripts/generif_papers.py
```
