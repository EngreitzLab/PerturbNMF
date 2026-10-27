---
name: driver-gene-paperqa-skill
description: Optional. Ask paper-qa, over one gene's own downloaded PDFs, whether that gene drives a program's annotated biology; returns a verdict, the cell-type match and the ranked evidence passages. Use only in the optional driver_genes step, after the label exists and the gene's PDFs are in Programs/P<k>/papers/. Costs OpenAI API calls (gpt-4o + embeddings).
---

## Operating rules
- Use `scripts/driver_gene_paperqa.py`, one gene per call, only with PDFs already downloaded for that gene
  (gene-function-papers-skill with `n_pdfs` > 0). It needs `OPENAI_API_KEY` and the `paper-qa` package.
- paper-qa's verdict is a first pass. Read its `evidence` passages and decide the gene's verdict yourself; write
  the step's output, `driver_genes.json`:
  `{"annotation": <label.json program_name>, "genes": {GENE: {"verdict", "cell_type_evidence", "reason",
  "pmids"}}, "explained": [genes whose verdict is supported]}`.
- If most genes come back insufficient, the label may be wrong: say so in `driver_genes.json` ("relabel_suggested")
  and revisit the label step rather than forcing verdicts.
- Model settings are in `scripts/paperqa_config.yaml`.

## Input
One JSON object on stdin. Required: `gene`, `annotation` (the label's program_name), `pdfs` (paths).
Optional: `role` (program gene | distinctive gene | regulator), `cell_type`.

## Output
`ok`, `gene`, `annotation`, `model`, `verdict`, `cell_type_evidence`, `answer`, `citations`, `evidence` (passages with
scores), `n_pdfs`, `cost_usd`.

## Execution
```bash
echo '{"gene":"LRPPRC","annotation":"Mitochondrial OXPHOS","pdfs":["Programs/P10/papers/LRPPRC/29146908.pdf"]}' | python scripts/driver_gene_paperqa.py
```
