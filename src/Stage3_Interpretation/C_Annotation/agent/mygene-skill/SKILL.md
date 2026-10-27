---
name: mygene-skill
description: Look up MyGene.info gene cards (official symbol, full name, aliases, NCBI summary, GO biological process terms, GeneRIF statements with PMIDs) for human gene symbols, including outdated symbols. Use for the gene_info step of a program, or whenever a gene's identity or basic function is needed.
---

## Operating rules
- Use `scripts/mygene.py`; one call takes up to 1000 symbols, so send a program's genes in one call.
- Old symbols resolve through aliases (KIAA1429 -> VIRMA); the row keeps the symbol you asked for.
- For a program, pass `write_to` = `Programs/P<k>/gene_info.jsonl`: each gene's card is merged into its row under
  `MyGene` (NCBI and UniProt add their own keys to the same row).
- GeneRIF statements are short curated claims with a PMID; treat them as leads, not as evidence about this screen.

## Input
One JSON object on stdin. Required: `genes` (list of symbols). Optional: `species` ("human"), `top_go_bp` (10),
`top_generif` (10), `write_to` (gene_info.jsonl path).

## Output
`ok`, `source`, `n_found`, `not_found`, and either `genes` ({symbol: card}) or `written_to`.
Failure: `ok=false`, `error.code` in `invalid_json`, `invalid_input`, `runtime_error`.

## Execution
```bash
echo '{"genes":["TFAM","KIAA1429"]}' | python scripts/mygene.py
```
