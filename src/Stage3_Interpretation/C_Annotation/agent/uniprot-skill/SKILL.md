---
name: uniprot-skill
description: Look up reviewed UniProt (Swiss-Prot) entries for human gene symbols — protein name, curated function, subcellular location, tissue specificity, pathway and the PMIDs behind the function text. Use for the gene_info step of a program, or when a protein's molecular function or location matters.
---

## Operating rules
- Use `scripts/uniprot.py`; it queries reviewed human entries only.
- For a program, pass `write_to` = `Programs/P<k>/gene_info.jsonl` (merged into each gene's row under `UniProt`).
- UniProt function text is curated but general; it says what the protein does, not what it does in this screen.

## Input
One JSON object on stdin. Required: `genes`. Optional: `taxid` (9606), `write_to`.

## Output
`ok`, `source`, `n_found`, `not_found`, and either `genes` ({symbol: entry}) or `written_to`.

## Execution
```bash
echo '{"genes":["TP53"]}' | python scripts/uniprot.py
```
