---
name: wikipedia-skill
description: Search English Wikipedia and return the plain-text start of the best-matching articles. Use for quick background on a gene family, pathway, complex, cell type or disease — never as a citable source.
---

## Operating rules
- Use `scripts/wikipedia.py`. Wikipedia is orientation only: do not cite it in label.json or interpretation.json and
  never add it to literature.jsonl.

## Input
One JSON object on stdin. Required: `query`. Optional: `max_results` (3, max 10), `max_chars` (3000 per article).

## Output
`ok`, `source`, `query`, `articles` (title, url, text).

## Execution
```bash
echo '{"query":"mitochondrial transcription factor A"}' | python scripts/wikipedia.py
```
