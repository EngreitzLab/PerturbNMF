---
name: nature-skill
description: Search Nature-portfolio journals (Nature, Nature Genetics, Nature Communications, ...) through Crossref with DOI prefix 10.1038; returns DOI, title, authors, year, journal, abstract and the article PDF link. Use when a Nature-family paper is likely the key reference.
---

## Operating rules
- Use `scripts/nature.py`; at most 15 results per call (default 5).
- Plain keywords; adding a journal name or year biases toward it. The PDF link needs a subscribing network.
- For a program, pass `write_to` = `Programs/P<k>/literature.jsonl` and `genes` (the program genes or regulators the
  query is about): each hit is appended once (deduplicated by PMID, then DOI, then title) as `status: candidate`.
  A candidate is not evidence until it has been read: use paper-fetch-skill to fetch and read it and to record its
  finding.
- Abstracts in the reply are cut to `abstract_chars` (600); the file keeps them whole.

## Input
One JSON object on stdin. Required: `query`. Optional: `max_results` (5, max 15), `abstract_chars` (600),
`write_to` + `genes` (both needed to write).

## Output
`ok`, `source`, `query`, `n_results`, `records` (pmid, doi, title, year, journal, study_type, abstract, url, ...),
and with `write_to`: `added`, `already_present`. Failure: `ok=false` with `error.code`.

## Execution
```bash
echo '{"query":"TFAM mitochondrial nucleoid","max_results":3}' | python scripts/nature.py
```
