---
name: crossref-skill
description: Search Crossref bibliographically (title, authors, journal, year) or resolve one DOI exactly; returns DOI, title, authors, year, journal and publisher PDF links. Use to verify or complete a citation, or to find a paper from a citation string.
---

## Operating rules
- Use `scripts/crossref.py`; at most 15 results per call (default 5).
- A DOI as the query (10.1038/... or https://doi.org/...) resolves exactly that work; a citation string like "Engreitz 2016 Nature enhancer" matches bibliographically. Abstracts are often missing.
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
echo '{"query":"10.1093/nar/gkg717"}' | python scripts/crossref.py
```
