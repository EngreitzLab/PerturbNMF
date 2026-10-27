---
name: openalex-skill
description: Search OpenAlex (full-text relevance over title, abstract and full text) for works with DOI, PMID, venue, citation count, abstract and open-access PDF links. Use for the literature step when PubMed misses full-text matches. Needs OPENALEX_API_KEY.
---

## Operating rules
- Use `scripts/openalex.py`; at most 15 results per call (default 5).
- Plain keywords work best; AND / OR / NOT (upper case) and "quoted phrases" are supported. Requires OPENALEX_API_KEY in the environment; without it the script returns ok=false (use pubmed-skill instead).
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
echo '{"query":"LRPPRC mitochondrial mRNA stability","max_results":5}' | python scripts/openalex.py
```
