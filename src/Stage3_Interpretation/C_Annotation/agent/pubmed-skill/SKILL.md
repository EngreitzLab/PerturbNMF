---
name: pubmed-skill
description: Search PubMed (NCBI E-utilities) and get structured records — PMID, DOI, PMCID, title, year, journal, study type, abstract, retraction and preprint flags. Use for the literature step of a program, for any gene or regulator question that needs papers, and to verify a PMID.
---

## Operating rules
- Use `scripts/pubmed.py`; at most 15 results per call (default 5).
- Field tags work: GATA4[tiab] AND heart development[mesh]; date ranges: AND (2020[dp]:2024[dp]). Set NCBI_API_KEY for a higher rate limit.
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
echo '{"query":"TFAM[tiab] AND endothelial","max_results":5}' | python scripts/pubmed.py
```
