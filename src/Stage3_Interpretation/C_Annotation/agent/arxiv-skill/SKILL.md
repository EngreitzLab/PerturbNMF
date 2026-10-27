---
name: arxiv-skill
description: Search arXiv preprints (quantitative biology, statistics, machine learning) with id, title, authors, year, category, abstract and PDF link. Use for methods or computational papers that are not in PubMed.
---

## Operating rules
- Use `scripts/arxiv.py`; at most 15 results per call (default 5).
- Biology results are mostly computational; prefer pubmed-skill for experimental biology.
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
echo '{"query":"perturb-seq gene program factorization","max_results":3}' | python scripts/arxiv.py
```
