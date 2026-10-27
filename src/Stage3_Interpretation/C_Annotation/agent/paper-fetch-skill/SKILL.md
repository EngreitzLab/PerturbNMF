---
name: paper-fetch-skill
description: Download a paper's open-access PDF (direct links, PMC open data, PMID->PMCID, Unpaywall), read chosen pages as text, and record what the paper shows in a program's literature.jsonl (finding, stance, context). Use in the literature step after a search skill has added candidate papers.
---

## Operating rules
- Use `scripts/paper_fetch.py` with `action` fetch, read or note.
- Read before you record: a paper becomes evidence only through `note` with `status: read` and a `finding` written
  from text you actually read (full text, or the abstract when no PDF is available — say which in `read_from`).
  Mark papers that turn out to be off-topic `status: irrelevant` instead of deleting them.
- Read page 1-3 first (abstract, introduction, first results); read further only when needed.
- `stance` is the paper's bearing on the program's hypothesis: supports, contradicts, mixed or context_only.
  `context` records species / cell type / system, so a finding from another cell type is not over-read.
- PDFs go in `Programs/P<k>/papers/`.

## Input
One JSON object on stdin.
- fetch: `out_dir` and at least one of `pmid`, `pmcid`, `doi`, `pdf_urls`; optional `name`, `literature` (+ the same
  pmid/doi/title to find its row, which then gets `pdf_path`).
- read: `pdf_path`; optional `start_page` (1), `max_pages` (3, max 10), `max_chars` (20000).
- note: `literature`, one of `pmid`/`doi`/`title`, `status` (read | irrelevant); for read: `finding` (required),
  `stance`, `context`, `read_from`, `genes` (added to the row's genes).

## Output
`ok`, `action`, and fetch: `pdf_path` (null when no open-access PDF), `message`; read: `text`, `pages`,
`page_count`, `truncated`; note: the updated `row`. Failure: `ok=false`, `error.code` in invalid_json,
invalid_input, not_found, runtime_error.

## Execution
```bash
echo '{"action":"fetch","pmid":"23845738","out_dir":"Programs/P10/papers","literature":"Programs/P10/literature.jsonl"}' | python scripts/paper_fetch.py
echo '{"action":"read","pdf_path":"Programs/P10/papers/23845738.pdf","max_pages":2}' | python scripts/paper_fetch.py
echo '{"action":"note","literature":"Programs/P10/literature.jsonl","pmid":"23845738","finding":"...","stance":"context_only","context":"mouse muscle"}' | python scripts/paper_fetch.py
```
