---
name: string-skill
description: STRING functional enrichment (GO, KEGG, Reactome, WikiPathways), PPI enrichment and high-confidence interaction partners for a ranked human gene list, and the typed STRING links among a set of regulators. Use when a program or gene set needs pathway enrichment or physical/functional partners; a program's own enrichment is already in Programs/P<k>/string_sig.tsv, so query STRING only for a different gene set.
---

## Operating rules
- Use `scripts/string_enrichment.py`; it waits 1 s between STRING calls, as STRING asks.
- A program's top-300 enrichment is computed during preparation (`string_sig.tsv`, `string_genes.json`). Read those
  files before calling STRING again.
- Pass `write_to` (a folder) to get the full tables on disk instead of in the reply; the reply lists only the top
  `max_terms` terms.
- Partner channels include text mining and co-expression: a partner is not evidence of a physical interaction unless
  its channels include `experiments` or `databases`.
- Regulator links are typed by their strongest evidence: `experimental` (the `experiments` channel), else `curated`
  (`databases`), else `predicted` (text mining, co-expression or genomic context only). A channel counts when its own
  score is >= 0.3.
- PPI enrichment is against STRING's genome: STRING computes none against a background above 2000 proteins, so
  `background_genes` applies to the term enrichment only.

## Input
One JSON object on stdin.
- Required: `genes` — ranked gene symbols (most important first).
- Optional: `taxid` (9606), `categories` (Process, Function, Component, KEGG, RCTM, WikiPathways), `max_fdr` (0.05),
  `required_score` (700 = high confidence), `max_partners` (15), `candidates` ({gene: role} whose partners are
  listed; default every gene), `regulators` (list whose links among themselves are returned; default the candidates
  with role `regulator`), `background_genes` (list; default STRING's genome), `id_cache` (JSON file reused
  across calls), `write_to` (folder for `string_sig.tsv` + `string_genes.json`), `max_terms` (25, reply only).

## Output
- Success: `ok`, `source`, `n_genes`, `n_mapped`, `n_terms`, `ppi_enrichment`, `terms` (category, description, fdr,
  n_genes), `regulator_network` (n_regulators, n_mapped, background, ppi_enrichment, n_edges_by_type, edges: a, b,
  score, type, channels, channel_scores), `written_to`.
- Failure: `ok=false`, `error.code` in `invalid_json`, `invalid_input`, `network_error`.

## Execution
```bash
echo '{"genes":["TFAM","POLRMT","TEFM","MTPAP","LRPPRC"]}' | python scripts/string_enrichment.py
```
