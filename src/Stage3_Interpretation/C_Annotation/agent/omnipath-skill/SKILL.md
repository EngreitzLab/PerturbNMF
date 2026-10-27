---
name: omnipath-skill
description: Test every perturbation-regulator x gene pair against OmniPath (transcriptional, post-translational, pathway, ligand-receptor, miRNA and lncRNA interactions, both directions) and record found and not-found pairs with direction, sign, resources and PMIDs. Use for the interactions step of a program, or to ask whether a regulator is known to act on a gene directly.
---

## Operating rules
- Use `scripts/omnipath.py`. For a program: `genes` = its top 15-20 genes (`genes_top300.tsv`), `regulators` = every
  regulator in `regulators_sig.tsv` (as {symbol: stats} to keep log2FC), `gene_info` = the program's
  `gene_info.jsonl` (aliases are then tried too), `write_to` = `Programs/P<k>/interactions.jsonl`.
- Every tested pair is written, `found: false` included: "no curated interaction" is a result (the effect is then
  more likely indirect), not missing data.
- A found pair's PMIDs are curated references for the interaction and may be cited for it.

## Input
One JSON object on stdin. Required: `genes`, `regulators` (list or {symbol: stats}). Optional: `gene_info`
(gene_info.jsonl path), `gene_category` ({gene: program_gene | distinctive_gene}), `datasets` (all), `max_pmids` (20),
`use_aliases` (true), `taxid` (9606), `write_to`.

## Output
`ok`, `source`, `n_pairs_tested`, `n_pairs_found`, and either `pairs` (every tested pair) or `written_to`, `added`,
`found` (the found pairs, compact).

## Execution
```bash
echo '{"genes":["TPT1","MT-CO1"],"regulators":["TP53","TFAM"]}' | python scripts/omnipath.py
```
