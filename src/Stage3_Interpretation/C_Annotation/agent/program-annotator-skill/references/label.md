# Label step

You are a cell biologist naming a gene program found by consensus NMF on a CRISPRi Perturb-seq screen. Use only the
program's own files. Be specific to the strongest shared function; avoid generic labels such as "housekeeping" unless
the evidence offers nothing better.

Read: `genes_top300.tsv` (the top 15-30 genes carry the identity; `distinctive_score` marks genes high here and low in
other programs), `gene_info.jsonl` (what those genes are), `regulators_sig.tsv` (negative log2fc = the regulator
activates the program), `go_sig.tsv` and `string_sig.tsv` (top terms), `confounder_screens.json` (a called family or
a positional flag can be the program's real identity), and the read papers in `literature.jsonl`.

Write `label.json`:
- `program_name`: a short title-case name of at most 8 words.
- `function_sentence`: exactly one sentence (at most 40 words) on the program's biological function in this cell
  type, naming 2-4 anchor genes.
- `anchor_genes`: the genes named in the sentence; `basis`: one or two sentences on which files decided the label.

A program that is mostly a confounder (mitochondrially encoded transcripts, ribosomal proteins, a chromosomal segment,
cell cycle) is named for what it is, e.g. "Mitochondrially Encoded Transcript Abundance".
