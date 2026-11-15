# Label step

You are a cell biologist naming a gene program found by consensus NMF on a CRISPRi Perturb-seq screen. This program
is one of K programs from the same experiment, and many of them share a broad theme (several cell-cycle
programs, several ECM programs). A label is only useful if a reader can tell which program it names, so name what sets
this program apart, not the theme it shares with its siblings. You see only this program's files; a later set-level
pass (2.Synthesis/resolve_label_collisions.py) compares every label and rewrites the ones that still collide.

Read: `genes_top300.tsv` (the top 15-30 genes carry the identity; `distinctive_score` marks genes high here and low in
other programs, the best source of a distinguisher), `gene_info.jsonl` (what those genes are), `go_sig.tsv` and
`string_sig.tsv` (top terms), `confounder_screens.json` (a called family or a positional flag can be the program's
real identity), the read papers in `literature.jsonl`, and `regulators_sig.tsv` (negative log2fc = the regulator
activates the program).

## Name it
- **Family first.** `label_family` is the plainest standard name of the broad process (`Cell cycle`, `Translation`,
  `Extracellular matrix`, `Interferon response`, `Unfolded protein response`), so sibling programs share the exact string. Use the
  established name, not a paraphrase.
- **Then the distinguisher.** `label_distinguisher` is a process, pathway, compartment, phase or state term (`G2M`,
  `Tip cell`, `Golgi`, `Hypoxia`) that the top and distinctive genes support. It is not a bare
  gene symbol and carries no gene in parentheses. The only exception is a gene that is the accepted name of the
  process (`ATF4 amino acid stress response`, `NRF2 oxidative stress response`).
- **`program_name`** is `<family> - <distinguisher>`, or the bare family when nothing defends a distinguisher. A
  program made of 2-3 distinct processes is named as a comma list of them (`Lysosome, OXPHOS, cholesterol synthesis`),
  with `label_family` and `label_distinguisher` empty strings.
- **Name it from its genes, not its regulators.** Regulators are evidence for the mechanism (interpretation step),
  not the name. A regulator that is significant for many programs (C4 of `regulator_context.md`) says
  nothing about which program this is. Put a regulator in the name only when it
  is the accepted name of the process.
- **Use the top genes.** At least two of the top 10 genes by loading must be among `anchor_genes`. If the top genes
  point somewhere the GO/STRING terms do not, follow the genes.
- **Conservatism overrides specificity.** Claim only what the genes support. A sharper name that overstates the
  evidence is worse than a plainer one that does not. Prefer the established name of a process over a
  family-plus-modifier you construct.
- 2-4 words is best, 6 at most (the ` - ` is not a word). Sentence case. Do not use "program", "process",
  "regulation of", "signature", "module", or quality words ("mixed", "heterogeneous", "unclear", "miscellaneous",
  "grab bag").

A program that is mostly a confounder (mitochondrially encoded transcripts, ribosomal proteins, a chromosomal segment,
cell cycle) is named for what it is, e.g. `Mitochondrial transcripts`.

## Write `label.json` (references/schemas.md)
- `program_name`, `label_family`, `label_distinguisher` as above.
- `distinguisher_evidence`: the genes (top or distinctive) the distinguisher rests on; empty when there is none.
- `function_sentence`: exactly one sentence (at most 40 words) on the program's biological function in this cell
  type, naming 2-4 anchor genes.
- `anchor_genes`: the genes named in the sentence; `basis`: one or two sentences on which files decided the label.
