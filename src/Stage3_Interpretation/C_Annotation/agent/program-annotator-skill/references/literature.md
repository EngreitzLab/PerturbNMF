# Literature step

Goal: the papers that say what the program's key genes and regulators do, read, with one finding each. Search hits
are candidates; only papers you read (full text, or the abstract when no PDF exists) become evidence.

1. Order the genes: regulators by adj_p (`regulators_sig.tsv`), then the top 8-10 program genes, then distinctive
   genes that look informative.
2. For each, call gene-function-papers-skill (pass its aliases from `gene_info.jsonl`, the program's cell type,
   `n_papers` 3-5, `n_pdfs` 1-2, `pdf_dir: Programs/P<k>/papers`, `write_to: Programs/P<k>/literature.jsonl`).
   Use generif-papers-skill when a gene's function papers are few or off-topic.
3. For the strongest regulator-program links, search the link itself with pubmed-skill (e.g.
   `LRPPRC[tiab] AND (mitochondrial mRNA OR polyadenylation)`), and with openalex-skill when PubMed finds little.
   For pairs that interactions.jsonl lists as found, the OmniPath PMIDs are good candidates (fetch them by PMID).
4. Read: paper-fetch-skill `fetch` (when no PDF yet), then `read` pages 1-3. Record with `note`: `finding` = one or two
   sentences on what the paper shows the gene does, with assay and system; `stance` toward the program's emerging
   theme; `context` = species / cell type. Use `status: irrelevant` for off-topic papers.
5. Stop at 8-15 read papers or when new searches return nothing new. Then close the step.
