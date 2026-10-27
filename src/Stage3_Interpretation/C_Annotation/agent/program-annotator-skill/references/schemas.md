# Output files of the annotate steps

All files are written in `Programs/P<k>/`. Lists of genes, regulators and PMIDs are checked against the program's
evidence when the step is ticked.

## label.json
```json
{"program_name": "Mitochondrial OXPHOS and Cytoplasmic Translation",
 "function_sentence": "One sentence naming 2-4 anchor genes such as MT-CO1, TFAM and RPL7.",
 "anchor_genes": ["MT-CO1", "TFAM", "RPL7"],
 "basis": "Which files decided the label."}
```

## driver_genes.json (optional step; references/driver_genes.md)
```json
{"annotation": "<label.json program_name>", "method": "agent-literature",
 "genes": {"TFAM": {"role": "regulator", "verdict": "supported", "cell_type_evidence": "other",
                    "reason": "one sentence from the papers read",
                    "pmids": ["16543724"],
                    "paper_findings": [{"pmid": "16543724", "title": "<title as in literature.jsonl>",
                                        "finding": "one sentence"}]}},
 "explained": ["TFAM"], "relabel_suggested": false, "note": ""}
```
role: program gene | distinctive gene | regulator. verdict: supported | mixed | contradicted | insufficient.
cell_type_evidence: same | related | other | none. Every PMID in `pmids` and `paper_findings` must be a read paper
(`status: read`) in `literature.jsonl`.

## interpretation.json
```json
{"label": "<label.json program_name>",
 "confounders": [{"confounder": "positional", "status": "contributing", "evidence": "10 of 50 top genes on chrM (E31)"}],
 "interpretation": {
   "upstream_trigger": null,
   "coregulation_mechanism": {"claim": "...", "mechanism_kind": "shared locus", "support_genes": ["MT-CO1"],
                              "support_regulators": ["TFAM"], "support_terms": ["..."], "confidence": "high",
                              "pmids": [], "evidence_ids": ["E06", "E21"]},
   "cellular_output": null},
 "modules": [{"name": "...", "genes": ["MT-CO1", "MT-ND1"], "mechanism": "...", "strength": "supported",
              "evidence_ids": ["E31"]}],
 "coherence": "partial"}
```
All eight confounders must be listed. Every non-null slot needs at least one evidence id.
