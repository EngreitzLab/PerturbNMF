# Bundled motif-database tables

Small metadata tables used by `motif_hit_calling.py` / `run_motif_enrichment.py` to map motifs to TF genes
and motif families. The motif PFM (MEME) files themselves are not bundled; download them (see
`../../README.md#resources`) and pass `--motif_file` or set `$PERTURBNMF_MOTIFCOMPENDIUM_MEME` /
`$PERTURBNMF_HOCOMOCO_V11_MEME`.

| File | Source | Notes |
|------|--------|-------|
| `MotifCompendium-Database-Human.metadata.2026-05-02_2ad26dc.tsv` | [kundajelab/MotifCompendium](https://github.com/kundajelab/MotifCompendium), `pipeline/data/MotifCompendium-Database-Human.metadata.tsv` at commit `2ad26dc` (2026-05-02) | Release of the current `MotifCompendium-Database-Human.meme.txt` (default FIMO database). Columns trimmed to `name, TF, readable_name, posneg, num_motifs, source, motif_string`. |
| `MotifCompendium-Database-Human.metadata.2025-09-24_5b20d47.tsv` | same repository and file at commit `5b20d47` (2025-09-24) | Release the ENCODE ChromBPNet TF-MoDISco reports were matched against. Columns trimmed to `name, TF, family, readable_name, posneg, num_motifs, source, motif_string`. |
| `MotifCompendium_LICENSE.txt` | kundajelab/MotifCompendium | MIT license covering the two MotifCompendium tables above. |
| `HOCOMOCOv11_full_annotation_HUMAN_mono.tsv` | [HOCOMOCO v11](https://hocomoco11.autosome.org/downloads) (`HOCOMOCOv11_full_annotation_HUMAN_mono.tsv`) | Used for TF genes and TFClass families of HOCOMOCO v11 motif ids. |

Motif indices change between MotifCompendium releases (e.g. `KLF_3`), so TF lists must come from the release the
motifs were named with; `motif_hit_calling.select_motifcompendium_metadata` picks the bundled release that contains the
most of the given motif names (`--motifcompendium_metadata` overrides).

## Attribution

- MotifCompendium: Kundaje Lab, MIT license (`MotifCompendium_LICENSE.txt`); method description at
  https://zenodo.org/records/17179111.
- HOCOMOCO v11: Kulakovskiy IV et al. HOCOMOCO: towards a complete collection of transcription factor binding models
  for human and mouse via large-scale ChIP-Seq analysis. *Nucleic Acids Research* 46, D252–D259 (2018). The
  annotation table is redistributed unchanged; see https://hocomoco11.autosome.org for the license terms of the
  HOCOMOCO data.
