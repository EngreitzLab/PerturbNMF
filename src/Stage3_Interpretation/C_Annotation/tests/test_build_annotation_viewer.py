"""ProgramAnnotatorV3 viewer data (build_annotation_viewer.load_from_config -> per-program dict).

Dimensions: motif tables configured / not configured. Behaviour: each program carries `motifs`,
equal to the selection prompt section E2 showed (None without motif tables), JSON-serialisable;
META names the motif source (the optional `motif_enrichment_label`, else the file name).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ProgramAnnotatorV3" / "scripts"))
import build_annotation_prompts as prompts  # noqa: E402
import build_annotation_viewer as viewer  # noqa: E402
from test_annotation_prompt_motif_section import SETTINGS, motif_tables, two_source_motif_tables, write_motif_tables  # noqa: E402


def build_program_zero(tmp_path: Path, extra_config: dict):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    pd.DataFrame([{"Name": g, "Score": 1.0 - i / 10, "program_id": 0, "UniquenessScore": float(i)}
                  for i, g in enumerate(["KLF4", "SOX2", "NANOG"])]).to_csv(data / "loading.csv", index=False)
    pd.DataFrame([{"program_id": 0, "target_gene": "KLF4", "log2_fc": -0.8, "significant": True,
                   "adj_pval": 1e-4}]).to_csv(data / "regulators.csv", index=False)
    dispatch = tmp_path / "dispatch" / "v3_p0"
    dispatch.mkdir(parents=True, exist_ok=True)
    (dispatch / "answer.json").write_text(json.dumps({"program_id": 0, "label": "Pluripotency"}))
    (dispatch / "prompt.md").write_text("KLF4 SOX2 NANOG\n# TASK\n")
    extra = extra_config(data) if callable(extra_config) else extra_config
    config = {"data_dir": str(data), "gene_loading": "loading.csv", "regulators": "regulators.csv",
              "settings": SETTINGS, **extra}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = argparse.Namespace(config=path, dispatch=tmp_path / "dispatch", arm="v3", citations=None)
    build, meta, _ = viewer.load_from_config(args)
    return build(0), meta


def test_no_motif_tables_gives_no_motif_field(tmp_path):
    program, meta = build_program_zero(tmp_path, {})
    assert program["motifs"] is None, f"expected motifs=None without motif tables, got {program['motifs']!r}"
    assert meta["motif_source"] == "", f"expected empty motif_source, got {meta['motif_source']!r}"


def test_motif_field_is_the_prompt_selection(tmp_path):
    program, meta = build_program_zero(
        tmp_path, lambda data: {**write_motif_tables(data), "motif_enrichment_label": "demo table"})
    motifs, candidates = motif_tables()
    motifs = motifs[motifs["program"] == "K10_0"].assign(program_id=0)
    candidates = candidates[candidates["program"] == "K10_0"].assign(program_id=0)
    expected = prompts.select_program_motifs(0, motifs, candidates)
    assert program["motifs"] == expected, f"viewer motifs {program['motifs']} != prompt selection {expected}"
    families = program["motifs"]["families"]["promoter"]
    assert families[0]["motifs"] == [["KLF4", 2.5, 1e-06]] and families[0]["family"] == "KLF4", families[0]
    assert [c["tf"] for f in families for c in f["candidates"]] == ["KLF4", "SOX2"], families
    assert meta["motif_source"] == "demo table", f"motif_source: {meta['motif_source']!r}"
    json.loads(json.dumps(program))  # embedded in the page as JSON


def test_two_motif_sources_get_separate_sections(tmp_path):
    def write_two_source_tables(data):
        motifs, candidates = two_source_motif_tables()
        motifs.to_csv(data / "motif_enrichment.tsv", sep="\t", index=False)
        candidates.to_csv(data / "candidate_tfs.tsv", sep="\t", index=False)
        return {"motif_enrichment": "motif_enrichment.tsv", "candidate_tfs": "candidate_tfs.tsv"}
    program, _ = build_program_zero(tmp_path, write_two_source_tables)
    motifs = program["motifs"]
    assert motifs["sections"] == [["promoter_fimo", "promoter", "FIMO/HOCOMOCO"],
                                  ["enhancer_fimo", "enhancer", "FIMO/HOCOMOCO"],
                                  ["promoter_finemo", "promoter", "Fi-NeMo"],
                                  ["enhancer_finemo", "enhancer", "Fi-NeMo"]], motifs["sections"]
    assert motifs["n_significant"] == {"promoter_fimo": 8, "enhancer_fimo": 1, "promoter_finemo": 1,
                                       "enhancer_finemo": 2}, motifs["n_significant"]
    assert motifs["section_sources"] == {"promoter_fimo": "fimo", "enhancer_fimo": "fimo", "promoter_finemo": "finemo",
                                         "enhancer_finemo": "finemo"}
    assert [c["tf"] for f in motifs["families"]["enhancer_finemo"] for c in f["candidates"]] == ["GATA2"]
    assert "m.sections ||" in viewer.PAGE, "the page must render one block per section"


def test_logos_only_for_shown_motifs(tmp_path):
    """`motif_logos` (Stage 2 {K}_motif_logos.json): META keeps only the logos of shown motifs, '<source>|<motif>'."""
    def write_tables_and_logos(data):
        keys = write_motif_tables(data)
        logos = {"version": 1, "logos": {"fimo": {
            "KLF4": {"kind": "information_content", "motif_id": "KLF4_HUMAN.H11MO.0.A", "matrix": [[0, 0, 1.5, 0]]},
            "AHR": {"kind": "information_content", "motif_id": "AHR_HUMAN.H11MO.0.B", "matrix": [[1, 0, 0, 0]]}}}}
        (data / "logos.json").write_text(json.dumps(logos))
        return {**keys, "motif_logos": "logos.json"}
    program, meta = build_program_zero(tmp_path, write_tables_and_logos)
    shown = viewer.select_shown_logos({0: program}, meta["motif_logos_all"])
    assert shown == {"fimo|KLF4": {"kind": "information_content", "matrix": [[0, 0, 1.5, 0]]}}, \
        "AHR is not significant -> not shown -> no logo; single-source table uses the file's only source"
    assert meta["logo_default_source"] == "fimo"
    assert "function logoSvg" in viewer.PAGE
