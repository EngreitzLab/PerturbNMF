"""ProgramAnnotatorV3 prompt section E2 (optional TF motifs from Stage 2) and the motif helpers
the viewer shares. Motif-free prompts must be unchanged."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ProgramAnnotatorV3" / "scripts"))
import build_annotation_prompts as prompts  # noqa: E402

CONDITIONS = [{"label": "D0", "description": "hiPSC"}, {"label": "D1", "description": "primitive streak"}]
SETTINGS = {"dataset_name": "test screen", "cell_system": "hiPSCs", "assay": "CRISPRi Perturb-seq",
            "k": 2, "significance_label": "CRT adjusted p < 0.05", "annotation_role": "stem cell biologist",
            "condition_variable": "timepoint", "condition_design": "ordered"}
GENES = {0: ["POU5F1", "NANOG", "SOX2", "LIN28A", "DPPA4", "TDGF1"],
         1: ["MIXL1", "TBXT", "EOMES", "GSC", "MESP1", "LHX1"]}


def screen(program_id: int) -> dict:
    empty = {"n_overlap": 0, "genes": [], "expected": 0.1, "p_value": None}
    return {
        "positional": {"top_chromosome": "chr6", "genes_on_top_chromosome": 2, "n_genes_located": 6,
                       "expected_on_top_chromosome": 0.5, "fold_enrichment": 4.0, "binomial_p": "0.08",
                       "densest_10mb_window": {"chrom": "chr6", "start": 1, "count": 2, "genes": ["POU5F1"]}},
        "gene_biotypes": {"protein_coding": 6},
        "symbol_families": {"histone": {"n_overlap": 0, "genes": [], "expected": 0.5}},
        "marker_sets": {"heat_shock": empty},
        "cis_targets_in_top_genes": {"n_overlap": 0, "genes": []},
        "regulators": {"n_tested": 3, "n_significant": 2},
        "condition_composition": {
            "condition_order": ["D0", "D1"], "share_of_total": {"D0": 0.8, "D1": 0.2},
            "peak_condition": "D0", "peak_share": 0.8, "peak_over_second": 4.0, "peak_over_lowest": 4.0,
            "marker_overlap": {c: {**empty, "description": c} for c in ("D0", "D1")},
        },
    }


@pytest.fixture
def resources():
    loading = pd.DataFrame([
        {"Name": g, "Score": 1.0 - i / 10, "program_id": pid, "UniquenessScore": float(i)}
        for pid, genes in GENES.items() for i, g in enumerate(genes)
    ])
    regulators = pd.DataFrame([
        # program 0: SMAD2 significant on D0 only, TP53 on D1 only; NODAL never
        {"program_id": 0, "target_gene": "SMAD2", "condition": "D0", "log2_fc": -1.2, "significant": True, "adj_pval": 0.001},
        {"program_id": 0, "target_gene": "SMAD2", "condition": "D1", "log2_fc": -0.3, "significant": False, "adj_pval": 0.5},
        {"program_id": 0, "target_gene": "TP53", "condition": "D0", "log2_fc": 0.1, "significant": False, "adj_pval": 0.9},
        {"program_id": 0, "target_gene": "TP53", "condition": "D1", "log2_fc": 0.9, "significant": True, "adj_pval": 0.02},
        {"program_id": 0, "target_gene": "NODAL", "condition": "D0", "log2_fc": 0.2, "significant": False, "adj_pval": 0.7},
        {"program_id": 0, "target_gene": "NODAL", "condition": "D1", "log2_fc": 0.1, "significant": False, "adj_pval": 0.8},
        {"program_id": 1, "target_gene": "SMAD2", "condition": "D0", "log2_fc": 0.2, "significant": False, "adj_pval": 0.7},
        {"program_id": 1, "target_gene": "SMAD2", "condition": "D1", "log2_fc": -2.0, "significant": True, "adj_pval": 1e-5},
    ])
    activity = pd.DataFrame([
        {"program_id": 0, "condition": "D0", "mean_score": 0.8}, {"program_id": 0, "condition": "D1", "mean_score": 0.2},
        {"program_id": 1, "condition": "D0", "mean_score": 0.1}, {"program_id": 1, "condition": "D1", "mean_score": 0.9},
    ])
    ncbi = {
        "0": {"gene_summaries": {"POU5F1": "Encodes OCT4, a pluripotency transcription factor. Alternative splicing "
                                           "results in multiple transcript variants. " + "Filler text. " * 40},
              "evidence_snippets": {"POU5F1": ["OCT4 maintains pluripotency (PMID:12345678)"]}},
        "1": {"gene_summaries": {"MIXL1": "Homeobox gene of the primitive streak."}},
    }
    return {
        "loading": loading, "regulators": regulators, "activity": activity, "conditions": CONDITIONS,
        "enrichment": pd.DataFrame(columns=["program_id", "category", "description", "fdr", "inputGenes"]),
        "coordinates": {g: {"chrom": "chr6", "gene_type": "protein_coding"} for genes in GENES.values() for g in genes},
        "screens": {"0": screen(0), "1": screen(1)}, "ncbi": ncbi, "excluded_pmids": frozenset(),
    }


# ---- TF motifs (optional section E2) ------------------------------------------------------------
# Dimensions: motif tables configured / not configured (None or absent key); program present in
# the table / missing; significant / depleted / non-significant rows; more significant motifs than
# shown; candidate tiers shown / hidden; single condition / multi-condition.

MOTIF_ROWS = [
    # program, element_type, tf, fdr, enrichment, significant
    ("K10_0", "promoter", "KLF4", 1e-6, 2.5, True),
    ("K10_0", "promoter", "SOX2", 1e-4, 1.8, True),
    ("K10_0", "promoter", "ZFX", 1e-8, 0.3, False),   # depleted: never shown
    ("K10_0", "promoter", "AHR", 0.4, 1.1, False),    # not significant
    *[("K10_0", "promoter", f"TF{i}", 1e-3, 1.5, True) for i in range(6)],  # 8 significant in total
    ("K10_0", "enhancer", "TEAD1", 2e-3, 1.4, True),
    ("K10_1", "promoter", "EOMES", 1e-3, 1.6, True),
    ("K10_1", "enhancer", "AHR", 0.9, 1.0, False),
]
CANDIDATE_ROWS = [
    # program, element_type, tf, tf_gene_symbol, fdr, loading rank, knockdown log2FC, knockdown fdr, tier
    ("K10_0", "promoter", "KLF4", "KLF4", 1e-6, 120, -0.8, 1e-4, "motif+regulator"),
    ("K10_0", "promoter", "SOX2", "SOX2", 1e-4, 3, None, None, "motif+expressed_in_program"),
    ("K10_0", "promoter", "TF1", "TF1", 1e-3, None, None, None, "motif_only"),
]


def motif_tables():
    motifs = pd.DataFrame(MOTIF_ROWS, columns=["program", "element_type", "tf", "fdr", "enrichment", "significant"])
    candidates = pd.DataFrame(CANDIDATE_ROWS, columns=[
        "program", "element_type", "tf", "tf_gene_symbol", "fdr", "tf_program_loading_rank",
        "knockdown_log2fc", "knockdown_fdr", "evidence_tier"])
    return motifs, candidates


def write_motif_tables(directory: Path) -> dict:
    """The two Stage 2 TSVs in `directory`; returns the config keys that name them."""
    motifs, candidates = motif_tables()
    motifs.to_csv(directory / "motif_enrichment.tsv", sep="\t", index=False)
    candidates.to_csv(directory / "candidate_tfs.tsv", sep="\t", index=False)
    return {"motif_enrichment": "motif_enrichment.tsv", "candidate_tfs": "candidate_tfs.tsv"}


def with_motifs(resources, tmp_path):
    loaded = prompts.read_motif_tables(write_motif_tables(tmp_path), tmp_path)
    assert set(loaded) == {"motif_enrichment", "candidate_tfs", "motif_test"}, f"read_motif_tables returned {set(loaded)}"
    assert loaded["motif_test"] == {"method": "ttest", "n_top": 300}, "no Stage 2 config -> Stage 2 defaults"
    return {**resources, **loaded}


def user_message(request: dict) -> str:
    return request["params"]["messages"][0]["content"]


def test_prompts_without_motif_tables_are_unchanged(resources, tmp_path):
    assert prompts.read_motif_tables({}, tmp_path) == {}, "no motif_enrichment key must load nothing"
    for pid in (0, 1):
        plain = prompts.build_prompt(pid, resources, SETTINGS)
        text = json.dumps(plain)
        assert "E2." not in text and "motif" not in text.lower(), f"P{pid}: motif text leaked into a motif-free prompt"
        assert prompts.build_prompt(pid, {**resources, "motif_enrichment": None}, SETTINGS) == plain, \
            f"P{pid}: motif_enrichment=None must give the same prompt as no key"


def test_section_e2_lists_motif_families_with_their_candidates(resources, tmp_path):
    """Table without motif_family (older Stage 2 output): each motif is its own family."""
    request = prompts.build_prompt(0, with_motifs(resources, tmp_path), SETTINGS)
    user = user_message(request)
    assert user.index("## E. Explanation-class screens") < user.index("## E2.") < user.index("## F. REFERENCE POOL"), \
        "E2 must sit between the screens (E) and the reference pool (F)"
    heading, guide, *lines = user.split("## E2. ")[1].split("## F. REFERENCE POOL")[0].strip().splitlines()
    assert guide == prompts.format_motif_guide() and "Correlative" in guide, "E2 opens with the how-to-read line"
    assert "of the top 300 genes versus expressed genes" in guide, "no Stage 2 config -> t-test on the top 300"
    assert "grouped by family" in guide
    section = "\n".join(lines)
    assert heading == "TF motifs in program promoters/enhancers (correlative)", f"heading: {heading!r}"
    assert lines == [
        "Promoter (8 of 10 motifs significant in 8 families; top 5 shown):",
        "- KLF4 2.50x FDR=1.0e-06 | candidate TFs: KLF4 (motif+regulator; knockdown log2FC=-0.80, adj p=1.0e-04)",
        "- SOX2 1.80x FDR=1.0e-04 | candidate TFs: SOX2 (motif+expressed_in_program, loading rank 3)",
        "- TF0 1.50x FDR=1.0e-03",
        "- TF1 1.50x FDR=1.0e-03",
        "- TF2 1.50x FDR=1.0e-03",
        "Enhancer (1 of 1 motifs significant in 1 family):",
        "- TEAD1 1.40x FDR=2.0e-03",
    ], lines
    assert "ZFX" not in section and "AHR" not in section, "depleted / non-significant motifs must not be shown"
    assert "motif_only" not in section, "weak candidate tiers are hidden"
    plain = prompts.build_prompt(0, resources, SETTINGS)
    assert request["params"]["system"] == plain["params"]["system"], "motifs must not change the system prompt"


def test_section_e2_edge_cases_and_multi_condition(resources, tmp_path):
    loaded = with_motifs(resources, tmp_path)  # the fixture is a multi-condition screen
    user = user_message(prompts.build_prompt(1, loaded, SETTINGS))
    assert "Enhancer (0 of 1 motifs significant): none" in user, "a tested element type with no hit must say none"
    assert "Candidate TFs: none (no enriched motif lists an expressed TF)" in user
    absent = {**loaded, "motif_enrichment": loaded["motif_enrichment"][loaded["motif_enrichment"]["program_id"] == 0]}
    user = user_message(prompts.build_prompt(1, absent, SETTINGS))
    assert (f"## E2. TF motifs in program promoters/enhancers (correlative)\n{prompts.format_motif_guide()}\n"
            "No motif enrichment results for this program.\n\n## F.") in user, "a program missing from the motif table must say so"
    for pid in (0, 1):
        request = prompts.build_prompt(pid, loaded, SETTINGS)
        assert request["params"]["system"] == prompts.build_prompt(pid, resources, SETTINGS)["params"]["system"], \
            f"P{pid}: motifs must not change the multi-condition system prompt"
        assert "C2. Program activity by condition" in user_message(request) and "## E2." in user_message(request)


def two_source_motif_tables():
    """The fixture tables as the FIMO rows plus a Fi-NeMo copy (other TFs / FDRs), as Stage 2
    --motif_source both writes them (Fi-NeMo rows first, to check the FIMO-first ordering)."""
    motifs, candidates = motif_tables()
    finemo_motifs = pd.DataFrame([("K10_0", "promoter", "KLF-SP", 1e-5, 1.9, True),
                                  ("K10_0", "enhancer", "ETS", 1e-9, 1.6, True),
                                  ("K10_0", "enhancer", "GATA", 1e-3, 1.3, True),
                                  ("K10_1", "promoter", "NFY", 0.5, 1.0, False)], columns=motifs.columns)
    finemo_candidates = pd.DataFrame([("K10_0", "enhancer", "GATA", "GATA2", 1e-3, 5, None, None,
                                       "motif+expressed_in_program")], columns=candidates.columns)
    motifs = pd.concat([finemo_motifs.assign(motif_source="finemo"), motifs.assign(motif_source="fimo")])
    candidates = pd.concat([finemo_candidates.assign(motif_source="finemo"), candidates.assign(motif_source="fimo")])
    return motifs, candidates


def section_e2(user: str) -> list:
    """The E2 lines after the heading and the how-to-read line."""
    return user.split("## E2. ")[1].split("## F. REFERENCE POOL")[0].strip().splitlines()[2:]


def loaded_motif_tables(motifs, candidates, directory: Path, resources) -> dict:
    motifs.to_csv(directory / "motif_enrichment.tsv", sep="\t", index=False)
    candidates.to_csv(directory / "candidate_tfs.tsv", sep="\t", index=False)
    config = {"motif_enrichment": "motif_enrichment.tsv", "candidate_tfs": "candidate_tfs.tsv"}
    return {**resources, **prompts.read_motif_tables(config, directory)}


FAMILY_MOTIF_ROWS = [
    # program, element_type, tf, motif_family, fdr, enrichment, significant
    ("K10_0", "promoter", "KLF-SP_0", "KLF-SP", 1e-6, 2.5, True),
    ("K10_0", "promoter", "GATA_0", "GATA", 1e-5, 1.4, True),
    ("K10_0", "promoter", "KLF_8", "KLF", 1e-4, 1.8, True),
    ("K10_0", "promoter", "KLF-SP_1", "KLF-SP", 1e-3, 1.5, True),
    ("K10_0", "promoter", "KLF-SP_2", "KLF-SP", 2e-3, 1.3, True),
    ("K10_0", "promoter", "KLF-SP_3", "KLF-SP", 3e-3, 1.2, True),
    ("K10_0", "promoter", "ETV_0", "ETV", 0.3, 1.1, False),
]
FAMILY_CANDIDATE_ROWS = [
    # program, element_type, tf, motif_family, tf_gene_symbol, fdr, loading rank, KD log2FC, KD fdr, tier
    ("K10_0", "promoter", "KLF-SP_0", "KLF-SP", "KLF2", 1e-6, 12, None, None, "motif+expressed_in_program"),
    ("K10_0", "promoter", "KLF-SP_0", "KLF-SP", "KLF4", 1e-6, 115, -0.3, 1e-3, "motif+regulator"),
    ("K10_0", "promoter", "KLF-SP_0", "KLF-SP", "SP1", 1e-6, None, None, None, "motif+expressed"),
    ("K10_0", "promoter", "KLF-SP_1", "KLF-SP", "KLF2", 1e-3, 12, None, None, "motif+expressed_in_program"),
    ("K10_0", "promoter", "GATA_0", "GATA", "GATA2", 1e-5, 634, None, None, "motif+expressed"),
]


def family_motif_tables():
    motifs = pd.DataFrame(FAMILY_MOTIF_ROWS, columns=["program", "element_type", "tf", "motif_family", "fdr",
                                                      "enrichment", "significant"])
    candidates = pd.DataFrame(FAMILY_CANDIDATE_ROWS, columns=[
        "program", "element_type", "tf", "motif_family", "tf_gene_symbol", "fdr", "tf_program_loading_rank",
        "knockdown_log2fc", "knockdown_fdr", "evidence_tier"])
    return motifs, candidates


def test_section_e2_groups_motifcompendium_clusters_by_family(resources, tmp_path):
    loaded = loaded_motif_tables(*family_motif_tables(), tmp_path, resources)
    lines = section_e2(user_message(prompts.build_prompt(0, loaded, SETTINGS)))
    assert lines == [
        "Promoter (6 of 7 motifs significant in 3 families):",
        "- KLF-SP: KLF-SP_0 2.50x FDR=1.0e-06, KLF-SP_1 1.50x FDR=1.0e-03, KLF-SP_2 1.30x FDR=2.0e-03 (+1 more) | "
        "candidate TFs: KLF4 (motif+regulator; knockdown log2FC=-0.30, adj p=1.0e-03); "
        "KLF2 (motif+expressed_in_program, loading rank 12); SP1 (motif+expressed)",
        "- GATA: GATA_0 1.40x FDR=1.0e-05 | candidate TFs: GATA2 (motif+expressed)",
        "- KLF: KLF_8 1.80x FDR=1.0e-04",
        "Enhancer (0 of 0 motifs significant): none",
    ], lines
    selection = prompts.select_program_motifs(0, loaded["motif_enrichment"], loaded["candidate_tfs"])
    klf = selection["families"]["promoter"][0]
    assert klf["family"] == "KLF-SP" and klf["n_significant"] == 4
    assert [c["tf"] for c in klf["candidates"]] == ["KLF4", "KLF2", "SP1"], "tier first; KLF2 once"
    assert klf["candidates"][1]["motif"] == "KLF-SP_0"


def test_section_e2_keeps_motif_sources_apart_fimo_first(resources, tmp_path):
    loaded = loaded_motif_tables(*two_source_motif_tables(), tmp_path, resources)
    lines = section_e2(user_message(prompts.build_prompt(0, loaded, SETTINGS)))
    assert lines[0] == ("Motif sources, tested separately: FIMO/HOCOMOCO = HOCOMOCO v11 motif scan; Fi-NeMo = "
                        "ChromBPNet motif calls in accessible chromatin, named by the matched database cluster."), lines[0]
    assert lines[1] == "Promoter (FIMO/HOCOMOCO) (8 of 10 motifs significant in 8 families; top 5 shown):", lines[1]
    assert lines[2].startswith("- KLF4 2.50x FDR=1.0e-06 | candidate TFs: KLF4 (motif+regulator"), lines[2]
    enhancer = lines.index("Enhancer (FIMO/HOCOMOCO) (1 of 1 motifs significant in 1 family):")
    assert lines[enhancer + 1] == "- TEAD1 1.40x FDR=2.0e-03"
    finemo = lines.index("Promoter (Fi-NeMo) (1 of 1 motifs significant in 1 family):")
    assert lines[finemo + 1] == "- KLF-SP 1.90x FDR=1.0e-05", "Fi-NeMo counts must not include FIMO rows"
    assert lines[finemo + 2] == "Enhancer (Fi-NeMo) (2 of 2 motifs significant in 2 families):"
    assert lines[finemo + 3:] == ["- ETS 1.60x FDR=1.0e-09",
                                  "- GATA 1.30x FDR=1.0e-03 | candidate TFs: GATA2 (motif+expressed_in_program, loading rank 5)"]


def test_fimo_label_follows_the_motif_database(resources, tmp_path):
    motifs, candidates = family_motif_tables()
    finemo = motifs.assign(motif_source="finemo")
    loaded = loaded_motif_tables(pd.concat([motifs.assign(motif_source="fimo"), finemo]),
                                 candidates.assign(motif_source="fimo"), tmp_path, resources)
    lines = section_e2(user_message(prompts.build_prompt(0, loaded, SETTINGS)))
    assert lines[0].startswith("Motif sources, tested separately: FIMO/MotifCompendium = motif scan with "
                               "MotifCompendium database clusters; Fi-NeMo"), lines[0]
    assert lines[1].startswith("Promoter (FIMO/MotifCompendium) (6 of 7"), lines[1]


def test_section_e2_single_motif_source_column_is_unchanged(resources, tmp_path):
    motifs, candidates = motif_tables()
    plain = section_e2(user_message(prompts.build_prompt(0, with_motifs(resources, tmp_path), SETTINGS)))
    one_source = loaded_motif_tables(motifs.assign(motif_source="finemo"), candidates.assign(motif_source="finemo"),
                                     tmp_path, resources)
    assert section_e2(user_message(prompts.build_prompt(0, one_source, SETTINGS))) == plain, \
        "a motif_source column with one value must give the same section as no column"


def test_family_members_sharing_an_fdr_are_ordered_by_loading_rank():
    motifs = pd.DataFrame([("K10_0", "enhancer", "KLF_0", "KLF", 1e-8, 2.0, True)],
                          columns=["program", "element_type", "tf", "motif_family", "fdr", "enrichment", "significant"])
    motifs["program_id"] = 0
    columns = ["program_id", "element_type", "tf", "tf_gene_symbol", "fdr", "tf_program_loading_rank", "evidence_tier"]
    family = pd.DataFrame([(0, "enhancer", "KLF_0", f"KLF{i}", 1e-8, rank, "motif+expressed_in_program")
                           for i, rank in [(10, 184), (13, 254), (2, 1), (3, 92), (4, 115), (6, 182), (7, 290),
                                           (9, 200), (11, 250)]], columns=columns)
    chosen = [c["tf"] for c in prompts.select_program_motifs(0, motifs, family)["families"]["enhancer"][0]["candidates"]]
    assert chosen == ["KLF2", "KLF3", "KLF4", "KLF6", "KLF10", "KLF9"][:prompts.CANDIDATES_PER_FAMILY], \
        f"same FDR -> loading rank order, capped at {prompts.CANDIDATES_PER_FAMILY}; got {chosen}"


def test_motif_test_wording_follows_the_stage2_method(resources, tmp_path):
    keys = write_motif_tables(tmp_path)
    (tmp_path / "motif_enrichment_config.yml").write_text(json.dumps({"arguments": {"motif_method": "ttest", "n_top": 200}}))
    loaded = {**resources, **prompts.read_motif_tables(keys, tmp_path)}
    assert loaded["motif_test"] == {"method": "ttest", "n_top": 200}, "n_top read from the Stage 2 config next to the table"
    assert "of the top 200 genes versus expressed genes" in user_message(prompts.build_prompt(0, loaded, SETTINGS))

    loaded = {**resources, **prompts.read_motif_tables({**keys, "motif_method": "correlation"}, tmp_path)}
    request = prompts.build_prompt(0, loaded, SETTINGS)
    guide = user_message(request).split("## E2. ")[1].splitlines()[1]
    assert "correlates positively with the gene's program loading" in guide and "(FDR < 0.05, r > 0)" in guide
    assert "top 200 genes" not in guide and "enrichment > 1" not in guide
    assert section_e2(user_message(request))[1].startswith("- KLF4 r=2.50 FDR=1.0e-06")


def test_program_number_accepts_ints_strings_and_prefixed_ids():
    got = [prompts.program_number(v) for v in (12, "12", "K10_12", "program")]
    assert got == [12, 12, 12, None], f"program_number gave {got}"
