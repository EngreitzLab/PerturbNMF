"""Promoter-neighbour geometry on a hand-built coordinate file."""
from gene_coordinates import (
    guide_positions_on_this_assembly, load_gene_tss, orientation, parse_guide_position, promoter_neighbours,
)

# name chrom start end strand gene_type — TSS is start on +, end on -
ROWS = [
    ("TARGET", "chr1", 10_000, 20_000, "+", "protein_coding"),
    ("DIVERGENT", "chr1", 2_000, 9_800, "-", "protein_coding"),       # TSS 9,800: head-to-head, 200 bp
    ("SAMESTRAND", "chr1", 12_900, 30_000, "+", "protein_coding"),     # TSS 12,900: 2.9 kb downstream
    ("DISTAL", "chr1", 60_000, 70_000, "+", "protein_coding"),
    ("OTHERCHROM", "chr2", 10_100, 11_000, "+", "protein_coding"),
    ("TARGET-READTHRU", "chr1", 10_000, 40_000, "+", "protein_coding"),
    ("PSEUDO", "chr1", 10_300, 10_900, "+", "processed_pseudogene"),
]


def write_coordinates(tmp_path):
    path = tmp_path / "gene_coordinates.tsv"
    path.write_text("".join("\t".join(map(str, r)) + "\n" for r in ROWS))
    return load_gene_tss(path)


def test_neighbours_within_window(tmp_path):
    tss = write_coordinates(tmp_path)
    found = {n["gene"]: n for n in promoter_neighbours("TARGET", tss, {}, window=3000)}
    assert set(found) == {"DIVERGENT", "SAMESTRAND"}  # not distal, other chrom, readthrough, pseudogene
    assert found["DIVERGENT"]["distance"] == 200 and found["DIVERGENT"]["orientation"] == "divergent"
    assert found["SAMESTRAND"]["orientation"] == "same strand"
    assert [n["gene"] for n in promoter_neighbours("TARGET", tss, {}, window=1000)] == ["DIVERGENT"]


def test_convergent_is_not_divergent():
    plus = {"tss": 10_000, "strand": "+"}
    assert orientation(plus, {"tss": 9_900, "strand": "-"}) == "divergent"
    assert orientation(plus, {"tss": 12_000, "strand": "-"}) == "opposite strand"


def test_guide_names_and_assembly_check(tmp_path):
    tss = write_coordinates(tmp_path)
    assert parse_guide_position("ACAA1_+_38178488.23-P1P2") == ("ACAA1", 38178488)
    assert parse_guide_position("non-targeting_00012") is None
    kept, dropped = guide_positions_on_this_assembly({"TARGET": [10_050, 10_120], "DISTAL": [900_000]}, tss)
    assert kept == {"TARGET": [10_050, 10_120]} and dropped == ["DISTAL"]


def test_igvf_guide_table(tmp_path):
    from gene_coordinates import load_guide_table_positions
    tss = write_coordinates(tmp_path)
    table = tmp_path / "guides.tsv"
    table.write_text(
        "guide_id\tspacer\ttargeting\tguide_chr\tguide_start\tguide_end\n"
        "TARGET__AAA\tAAA\tTRUE\tchr1\t10000\t10020\n"
        "TARGET_TSS2__CCC\tCCC\tTRUE\tchr1\t50000\t50020\n"      # alternative TSS: folded into TARGET
        "TARGET__GGG\tGGG\tTRUE\tchr7\t10000\t10020\n"           # wrong chromosome: dropped
        "non-targeting__TTT\tTTT\tFALSE\t\t\t\n"
    )
    assert load_guide_table_positions(table, tss) == {"TARGET": [10010, 50010]}


def test_same_strand_locus_at_the_target_tss_is_not_a_neighbour(tmp_path):
    rows = ROWS + [("ENSG00000285953", "chr1", 10_223, 90_000, "+", "protein_coding")]
    path = tmp_path / "coords.tsv"
    path.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    found = {n["gene"] for n in promoter_neighbours("TARGET", load_gene_tss(path), {}, window=3000)}
    assert "ENSG00000285953" not in found and "DIVERGENT" in found
