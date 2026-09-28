"""Unit tests for Stage2_Evaluation/A_Metrics/src/motif_hit_calling.py (tiny synthetic GTF / FASTA / links).

The module is imported by file path so the tests run without the A_Metrics package's heavy imports.
FIMO scanning tests run only where a backend is installed (MEME ``fimo`` on PATH or FIMO_BINARY, or memelite).
"""

import gzip
import importlib.util
import os
import shutil

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "Stage2_Evaluation",
                           "A_Metrics", "src", "motif_hit_calling.py")
spec = importlib.util.spec_from_file_location("motif_hit_calling", MODULE_PATH)
motif_hit_calling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motif_hit_calling)

GTF_LINES = [
    # + gene with a canonical transcript starting downstream of the gene start
    'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding"; gene_name "PLUS";',
    'chr1\tHAVANA\ttranscript\t1001\t2000\t.\t+\t.\tgene_id "G1"; transcript_id "T1a"; gene_name "PLUS";',
    'chr1\tHAVANA\ttranscript\t1101\t2000\t.\t+\t.\tgene_id "G1"; transcript_id "T1b"; gene_name "PLUS"; tag "basic"; tag "Ensembl_canonical";',
    'chr1\tHAVANA\texon\t1101\t1200\t.\t+\t.\tgene_id "G1"; transcript_id "T1b"; gene_name "PLUS";',
    # - gene without canonical tag -> gene end
    'chr1\tHAVANA\tgene\t3001\t4000\t.\t-\t.\tgene_id "G2"; gene_type "lncRNA"; gene_name "MINUS";',
    # duplicated symbol: lncRNA copy on chr1 vs protein_coding copy on chr2 -> keep protein_coding
    'chr1\tHAVANA\tgene\t5001\t5100\t.\t+\t.\tgene_id "G3a"; gene_type "lncRNA"; gene_name "DUP";',
    'chr2\tHAVANA\tgene\t501\t600\t.\t+\t.\tgene_id "G3b"; gene_type "protein_coding"; gene_name "DUP";',
]


@pytest.fixture
def gtf_path(tmp_path):
    path = tmp_path / "genes.gtf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("##description: test\n" + "\n".join(GTF_LINES) + "\n")
    return str(path)


def test_read_gtf_gene_tss_canonical_strand_and_duplicates(gtf_path):
    tss = motif_hit_calling.read_gtf_gene_tss(gtf_path).set_index("gene")
    assert sorted(tss.index) == ["DUP", "MINUS", "PLUS"]
    assert tss.loc["PLUS", "tss"] == 1100 and tss.loc["PLUS", "is_canonical_tss"]   # 0-based canonical start
    assert tss.loc["MINUS", "tss"] == 3999 and tss.loc["MINUS", "strand"] == "-"      # 0-based gene end base
    assert tss.loc["DUP", "chrom"] == "chr2"
    only_coding = motif_hit_calling.read_gtf_gene_tss(gtf_path, gene_types=["protein_coding"])
    assert sorted(only_coding["gene"]) == ["DUP", "PLUS"]


def test_build_promoter_regions_strand_aware():
    gene_tss = pd.DataFrame({"chrom": ["chr1", "chr1"], "tss": [1100, 3999], "strand": ["+", "-"],
                             "gene": ["PLUS", "MINUS"]})
    regions = motif_hit_calling.build_promoter_regions(gene_tss, upstream=250, downstream=50).set_index("gene")
    assert tuple(regions.loc["PLUS", ["start", "end"]]) == (850, 1151)     # TSS-250 .. TSS+50 inclusive
    assert tuple(regions.loc["MINUS", ["start", "end"]]) == (3949, 4250)   # upstream is to the right
    assert ((regions["end"] - regions["start"]) == 301).all()
    assert (regions["sequence_name"] == regions.index).all()


def test_build_promoter_regions_schnitzler2024_matches_paper_bed(tmp_path):
    # Paper: TSS500bp.bed = [TSS-250, TSS+250) with TSS = start (+) / end (-); then [start, start+301).
    bed = tmp_path / "bounds.bed"
    bed.write_text("chr1\t11873\t14409\tDDX11L1\t0\t+\nchr1\t14361\t29370\tWASH7P\t0\t-\n"
                   "chr1\t14361\t29370\tWASH7P\t0\t-\n")
    gene_tss = motif_hit_calling.read_bed_gene_tss(str(bed), window_mode="schnitzler2024")
    assert len(gene_tss) == 2
    regions = motif_hit_calling.build_promoter_regions(gene_tss, window_mode="schnitzler2024").set_index("gene")
    assert tuple(regions.loc["DDX11L1", ["start", "end"]]) == (11623, 11924)   # paper hg19: 11623 11924
    assert tuple(regions.loc["WASH7P", ["start", "end"]]) == (29120, 29421)    # paper hg19: 29120 29421
    with pytest.raises(ValueError):
        motif_hit_calling.build_promoter_regions(gene_tss, window_mode="bogus")


ABC_HEADER = "chr\tstart\tend\tname\tclass\tTargetGene\tABC.Score\n"


def test_read_abc_links_tsv_drops_promoters_and_thresholds(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_text(ABC_HEADER
                    + "chr1\t100\t600\tintergenic|chr1:100-600\tintergenic\tGENEA\t0.05\n"
                    + "chr1\t100\t600\tintergenic|chr1:100-600\tintergenic\tGENEB\t0.01\n"
                    + "chr1\t900\t1400\tpromoter|chr1:900-1400\tpromoter\tGENEA\t0.9\n"
                    + "chr2\t10\t510\tgenic|chr2:10-510\tgenic\tGENEC\t0.02\n")
    links = motif_hit_calling.read_enhancer_gene_links(str(path), score_threshold=0.015)
    assert links["gene"].tolist() == ["GENEA", "GENEC"]
    assert links["element_class"].tolist() == ["intergenic", "genic"]
    regions = motif_hit_calling.build_enhancer_regions(links)
    assert regions["sequence_name"].tolist() == ["chr1:100-600|intergenic|chr1:100-600|GENEA",
                                                 "chr2:10-510|genic|chr2:10-510|GENEC"]
    parsed = motif_hit_calling.parse_enhancer_sequence_name(regions["sequence_name"])
    assert parsed["gene"].tolist() == ["GENEA", "GENEC"]
    assert parsed["element_class"].tolist() == ["intergenic", "genic"]
    kept = motif_hit_calling.read_enhancer_gene_links(str(path), drop_promoters=False)
    assert len(kept) == 4


def test_read_abc_headerless(tmp_path):
    row = ["chr1", "100", "600", "genic|chr1:100-600", "genic"] + ["0"] + ["GENEA"] + ["0"] * 13 + ["0.2"] + ["0", "0"] + ["CellA_Ctrl"]
    assert len(row) == len(motif_hit_calling.ABC_HEADERLESS_COLUMNS)
    path = tmp_path / "abc_noheader.txt"
    path.write_text("\t".join(row) + "\n")
    links = motif_hit_calling.read_enhancer_gene_links(str(path))
    assert links.loc[0, "score"] == pytest.approx(0.2)
    assert links.loc[0, "gene"] == "GENEA"


def test_read_igvf_bedpe(tmp_path):
    path = tmp_path / "links.bedpe.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t827272\t827840\tchr1\t827589\t827590\tchr1:827272-827840_LINC01128\t0.95\t.\t.\n"
                     "chr1\t904545\t905045\tchr1\t925739\t925740\tchr1:904545-905045_SAMD11\t0.25\t.\t.\n"
                     "chr1\t925000\t925300\tchr1\t925739\t925740\tchr1:925000-925300_SAMD11\t0.6\t.\t.\n")
    kept = motif_hit_calling.read_enhancer_gene_links(str(path), score_threshold=0.5, drop_promoters=False)
    assert kept["element_class"].tolist() == ["promoter", "promoter"], (
        f"elements within 500 bp of their target TSS are self-promoters, got {kept['element_class'].tolist()}")
    names = motif_hit_calling.build_enhancer_regions(kept)["sequence_name"]
    assert names.tolist()[0] == "chr1:827272-827840|promoter|chr1:827272-827840|LINC01128"
    distal = motif_hit_calling.read_enhancer_gene_links(str(path))
    assert distal[["gene", "element_class", "start"]].values.tolist() == [["SAMD11", ".", 904545]], (
        f"default drops self-promoter elements (and keeps the distal one), got\n{distal}")


def test_read_e2g_tsv_score_alias(tmp_path):
    path = tmp_path / "e2g.tsv"
    path.write_text("chr\tstart\tend\tname\tclass\tTargetGene\tScore\n"
                    "chr1\t1\t500\tmy element\tdistal\tGENEX\t0.7\n")
    links = motif_hit_calling.read_enhancer_gene_links(str(path), score_threshold=0.5)
    names = motif_hit_calling.build_enhancer_regions(links)["sequence_name"]
    assert names.tolist() == ["chr1:1-500|distal|my_element|GENEX"]   # whitespace sanitised


def test_target_gene_from_sequence_name():
    names = pd.Series(["chr1:1-2|genic|chr1:1-2|GENEA", "PROMGENE"])
    assert motif_hit_calling.target_gene_from_sequence_name(names).tolist() == ["GENEA", "PROMGENE"]


def test_write_region_fasta_dedups_clips_and_skips(tmp_path):
    pytest.importorskip("pyfaidx")
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\nACGTacgtAAAACCCCGGGGTTTT\n")
    regions = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chrZ"], "start": [0, 0, 20, 0],
                            "end": [8, 8, 30, 5]})
    out = tmp_path / "regions.fa"
    written = motif_hit_calling.write_region_fasta(regions, str(genome), str(out))
    assert len(written) == 2
    assert out.read_text() == ">chr1:0-8\nACGTacgt\n>chr1:20-30\nTTTT\n"


def test_finemo_hits_intersected_with_regions(tmp_path):
    path = tmp_path / "hits.tsv"
    pd.DataFrame({
        "chr": ["chr1", "chr1", "chr1", "chr2"], "start": [105, 590, 150, 5], "end": [115, 610, 160, 15],
        "start_untrimmed": 0, "end_untrimmed": 0, "motif_name": ["GATA", "CTCF", "AP1", "SP1"],
        "hit_coefficient": [1.5, 2.0, 0.7, 3.0], "hit_correlation": 0.9, "hit_importance": 1.0,
        "strand": ["+", "-", "-", "+"], "peak_name": "p", "peak_id": 0,
    }).to_csv(path, sep="\t", index=False)
    hits = motif_hit_calling.read_finemo_hits(str(path))
    regions = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 140], "end": [600, 400], "strand": ".",
                            "sequence_name": ["chr1:100-600|e|x|GA", "chr1:140-400|e|y|GB"], "gene": ["GA", "GB"]})
    table = motif_hit_calling.call_hits_from_finemo(hits, regions, motif_name_map={"AP1": "FOS_HUMAN.H11MO"})
    table = table.sort_values(["sequence_name", "start"]).reset_index(drop=True)
    assert list(table.columns) == motif_hit_calling.FIMO_COLUMNS
    # GATA in region 1 only; AP1 in both overlapping regions; CTCF crosses the end; SP1 on chr2
    assert table[["motif_id", "sequence_name", "start", "stop"]].values.tolist() == [
        ["GATA", "chr1:100-600|e|x|GA", 6, 15],
        ["FOS_HUMAN.H11MO", "chr1:100-600|e|x|GA", 51, 60],
        ["FOS_HUMAN.H11MO", "chr1:140-400|e|y|GB", 11, 20],
    ]
    assert table["p-value"].isna().all()


MOTIF_FILE = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF GATA1_TEST.H11MO.0.A
letter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0
0.97 0.01 0.01 0.01
0.01 0.01 0.01 0.97
0.01 0.01 0.97 0.01
0.97 0.01 0.01 0.01
0.97 0.01 0.01 0.01
0.01 0.01 0.97 0.01
"""


def available_backends():
    backends = []
    if shutil.which(os.environ.get("FIMO_BINARY", "fimo")):
        backends.append("meme")
    if importlib.util.find_spec("memelite") is not None:
        backends.append("memelite")
    return backends


@pytest.mark.parametrize("backend", available_backends() or [pytest.param("none", marks=pytest.mark.skip("no FIMO backend"))])
def test_scan_regions_with_fimo_finds_planted_motif(tmp_path, backend):
    rng = np.random.default_rng(0)
    background = "".join(rng.choice(list("ACGT"), size=400))
    # plant ATGAAG (+) at 0-based 100 and its reverse complement CTTCAT at 300
    genome_sequence = background[:100] + "ATGAAG" + background[106:300] + "CTTCAT" + background[306:]
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + genome_sequence + "\n")
    motifs = tmp_path / "motifs.meme"
    motifs.write_text(MOTIF_FILE)
    regions = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "start": [50, 50, 250], "end": [200, 200, 350],
                            "strand": ".", "sequence_name": ["GENEA", "GENEB", "GENEC"], "gene": ["GENEA", "GENEB", "GENEC"]})
    out = tmp_path / "hits.tsv"
    n_hits = motif_hit_calling.scan_regions_with_fimo(
        regions, str(genome), str(motifs), str(out), backend=backend, threshold=1e-3,
        fimo_binary=os.environ.get("FIMO_BINARY", "fimo"), n_chunks=2, n_jobs=2)
    hits = pd.read_csv(out, sep="\t")
    assert list(hits.columns) == motif_hit_calling.FIMO_COLUMNS
    assert n_hits == len(hits)
    planted = hits[hits["matched_sequence"].str.upper().isin(["ATGAAG", "CTTCAT"])]
    found = set(zip(planted["sequence_name"], planted["start"], planted["stop"], planted["strand"]))
    # 1-based within region: region 50-200 -> planted at 51..56 (+); region 250-350 -> 51..56 (-)
    assert {("GENEA", 51, 56, "+"), ("GENEB", 51, 56, "+"), ("GENEC", 51, 56, "-")} <= found


# ---------------------------------------------------------------------------
# Fi-NeMo motif naming (ENCODE "sequence motifs report" html + "instances" tsv)
# ---------------------------------------------------------------------------

def report_pattern_section(pattern_id, matches):
    """One TF-MoDISco report pattern section in ENCODE's html layout (logos as data URIs)."""
    rows = "".join(
        f'<tr>\n<td class="num_col">{rank}</td>\n<td><code>{name}</code></td>\n'
        f'<td><img src="data:image/png;base64,AAAA" alt="Match {rank - 1} Logo" class="tomtom-match-logo"></td>\n'
        f'<td class="num_col">{qvalue}</td>\n</tr>\n' for rank, (name, qvalue) in enumerate(matches, start=1))
    table = (f'<h4>Tomtom Matches</h4><table class="tomtom-table"><thead><tr><th class="num_col">Rank</th>'
             f'<th>Match</th><th>Logo</th><th class="num_col">Q-value</th></tr></thead><tbody>{rows}</tbody></table>'
             if matches else "")
    return (f'<div class="pattern-section" id="pattern-x">\n<div class="pattern-title">\n  NAME '
            f'<small>({pattern_id})</small>\n</div>\n<img src="data:image/png;base64,BBBB">'
            f'<table class="stats-table"><tr><td class="num_col">1</td></tr></table>{table}\n</div>\n')


def write_report(path, sections):
    html_text = ('<html><body><div class="summary-container"><span class="pattern-id">ignored</span></div>'
                 + "".join(report_pattern_section(p, m) for p, m in sections) + "</body></html>")
    path.write_text(html_text)


PATTERN_PREFIX = "pos_patterns.ENCSR000AAA_DNase_example-cell-type_ENCSR000AAB_counts_pattern_"


@pytest.fixture
def report_path(tmp_path):
    path = tmp_path / "seq_motifs_report.counts.fold_mean.ENCSR313RDW.html"
    write_report(path, [
        (PATTERN_PREFIX + "0", [("KLF-SP_0", "1.2e-10"), ("KLF_3", "4e-06")]),
        (PATTERN_PREFIX + "1", [("GATA_2", "0.2")]),                          # not significant
        (PATTERN_PREFIX + "2", []),                                            # no TOMTOM table
        ("neg_patterns.ENCSR000EOG_DNase_x_ENCSR313RDW_counts_pattern_0", [("ETV-ELF-NFAT-ELK_0", "1e-4")]),
    ])
    return path


def test_read_finemo_motif_annotation(report_path):
    annotation = motif_hit_calling.read_finemo_motif_annotation(str(report_path))
    assert list(annotation.columns) == motif_hit_calling.MOTIF_ANNOTATION_COLUMNS
    assert annotation[["match_rank", "match"]].fillna("-").values.tolist() == [
        [1, "KLF-SP_0"], [2, "KLF_3"], [1, "GATA_2"], [0, "-"], [1, "ETV-ELF-NFAT-ELK_0"]], annotation
    assert annotation["qvalue"].iloc[0] == pytest.approx(1.2e-10)
    assert annotation["pattern_id"].iloc[0] == PATTERN_PREFIX + "0"


def test_name_finemo_patterns_top_match_regardless_of_qvalue(report_path):
    names = motif_hit_calling.name_finemo_patterns(motif_hit_calling.read_finemo_motif_annotation(str(report_path)))
    assert list(names.columns) == motif_hit_calling.PATTERN_NAME_COLUMNS
    by_pattern = names.set_index("pattern_id")
    assert by_pattern.loc[PATTERN_PREFIX + "0", "tf"] == "KLF-SP_0", "test unit = database cluster"
    assert by_pattern.loc[PATTERN_PREFIX + "0", "motif_family"] == "KLF-SP", "family = cluster without _<n>"
    assert by_pattern.loc[PATTERN_PREFIX + "1", "tf"] == "GATA_2", "default: named whatever the q-value (q=0.2)"
    assert by_pattern.loc[PATTERN_PREFIX + "1", "top_match_qvalue"] == pytest.approx(0.2)
    assert by_pattern.loc[PATTERN_PREFIX + "2", "tf"] == "pos-counts-pattern-2", "no match keeps the pattern label"
    assert by_pattern.loc[PATTERN_PREFIX + "2", "motif_family"] == "pos-counts-pattern-2"
    assert by_pattern.loc[PATTERN_PREFIX + "0", "all_matches"] == "KLF-SP_0(1.2e-10);KLF_3(4e-06)"
    motif_ids = motif_hit_calling.build_finemo_motif_name_map(names)
    assert motif_ids[PATTERN_PREFIX + "0"] == "KLF-SP_0"
    assert motif_ids[PATTERN_PREFIX + "2"] == "pos-counts-pattern-2"


def test_name_finemo_patterns_optional_qvalue_threshold(report_path):
    names = motif_hit_calling.name_finemo_patterns(
        motif_hit_calling.read_finemo_motif_annotation(str(report_path)), qvalue_threshold=0.05)
    by_pattern = names.set_index("pattern_id")
    assert by_pattern.loc[PATTERN_PREFIX + "0", "tf"] == "KLF-SP_0"
    assert by_pattern.loc[PATTERN_PREFIX + "1", "tf"] == "pos-counts-pattern-1", "q >= threshold keeps the label"
    assert not by_pattern.loc[PATTERN_PREFIX + "1", "is_named"]
    assert by_pattern.loc[PATTERN_PREFIX + "1", "top_match_qvalue"] == pytest.approx(0.2), "q kept for unnamed"


def test_add_database_tfs_picks_matching_release(tmp_path):
    """ENCODE ChromBPNet reports use the 2025-09 release (NF2L-NFE_0 exists only there; KLF_3 = KLF15 there,
    KLF10/KLF11 in 2026-05): the release with the most match names is used."""
    names = pd.DataFrame({"pattern_id": ["p0", "p1", "p2", "p3"], "tf": ["NF2L-NFE_0", "KLF_3", "GATA_0", "x"],
                          "motif_family": ["NF2L-NFE", "KLF", "GATA", "x"],
                          "top_match": ["NF2L-NFE_0", "KLF_3", "GATA_0", np.nan], "top_match_qvalue": 0.01,
                          "is_named": [True, True, True, False], "all_matches": ""})
    table = motif_hit_calling.add_database_tfs_to_pattern_names(names)
    assert table["motifcompendium_metadata"].iloc[0].endswith("5b20d47.tsv"), table["motifcompendium_metadata"]
    tf_lists = dict(zip(table["tf"], table["database_tfs"]))
    assert tf_lists["KLF_3"] == "KLF15"
    assert "NFE2L2" in tf_lists["NF2L-NFE_0"].split(",") or "NF2L2" in tf_lists["NF2L-NFE_0"].split(",")
    assert set(tf_lists["GATA_0"].split(",")) >= {"GATA1", "GATA2", "GATA6", "TAL1", "TRPS1"}
    assert tf_lists["x"] == "", "unnamed pattern: no TF list"
    newest = motif_hit_calling.select_motifcompendium_metadata(["KLF-SP_0", "ETV-ELF-NFAT-ELK_0", "GATA_0"])
    assert newest.endswith("2ad26dc.tsv"), "names of the default PFM release -> 2026-05 metadata"
    klf3_new = motif_hit_calling.read_motifcompendium_metadata(newest).set_index("name").loc["KLF_3", "database_tfs"]
    assert klf3_new == "KLF10,KLF11"


def test_local_annotation_table_names_and_hits(tmp_path):
    """export_motif_hits_for_perturbnmf.py outputs: motif_annotation.tsv + '#chrom'-headed hits with motif_id."""
    annotation = tmp_path / "motif_annotation.tsv"
    pd.DataFrame({"motif_id": ["cluster_0", "cluster_1", "cluster_2", "cluster_3"], "cluster_id": [0, 1, 2, 3],
                  "motif_label": ["KLF-SP_0", "cluster_1", "GATA2", "CTCF_0"],
                  "database_motif": ["KLF-SP_0", np.nan, "GATA_1", "CTCF_0"],
                  "database_match_score": [0.93, np.nan, 0.85, 0.9],
                  "candidate_tfs": ["KLF2,KLF4,SP1", np.nan, "GATA2", "CTCF"],
                  "posneg": ["pos", "pos", "pos", "neg"]}).to_csv(annotation, sep="\t", index=False)
    names = motif_hit_calling.name_finemo_patterns_from_annotation_table(str(annotation), ["pos_patterns"])
    assert names["pattern_id"].tolist() == ["cluster_0", "cluster_1", "cluster_2"], "neg motif dropped"
    assert names["tf"].tolist() == ["KLF-SP_0", "cluster-1", "GATA_1"]
    assert names["motif_family"].tolist() == ["KLF-SP", "cluster-1", "GATA"]
    assert names["database_tfs"].tolist() == ["KLF2,KLF4,SP1", "", "GATA2"]
    assert names["is_named"].tolist() == [True, False, True]

    hits_path = tmp_path / "motif_hits_crispri.tsv"
    with open(hits_path, "w") as handle:
        handle.write("#chrom\tstart\tend\tstrand\tmotif_id\tmotif_label\tposneg\tscore\thit_similarity\n")
        handle.write("chr1\t110\t120\t+\tcluster_0\tKLF-SP_0\tpos\t0.5\t0.9\n")
        handle.write("chr1\t130\t140\t-\tcluster_2\tGATA2\tpos\t0.7\t0.9\n")
    hits = motif_hit_calling.read_finemo_hits(str(hits_path))
    assert hits["motif_name"].tolist() == ["cluster_0", "cluster_2"] and hits["score"].tolist() == [0.5, 0.7]
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200], "strand": "+",
                            "sequence_name": ["GENE1"], "gene": ["GENE1"]})
    table = motif_hit_calling.call_hits_from_finemo(hits, regions, motif_hit_calling.build_finemo_motif_name_map(names))
    assert table["motif_id"].tolist() == ["KLF-SP_0", "GATA_1"]


def test_collapse_motifcompendium_name():
    assert [motif_hit_calling.collapse_motifcompendium_name(n) for n in ["KLF-SP_0", "ZNF_141", "CTCF", "A_B_12"]] == [
        "KLF-SP", "ZNF", "CTCF", "A_B"]


def write_instances(path, rows):
    columns = ["chr", "start", "end", "start_untrimmed", "end_untrimmed", "motif_name", "hit_coefficient",
               "hit_coefficient_global", "hit_similarity", "hit_correlation", "hit_importance",
               "hit_importance_sq", "strand", "peak_name", "peak_id"]
    table = pd.DataFrame([dict(zip(["chr", "start", "end", "motif_name", "strand", "peak_id"], r)) for r in rows])
    for column in columns:
        if column not in table:
            table[column] = 1.0 if column.startswith("hit") else ""
    table["start_untrimmed"], table["end_untrimmed"] = table["start"] - 5, table["end"] + 5
    table[columns].to_csv(path, sep="\t", index=False, compression="infer")


def test_read_finemo_hits_dedup_prefix_and_space_normalization(tmp_path, report_path):
    path = tmp_path / "seq_motifs_instances.counts.lambda_0p7.ENCSR313RDW.tsv.gz"
    spaced = PATTERN_PREFIX.replace("example-cell-type", "example cell type")
    write_instances(path, [
        ("chr1", 110, 120, spaced + "0", "+", 1),
        ("chr1", 110, 120, spaced + "0", "+", 2),          # same hit, overlapping peak -> dropped
        ("chr1", 130, 140, spaced + "1", "-", 1),
        ("chr1", 150, 160, "neg_patterns.ENCSR000EOG_DNase_x_ENCSR313RDW_counts_pattern_0", "+", 1),
    ])
    hits = motif_hit_calling.read_finemo_hits(str(path), pattern_prefixes=["pos_patterns"])
    assert len(hits) == 2, f"duplicate peak row and neg pattern dropped, got\n{hits}"
    names = motif_hit_calling.name_finemo_patterns(motif_hit_calling.read_finemo_motif_annotation(str(report_path)))
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200], "strand": "+",
                            "sequence_name": ["GENE1"], "gene": ["GENE1"]})
    table = motif_hit_calling.call_hits_from_finemo(hits, regions, motif_hit_calling.build_finemo_motif_name_map(names))
    assert table["motif_id"].tolist() == ["KLF-SP_0", "GATA_2"], (
        "report ids use '-' for spaces; instance ids with spaces must still map")
    assert table["motif_alt_id"].iloc[0] == spaced + "0", "original pattern id kept in motif_alt_id"


def test_find_finemo_files(tmp_path):
    for head in ("counts", "profile"):
        for lam in ("0p6", "0p7"):
            folder = tmp_path / "inst" / head / f"seq_motifs_instances.{head}.lambda_{lam}"
            folder.mkdir(parents=True)
            (folder / f"seq_motifs_instances.{head}.lambda_{lam}.ENCSR313RDW.tsv").write_text("chr\n")
            (folder / f"seq_motifs_instances.{head}.lambda_{lam}.ENCSR313RDW.bed").write_text("")
        (tmp_path / "rep" / head).mkdir(parents=True)
        (tmp_path / "rep" / head / f"seq_motifs_report.{head}.fold_mean.ENCSR313RDW.html").write_text("")
    found = motif_hit_calling.find_finemo_instances_file(str(tmp_path / "inst"), "counts", 0.7)
    assert found.endswith("counts/seq_motifs_instances.counts.lambda_0p7/seq_motifs_instances.counts.lambda_0p7.ENCSR313RDW.tsv")
    assert motif_hit_calling.find_finemo_report_file(str(tmp_path / "rep"), "profile").endswith(
        "seq_motifs_report.profile.fold_mean.ENCSR313RDW.html")
    with pytest.raises(FileNotFoundError):
        motif_hit_calling.find_finemo_instances_file(str(tmp_path / "inst"), "counts", 0.9)


def test_find_finemo_files_chrombpnet_layout(tmp_path):
    """ChromBPNet (DNase) tars: {head}/lambda_0.7/finemo.motif_hits.{head}.0.7.<ENCSR>.tsv.gz, tfmodisco.report.*.html."""
    for lam in ("0.7", "0.8"):
        folder = tmp_path / "inst" / "counts" / f"lambda_{lam}"
        folder.mkdir(parents=True)
        (folder / f"finemo.motif_hits.counts.{lam}.ENCSR000EOG.tsv.gz").write_bytes(b"")
        (folder / f"finemo.motif_hits.counts.{lam}.ENCSR000EOG.bed.gz").write_bytes(b"")
    (tmp_path / "rep" / "counts").mkdir(parents=True)
    (tmp_path / "rep" / "counts" / "tfmodisco.report.counts.ENCSR000EOG.html").write_text("")
    found = motif_hit_calling.find_finemo_instances_file(str(tmp_path / "inst"), "counts", 0.7)
    assert found.endswith("counts/lambda_0.7/finemo.motif_hits.counts.0.7.ENCSR000EOG.tsv.gz"), found
    assert motif_hit_calling.find_finemo_report_file(str(tmp_path / "rep"), "counts").endswith(
        "tfmodisco.report.counts.ENCSR000EOG.html")


# ---------------------------------------------------------------------------
# Review fixes (regression tests)
# ---------------------------------------------------------------------------

def test_unmapped_finemo_patterns_get_labels_not_pos(tmp_path):
    hits = pd.DataFrame({"chrom": "chr1", "start": [110, 130, 150], "end": [120, 140, 160], "strand": "+",
                         "motif_name": [PATTERN_PREFIX + "0", PATTERN_PREFIX + "3", "GATA"], "score": 1.0})
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200], "strand": "+",
                            "sequence_name": ["GENE1"], "gene": ["GENE1"]})
    table = motif_hit_calling.call_hits_from_finemo(hits, regions, {PATTERN_PREFIX + "0": "KLF-SP_pos-counts-pattern-0"})
    assert table["motif_id"].tolist() == ["KLF-SP_pos-counts-pattern-0", "pos-counts-pattern-3", "GATA"]
    tfs = table["motif_id"].str.split("_", n=1).str[0].tolist()     # motif_enrichment.collapse_motif_to_tf
    assert tfs == ["KLF-SP", "pos-counts-pattern-3", "GATA"], "an unmapped pattern must not collapse to 'pos'"
    unmapped_only = motif_hit_calling.call_hits_from_finemo(hits, regions, None)
    assert unmapped_only["motif_id"].tolist()[:2] == ["pos-counts-pattern-0", "pos-counts-pattern-3"]


def test_bed_minus_strand_tss_is_last_base_in_strand_aware_mode(tmp_path):
    bed = tmp_path / "bounds.bed"
    bed.write_text("chr1\t1000\t2000\tPLUS\t0\t+\nchr1\t3000\t4000\tMINUS\t0\t-\n")
    strand_aware = motif_hit_calling.read_bed_gene_tss(str(bed)).set_index("gene")
    assert strand_aware.loc["PLUS", "tss"] == 1000 and strand_aware.loc["MINUS", "tss"] == 3999
    gtf_equivalent = motif_hit_calling.build_promoter_regions(strand_aware.reset_index()).set_index("gene")
    assert tuple(gtf_equivalent.loc["MINUS", ["start", "end"]]) == (3949, 4250), "same window as the GTF path"
    paper = motif_hit_calling.read_bed_gene_tss(str(bed), window_mode="schnitzler2024").set_index("gene")
    assert paper.loc["MINUS", "tss"] == 4000, "schnitzler2024 keeps the paper's BED end"


def test_score_threshold_without_score_column_raises(tmp_path):
    path = tmp_path / "links.tsv"
    path.write_text("chr\tstart\tend\tTargetGene\nchr1\t100\t600\tGENEA\n")
    assert len(motif_hit_calling.read_enhancer_gene_links(str(path))) == 1
    with pytest.raises(ValueError, match="score"):
        motif_hit_calling.read_enhancer_gene_links(str(path), score_threshold=0.1)
    with pytest.raises(ValueError, match="not found"):
        motif_hit_calling.read_enhancer_gene_links(str(path), score_column="Nope")


def write_fai(fasta_path, lengths):
    fasta_path.write_text("")
    with open(str(fasta_path) + ".fai", "w") as handle:
        for chrom, length in lengths.items():
            handle.write(f"{chrom}\t{length}\t0\t60\t61\n")


def test_match_chromosome_names_and_missing_fraction_gate(tmp_path):
    assert motif_hit_calling.match_chromosome_names(["chr1", "chrM", "chrUn_x"], ["1", "MT"]) == {"chr1": "1", "chrM": "MT"}
    assert motif_hit_calling.match_chromosome_names(["1", "X"], ["chr1", "chrX"]) == {"1": "chr1", "X": "chrX"}
    genome = tmp_path / "genome.fa"
    write_fai(genome, {"1": 1000, "2": 1000})
    regions = pd.DataFrame({"chrom": ["chr1"] * 19 + ["chrZ"], "start": range(0, 200, 10), "end": range(5, 205, 10)})
    assert motif_hit_calling.check_region_chromosomes(regions, str(genome)) == {"chr1": "1"}    # 5% dropped: ok
    regions.loc[18, "chrom"] = "chrZ"
    with pytest.raises(ValueError, match="absent"):
        motif_hit_calling.check_region_chromosomes(regions, str(genome))                     # 10% dropped


def test_write_region_fasta_resolves_chr_prefix(tmp_path):
    pytest.importorskip("pyfaidx")
    genome = tmp_path / "genome.fa"
    genome.write_text(">1\nACGTacgtAAAACCCCGGGGTTTT\n")
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [8]})
    names = motif_hit_calling.check_region_chromosomes(regions, str(genome))
    out = tmp_path / "regions.fa"
    motif_hit_calling.write_region_fasta(regions, str(genome), str(out), names)
    assert out.read_text() == ">chr1:0-8\nACGTacgt\n", "record id keeps the region chromosome name"


def test_scan_regions_with_fimo_raises_on_zero_hits(tmp_path, monkeypatch):
    pytest.importorskip("pyfaidx")
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "A" * 400 + "\n")
    monkeypatch.setattr(motif_hit_calling, "resolve_fimo_backend", lambda backend, binary="fimo": "meme")
    monkeypatch.setattr(motif_hit_calling, "run_meme_fimo",
                        lambda *args, **kwargs: pd.DataFrame(columns=motif_hit_calling.FIMO_COLUMNS))
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [300], "strand": ".",
                            "sequence_name": ["G"], "gene": ["G"]})
    with pytest.raises(RuntimeError, match="0 hits"):
        motif_hit_calling.scan_regions_with_fimo(regions, str(genome), "motifs.meme", str(tmp_path / "hits.tsv"))


def test_genome_build_helpers(tmp_path):
    assert motif_hit_calling.normalize_genome_build("GRCh38") == "hg38"
    assert motif_hit_calling.normalize_genome_build("GRCh37") == "hg19"
    assert motif_hit_calling.normalize_genome_build("") is None
    assert motif_hit_calling.infer_genome_build_from_path("/x/ABC_hg19_Predictions.txt.gz") == "hg19"
    assert motif_hit_calling.infer_genome_build_from_path("/x/IGVFFI0000AAA.bedpe.gz") is None
    hg19 = tmp_path / "genome.fa"
    write_fai(hg19, {"chr1": 249250621})
    assert motif_hit_calling.infer_genome_build_from_fasta(str(hg19)) == "hg19"
    motif_hit_calling.check_genome_builds("hg19", {"fasta": "hg19", "links": "GRCh37", "gtf": None})
    with pytest.raises(ValueError, match="mismatch"):
        motif_hit_calling.check_genome_builds("hg38", {"fasta": "hg19"})


def test_describe_fimo_backend_records_binary_and_version(tmp_path):
    fake_fimo = tmp_path / "fimo"
    fake_fimo.write_text("#!/bin/sh\necho 5.3.3\n")
    fake_fimo.chmod(0o755)
    described = motif_hit_calling.describe_fimo_backend("auto", str(fake_fimo))
    assert described == {"fimo_backend": "meme", "fimo_binary_path": os.path.realpath(str(fake_fimo)),
                         "fimo_version": "5.3.3"}
    assert motif_hit_calling.describe_fimo_backend("auto", str(tmp_path / "missing"))["fimo_backend"] == "memelite"

