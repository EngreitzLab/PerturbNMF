"""Unit tests for Stage2_Evaluation/A_Metrics/Slurm_Version/run_motif_enrichment.py (orchestrator).

Synthetic PerturbNMF run (3 programs x 40 genes), precomputed FIMO tables, a tiny GTF + enhancer links,
and ENCODE-style Fi-NeMo instances / report files. No FIMO binary needed.

Test strategy
  fimo + precomputed hits:  output files / columns; results equal direct motif_enrichment calls
                            (threshold per element type, promoter-class enhancer hits dropped);
                            candidate TFs carry knockdown evidence from the run's perturbation table
  both sources:             motif_source column, BH per source, finemo TFs = TOMTOM families,
                            family -> gene expansion in candidates; pattern-name table written;
                            hit tables cached (second run does not rebuild regions)
  correlation:              enrichment = r, significant = fdr < 0.05 & r > 0
  inputs:                   manifest rank-1 row supplies enhancer links; missing finemo inputs -> exit
"""

import gzip
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
SCRIPT_PATH = os.path.join(REPO, "src", "Stage2_Evaluation", "A_Metrics", "Slurm_Version", "run_motif_enrichment.py")
spec = importlib.util.spec_from_file_location("run_motif_enrichment", SCRIPT_PATH)
rme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rme)
motif_enrichment = rme.motif_enrichment

FIMO_HEADER = "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence\n"
GENES = [f"G{i}" for i in range(36)] + ["KLF2", "KLF4", "GATA2", "SP1"]
PATTERN = "pos_patterns.ENCSR000EOG_DNase_example-cell_ENCSR313RDW_counts_pattern_"


def write_fimo(path, rows):
    with open(path, "w") as handle:
        handle.write(FIMO_HEADER)
        for motif_id, sequence_name, pvalue in rows:
            handle.write(f"{motif_id}\t\t{sequence_name}\t1\t10\t+\t10.0\t{pvalue}\t\tACGT\n")


@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(1)
    out_dir, run = tmp_path / "out", "run1"
    inference = out_dir / run / "Inference"
    evaluation = out_dir / run / "Evaluation" / "3_0_2"
    inference.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    scores = pd.DataFrame(rng.gamma(1.0, 1.0, size=(3, len(GENES))), index=[1, 2, 3], columns=GENES)
    scores.loc[1, ["G0", "G1", "G2", "G3", "G4", "KLF4"]] += 10          # program 1 top genes
    scores.to_csv(inference / "Inference.gene_spectra_score.k_3.dt_0_2.txt", sep="\t")
    pd.DataFrame({"target_name": ["KLF4", "GATA2"], "program_name": [1, 2], "log2FC": [-0.5, 0.2],
                  "adj_pval": [0.01, 0.5]}).to_csv(
        evaluation / "3_perturbation_association_results_EC.txt", sep="\t", index=False)

    # promoter FIMO: KLF4 motif enriched in program-1 top genes; one hit above the promoter threshold
    promoter_rows = [("KLF4_HUMAN.H11MO.0.A", g, "1e-5") for g in ["G0", "G1", "G2", "G3", "G4"] for _ in range(3)]
    promoter_rows += [("KLF4_HUMAN.H11MO.0.A", g, "1e-5") for g in GENES[5:30]]
    promoter_rows += [("KLF4_HUMAN.H11MO.0.A", g, "1e-5") for g in GENES[5:30:2]]      # background varies
    promoter_rows += [("GATA2_HUMAN.H11MO.0.A", g, "5e-5") for g in GENES[3:25]]
    promoter_rows += [("GATA2_HUMAN.H11MO.0.A", "G30", "2e-4")]
    write_fimo(tmp_path / "promoter_hits.tsv", promoter_rows)
    enhancer_rows = [("GATA2_HUMAN.H11MO.0.A", f"chr1:{i}-{i + 100}|genic|e{i}|{g}", "1e-7")
                     for i, g in enumerate(GENES[:30])]
    enhancer_rows += [("GATA2_HUMAN.H11MO.0.A", "chr1:1-2|promoter|p1|G0", "1e-7")] * 5
    enhancer_rows += [("KLF4_HUMAN.H11MO.0.A", f"chr1:{i}-{i + 50}|genic|f{i}|{g}", "1e-5")
                      for i, g in enumerate(GENES[:30])]                            # above 1e-6 -> dropped
    enhancer_rows += [("SP1_HUMAN.H11MO.0.A", f"chr1:{i}-{i + 50}|genic|f{i}|{g}", "1e-8")
                      for i, g in enumerate(["G0", "G1", "G2", "G3", "G4", "G9", "G12"])]
    write_fimo(tmp_path / "enhancer_hits.tsv", enhancer_rows)

    # GTF: one + gene per symbol, TSS at 1000 * (index + 1) on chr1
    with gzip.open(tmp_path / "genes.gtf.gz", "wt") as handle:
        for index, gene in enumerate(GENES):
            start = 1000 * (index + 1) + 1
            handle.write(f'chr1\tX\tgene\t{start}\t{start + 500}\t.\t+\t.\tgene_id "E{index}"; '
                         f'gene_type "protein_coding"; gene_name "{gene}";\n')
    # enhancer links (bedpe): one distal element per gene, 600 bp downstream of its TSS
    with gzip.open(tmp_path / "links.bedpe.gz", "wt") as handle:
        for index, gene in enumerate(GENES):
            tss, start = 1000 * (index + 1), 1000 * (index + 1) + 600
            handle.write(f"chr1\t{start}\t{start + 200}\tchr1\t{tss}\t{tss + 1}\tchr1:{start}-{start + 200}_{gene}\t0.5\n")

    # Fi-NeMo: KLF-SP pattern hits in promoters of program-1 top genes; GATA pattern everywhere
    instance_rows = []
    for index, gene in enumerate(GENES):
        tss = 1000 * (index + 1)
        instance_rows.append(("chr1", tss - 100, tss - 90, PATTERN + "1", "+", 0))                # GATA, promoter
        instance_rows.append(("chr1", tss - 100, tss - 90, PATTERN + "1", "+", 1))                # duplicate peak
        instance_rows.append(("chr1", tss + 650, tss + 660, PATTERN + "1", "-", 0))                # GATA, enhancer
        if gene in ("G0", "G1", "G2", "G3", "G4", "KLF4") or index % 7 == 0:
            n = 4 if gene in ("G0", "G1", "G2", "G3", "G4", "KLF4") else 1
            for j in range(n):
                instance_rows.append(("chr1", tss - 200 + 12 * j, tss - 190 + 12 * j, PATTERN + "0", "+", 0))
    instances = pd.DataFrame(instance_rows, columns=["chr", "start", "end", "motif_name", "strand", "peak_id"])
    for column in ["start_untrimmed", "end_untrimmed", "hit_coefficient_global", "hit_similarity",
                   "hit_correlation", "hit_importance", "hit_importance_sq", "peak_name"]:
        instances[column] = 0
    instances["hit_coefficient"] = 1.0
    folder = tmp_path / "instances" / "counts" / "seq_motifs_instances.counts.lambda_0p7"
    folder.mkdir(parents=True)
    instances.to_csv(folder / "seq_motifs_instances.counts.lambda_0p7.ENCSR313RDW.tsv", sep="\t", index=False)
    report = tmp_path / "report.html"
    sections = [(PATTERN + "0", "KLF-SP_0", "1e-9"), (PATTERN + "1", "GATA_2", "1e-5")]
    report.write_text("<html><body>" + "".join(
        f'<div class="pattern-section"><div class="pattern-title">x <small>({p})</small></div>'
        f'<table class="tomtom-table"><tbody><tr><td class="num_col">1</td><td><code>{m}</code></td>'
        f'<td><img src="data:image/png;base64,AA"></td><td class="num_col">{q}</td></tr></tbody></table></div>'
        for p, m, q in sections) + "</body></html>")
    return tmp_path, out_dir, run


def base_argv(tmp_path, out_dir, run):
    return ["--out_dir", str(out_dir), "--run_name", run, "--K", "3", "--sel_threshs", "0.2", "--n_top", "6",
            "--gene_annotation", str(tmp_path / "genes.gtf.gz"), "--motif_min_universe_genes", "10",
            "--motif_db", "hocomoco_v11"]


def test_fimo_with_precomputed_hits_matches_direct_calls(run_dir):
    tmp_path, out_dir, run = run_dir
    argv = base_argv(tmp_path, out_dir, run) + ["--promoter_hits", str(tmp_path / "promoter_hits.tsv"),
                                                "--enhancer_hits", str(tmp_path / "enhancer_hits.tsv"),
                                                "--motif_file", str(tmp_path / "absent.meme")]
    rme.main(argv)
    folder = out_dir / run / "Evaluation" / "3_0_2"
    results = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")
    expected_columns = list(motif_enrichment.RESULT_COLUMNS)
    expected_columns.insert(expected_columns.index("tf") + 1, "motif_family")
    assert list(results.columns) == expected_columns + ["significant"]
    assert results.set_index("tf")["motif_family"].to_dict()["KLF4"] == "Three-zinc finger Krüppel-related factors"
    assert (folder / "3_candidate_tfs.txt").exists() and (folder / "3_motif_enrichment_config.yml").exists()
    assert json.loads((folder / "3_motif_logos.json").read_text())["logos"] == {}, "no motif file -> no logos"

    scores = pd.read_csv(out_dir / run / "Inference" / "Inference.gene_spectra_score.k_3.dt_0_2.txt",
                         sep="\t", index_col=0)
    program_genes = motif_enrichment.select_top_program_genes(scores, 6)
    promoter_counts = motif_enrichment.count_hits_per_gene_tf(
        motif_enrichment.read_fimo_hits(str(tmp_path / "promoter_hits.tsv"), 1e-4), genes=scores.columns)
    expected = motif_enrichment.test_motif_enrichment_ttest(promoter_counts, program_genes, "promoter")
    got = results[results["element_type"] == "promoter"].reset_index(drop=True)
    np.testing.assert_allclose(got["pvalue"], expected["pvalue"], rtol=1e-12)
    np.testing.assert_allclose(got["enrichment"], expected["enrichment"], rtol=1e-12)
    enhancer = results[results["element_type"] == "enhancer"]
    assert set(enhancer["tf"]) == {"GATA2", "SP1"}, "KLF4 enhancer hits are above 1e-6 and must be dropped"
    assert enhancer["n_program_genes_tested"].max() <= 6

    candidates = pd.read_csv(folder / "3_candidate_tfs.txt", sep="\t")
    klf4 = candidates[(candidates["program"].astype(str) == "1") & (candidates["tf"] == "KLF4")]
    assert len(klf4) == 1 and klf4["evidence_tier"].iloc[0] == "motif+regulator", candidates


def test_both_sources_with_finemo_and_cache(run_dir, monkeypatch):
    tmp_path, out_dir, run = run_dir
    argv = base_argv(tmp_path, out_dir, run) + [
        "--promoter_hits", str(tmp_path / "promoter_hits.tsv"), "--enhancer_hits", str(tmp_path / "enhancer_hits.tsv"),
        "--enhancer_links", str(tmp_path / "links.bedpe.gz"), "--motif_source", "both",
        "--finemo_instances", str(tmp_path / "instances"), "--finemo_report", str(tmp_path / "report.html")]
    rme.main(argv)
    folder = out_dir / run / "Evaluation" / "3_0_2"
    results = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")
    assert set(results["motif_source"]) == {"fimo", "finemo"}
    finemo = results[results["motif_source"] == "finemo"]
    assert set(finemo["tf"]) == {"KLF-SP_0", "GATA_2"}, f"finemo tests = TOMTOM clusters, got {set(finemo['tf'])}"
    assert dict(zip(finemo["tf"], finemo["motif_family"])) == {"KLF-SP_0": "KLF-SP", "GATA_2": "GATA"}
    assert dict(zip(finemo["tf"], finemo["motif_match_qvalue"])) == {"KLF-SP_0": 1e-9, "GATA_2": 1e-5}
    assert results.loc[results["motif_source"] == "fimo", "motif_match_qvalue"].isna().all()
    for (_, _), group in results.groupby(["motif_source", "element_type"]):
        np.testing.assert_allclose(group["fdr"], motif_enrichment.adjust_pvalues_bh(group["pvalue"].to_numpy()))
    klf = finemo[(finemo["program"] == 1) & (finemo["tf"] == "KLF-SP_0") & (finemo["element_type"] == "promoter")]
    assert klf["significant"].iloc[0], f"KLF-SP pattern enriched in program 1 promoters: {klf}"
    names = pd.read_csv(folder / "3_finemo_pattern_names.tsv", sep="\t")
    assert names["tf"].tolist() == ["KLF-SP_0", "GATA_2"]
    assert names["motif_family"].tolist() == ["KLF-SP", "GATA"]
    assert "KLF4" in names["database_tfs"].iloc[0].split(",") and names["database_tfs"].iloc[1] == "GATA1"

    candidates = pd.read_csv(folder / "3_candidate_tfs.txt", sep="\t")
    finemo_klf = candidates[(candidates["motif_source"] == "finemo") & (candidates["tf"] == "KLF-SP_0")
                            & (candidates["program"].astype(str) == "1")]
    assert {"KLF2", "KLF4", "SP1"} <= set(finemo_klf["tf_gene_symbol"]), finemo_klf
    assert finemo_klf.set_index("tf_gene_symbol").loc["KLF4", "evidence_tier"] == "motif+regulator"
    assert set(finemo_klf["tf_gene_symbol_source"]) == {"motifcompendium"}
    assert (finemo_klf["motif_family"] == "KLF-SP").all()
    assert list(candidates.columns[:4]) == ["program", "element_type", "tf", "motif_family"]

    cache = out_dir / run / "Evaluation" / "motif_hits"
    assert sorted(p.name.split("_")[0] + "_" + p.name.split("_")[1] for p in cache.iterdir()
                  if p.name.endswith(tuple("0123456789abcdef")) and "_finemo_" in p.name) == [
        "enhancer_finemo", "promoter_finemo"]
    monkeypatch.setattr(rme, "build_regions", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    rme.main(argv)                          # cached hit tables -> no region building
    again = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")
    pd.testing.assert_frame_equal(results, again)


def test_correlation_method(run_dir):
    tmp_path, out_dir, run = run_dir
    argv = base_argv(tmp_path, out_dir, run) + ["--promoter_hits", str(tmp_path / "promoter_hits.tsv"),
                                                "--motif_element_types", "promoter", "--motif_method", "correlation"]
    rme.main(argv)
    results = pd.read_csv(out_dir / run / "Evaluation" / "3_0_2" / "3_motif_enrichment.txt", sep="\t")
    assert results["enrichment"].between(-1, 1).all(), "enrichment holds the correlation coefficient"
    assert (results["significant"] == ((results["fdr"] < 0.05) & (results["enrichment"] > 0))).all()
    assert (results["n_program_genes_tested"] == len(GENES)).all(), "universe = all expressed genes"


def test_manifest_supplies_enhancer_links_and_finemo_needs_inputs(run_dir):
    tmp_path, out_dir, run = run_dir
    manifest = tmp_path / "regulatory_resources_manifest.tsv"
    pd.DataFrame([{"rank": 1, "portal": "IGVF", "resource_type": "e2g_links", "download_url": "https://x/y.bedpe.gz",
                   "local_path": str(tmp_path / "links.bedpe.gz")},
                  {"rank": 2, "portal": "IGVF", "resource_type": "e2g_links", "download_url": "",
                   "local_path": "/nonexistent"}]).to_csv(manifest, sep="\t", index=False)
    args = rme.parse_arguments(base_argv(tmp_path, out_dir, run) + ["--regulatory_resources_manifest", str(manifest)])
    resources = rme.resolve_resources(args, str(tmp_path / "cache"))
    assert resources["enhancer_links"] == str(tmp_path / "links.bedpe.gz")
    args = rme.parse_arguments(base_argv(tmp_path, out_dir, run) + ["--motif_source", "finemo"])
    with pytest.raises(SystemExit):
        rme.resolve_resources(args, str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Review fixes (regression tests)
# ---------------------------------------------------------------------------

def test_manifest_finemo_instances_and_report_come_from_one_dataset(run_dir):
    tmp_path, out_dir, run = run_dir
    other_report = tmp_path / "other_report.html"
    other_report.write_text("<html></html>")
    manifest = tmp_path / "regulatory_resources_manifest.tsv"
    rows = [  # rank-1 instances (ENCSR_A) has no report; rank-1 report belongs to ENCSR_C
        ("1", "motif_instances", "ENCSR_A", "/nonexistent/a.tsv"),
        ("2", "motif_instances", "ENCSR_B", str(tmp_path / "instances")),
        ("1", "motif_report", "ENCSR_C", str(other_report)),
        ("2", "motif_report", "ENCSR_B", str(tmp_path / "report.html")),
    ]
    pd.DataFrame([{"rank": r, "portal": "ENCODE", "resource_type": t, "dataset_accession": d, "assembly": "GRCh38",
                   "download_url": "", "local_path": p} for r, t, d, p in rows]).to_csv(manifest, sep="\t", index=False)
    args = rme.parse_arguments(base_argv(tmp_path, out_dir, run) + [
        "--motif_source", "finemo", "--regulatory_resources_manifest", str(manifest)])
    resources = rme.resolve_resources(args, str(tmp_path / "cache"))
    assert resources["finemo_report"] == str(tmp_path / "report.html"), "report must be ENCSR_B's, like the instances"
    assert resources["finemo_instances"].startswith(str(tmp_path / "instances"))
    assert resources["assemblies"] == {"finemo_instances": "GRCh38", "finemo_report": "GRCh38"}


def test_manifest_assembly_mismatch_with_genome_build_raises(run_dir):
    tmp_path, out_dir, run = run_dir
    manifest = tmp_path / "regulatory_resources_manifest.tsv"
    pd.DataFrame([{"rank": 1, "portal": "ENCODE", "resource_type": "e2g_links", "dataset_accession": "ENCSR_X",
                   "assembly": "GRCh37", "download_url": "", "local_path": str(tmp_path / "links.bedpe.gz")}]
                 ).to_csv(manifest, sep="\t", index=False)
    argv = base_argv(tmp_path, out_dir, run) + ["--promoter_hits", str(tmp_path / "promoter_hits.tsv"),
                                                "--enhancer_hits", str(tmp_path / "enhancer_hits.tsv"),
                                                "--regulatory_resources_manifest", str(manifest)]
    with pytest.raises(ValueError, match="genome build mismatch"):
        rme.main(argv)
    rme.main(argv + ["--genome_build", "GRCh37", "--genome_fasta", str(tmp_path / "none.fa")])   # declared hg19: ok


def test_build_directory_atomically_concurrent_jobs(tmp_path):
    import threading
    import time
    target = str(tmp_path / "cache" / "table")
    os.makedirs(os.path.dirname(target))
    builders = []

    def build_into(directory):
        builders.append(directory)
        with open(os.path.join(directory, "motif_hits.tsv"), "w") as handle:
            for line in range(200):
                handle.write(f"{directory}\t{line}\n")
                if line % 50 == 0:
                    time.sleep(0.01)

    results = []
    threads = [threading.Thread(target=lambda: results.append(rme.build_directory_atomically(target, build_into)))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [target] * 4
    lines = open(os.path.join(target, "motif_hits.tsv")).read().splitlines()
    assert len(lines) == 200 and len({line.split("\t")[0] for line in lines}) == 1, "one complete, unmixed build"
    assert os.path.exists(os.path.join(target, "DONE"))
    assert os.listdir(os.path.dirname(target)) == ["table"], "losing builds' temporary directories removed"

    def build_while_another_job_wins(directory):      # another job renames its table in first
        os.makedirs(target + "2")
        open(os.path.join(target + "2", "DONE"), "w").close()
        open(os.path.join(target + "2", "winner"), "w").close()
    assert rme.build_directory_atomically(target + "2", build_while_another_job_wins) == target + "2"
    assert os.path.exists(os.path.join(target + "2", "winner"))
    assert sorted(os.listdir(os.path.dirname(target))) == ["table", "table2"]


def test_cache_key_stores_resolved_fimo_backend_binary_and_version(run_dir, tmp_path):
    run_tmp, out_dir, run = run_dir
    fake_fimo = tmp_path / "bin" / "fimo"
    fake_fimo.parent.mkdir()
    fake_fimo.write_text("#!/bin/sh\necho 5.3.3\n")
    fake_fimo.chmod(0o755)
    (tmp_path / "genome.fa").write_text(">chr1\nACGT\n")
    (tmp_path / "motifs.meme").write_text("MEME version 4\n")
    args = rme.parse_arguments(base_argv(run_tmp, out_dir, run) + [
        "--fimo_binary", str(fake_fimo), "--genome_fasta", str(tmp_path / "genome.fa"),
        "--motif_file", str(tmp_path / "motifs.meme")])
    parameters = rme.hit_table_parameters("promoter", "fimo", args, {"enhancer_links": None})
    assert parameters["fimo_backend"] == "meme", "auto must be resolved before it enters the cache key"
    assert parameters["fimo_binary_path"] == os.path.realpath(str(fake_fimo))
    assert parameters["fimo_version"] == "5.3.3" and parameters["genome_build"] == "hg38"


def test_ensembl_gene_ids_are_mapped_to_symbols(run_dir):
    tmp_path, out_dir, run = run_dir
    argv = base_argv(tmp_path, out_dir, run) + ["--promoter_hits", str(tmp_path / "promoter_hits.tsv"),
                                                "--motif_element_types", "promoter"]
    rme.main(argv)
    folder = out_dir / run / "Evaluation" / "3_0_2"
    by_symbol = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")

    gene_ids = {gene: f"ENSG{index:011d}.{index % 3 + 1}" for index, gene in enumerate(GENES)}
    with gzip.open(tmp_path / "genes_ids.gtf.gz", "wt") as handle:
        for index, gene in enumerate(GENES):
            start = 1000 * (index + 1) + 1
            handle.write(f'chr1\tX\tgene\t{start}\t{start + 500}\t.\t+\t.\tgene_id "{gene_ids[gene].split(".")[0]}"; '
                         f'gene_type "protein_coding"; gene_name "{gene}";\n')
    scores = pd.read_csv(out_dir / run / "Inference" / "Inference.gene_spectra_score.k_3.dt_0_2.txt", sep="\t", index_col=0)
    ids_path = tmp_path / "scores_by_id.txt"
    scores.rename(columns=gene_ids).to_csv(ids_path, sep="\t")
    argv_ids = [a if a != str(tmp_path / "genes.gtf.gz") else str(tmp_path / "genes_ids.gtf.gz") for a in argv]
    rme.main(argv_ids + ["--gene_spectra_score_path", str(ids_path)])
    by_id = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")
    pd.testing.assert_frame_equal(by_symbol, by_id)

    bed = tmp_path / "genes.bed"            # BED annotation: no id -> symbol map -> empty universe -> error
    bed.write_text("".join(f"chr1\t{1000 * (i + 1)}\t{1000 * (i + 1) + 500}\t{g}\t0\t+\n" for i, g in enumerate(GENES)))
    argv_bed = [a if a != str(tmp_path / "genes.gtf.gz") else str(bed) for a in argv]
    with pytest.raises(SystemExit, match="motif_min_universe_genes"):
        rme.main(argv_bed + ["--gene_spectra_score_path", str(ids_path)])


# ---------------------------------------------------------------------------
# MotifCompendium FIMO (default database), local Fi-NeMo tables, logos
# ---------------------------------------------------------------------------

def write_meme(path, motifs):
    """Minimal MEME file: {motif id: list of [A, C, G, T] rows}."""
    lines = ["MEME version 4", "", "ALPHABET= ACGT", "", "strands: + -", ""]
    for motif_id, rows in motifs.items():
        lines += [f"MOTIF {motif_id}", f"letter-probability matrix: alength= 4 w= {len(rows)} nsites= 20 E= 0"]
        lines += [" ".join(f"{value:.3f}" for value in row) for row in rows] + [""]
    path.write_text("\n".join(lines))


def test_motifcompendium_fimo_tests_each_cluster_and_uses_database_tf_lists(run_dir):
    """Default --motif_db motifcompendium: KLF-SP_0 and KLF-SP_1 are tested separately (never pooled at '_'),
    motif_family = KLF-SP, candidate genes = the cluster TF list (KLF-SP_1 has no KLF4)."""
    tmp_path, out_dir, run = run_dir
    promoter_rows = [("KLF-SP_0", g, "1e-5") for g in ["G0", "G1", "G2", "G3", "G4"] for _ in range(3)]
    promoter_rows += [("KLF-SP_0", g, "1e-5") for g in GENES[5:30]]
    promoter_rows += [("KLF-SP_0", g, "1e-5") for g in GENES[5:30:2]]              # background varies
    promoter_rows += [("KLF-SP_1", g, "1e-5") for g in GENES[3:30]]
    promoter_rows += [("GATA_0", g, "5e-5") for g in GENES[3:25]]
    write_fimo(tmp_path / "mc_promoter_hits.tsv", promoter_rows)
    write_meme(tmp_path / "mc.meme", {"KLF-SP_0": [[0.1, 0.1, 0.7, 0.1], [0.25, 0.25, 0.25, 0.25]],
                                      "KLF-SP_1": [[0.7, 0.1, 0.1, 0.1]], "GATA_0": [[0.1, 0.1, 0.7, 0.1]]})
    argv = [a for a in base_argv(tmp_path, out_dir, run) if a not in ("--motif_db", "hocomoco_v11")]
    rme.main(argv + ["--promoter_hits", str(tmp_path / "mc_promoter_hits.tsv"), "--motif_element_types", "promoter",
                     "--motif_file", str(tmp_path / "mc.meme")])
    folder = out_dir / run / "Evaluation" / "3_0_2"
    results = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")
    assert set(results["tf"]) == {"KLF-SP_0", "KLF-SP_1", "GATA_0"}, "one test per cluster"
    assert dict(zip(results["tf"], results["motif_family"])) == {"KLF-SP_0": "KLF-SP", "KLF-SP_1": "KLF-SP",
                                                                 "GATA_0": "GATA"}
    assert results.loc[(results["program"] == 1) & (results["tf"] == "KLF-SP_0"), "significant"].iloc[0]
    candidates = pd.read_csv(folder / "3_candidate_tfs.txt", sep="\t")
    klf = candidates[(candidates["tf"] == "KLF-SP_0") & (candidates["program"] == 1)]
    assert {"KLF2", "KLF4", "SP1"} <= set(klf["tf_gene_symbol"]) and "GATA2" not in set(klf["tf_gene_symbol"])
    assert klf.set_index("tf_gene_symbol").loc["KLF4", "evidence_tier"] == "motif+regulator"
    config = json.loads((folder / "3_motif_enrichment_config.yml").read_text())
    assert config["motif_database"] == "motifcompendium" and config["motif_file"] == str(tmp_path / "mc.meme")
    logos = json.loads((folder / "3_motif_logos.json").read_text())["logos"]["fimo"]
    assert "KLF-SP_0" in logos and logos["KLF-SP_0"]["kind"] == "information_content"
    matrix = np.array(logos["KLF-SP_0"]["matrix"])
    assert matrix.shape == (1, 4) and matrix[0, 2] == pytest.approx(0.45, abs=0.01), "uniform (0-bit) flank trimmed"


def test_motifcompendium_default_rejects_hocomoco_hit_tables(run_dir):
    tmp_path, out_dir, run = run_dir
    argv = [a for a in base_argv(tmp_path, out_dir, run) if a not in ("--motif_db", "hocomoco_v11")]
    with pytest.raises(SystemExit, match="hocomoco_v11"):
        rme.main(argv + ["--promoter_hits", str(tmp_path / "promoter_hits.tsv"), "--motif_element_types", "promoter"])


def test_finemo_local_annotation_table(run_dir):
    """--finemo_annotation (export_motif_hits_for_perturbnmf.py) replaces the ENCODE report; tests per
    database cluster with the annotation's candidate_tfs as TF list."""
    tmp_path, out_dir, run = run_dir
    pd.DataFrame({"motif_id": ["cluster_0", "cluster_1"], "motif_label": ["KLF-SP_0", "GATA2"],
                  "database_motif": ["KLF-SP_0", "GATA_1"], "database_match_score": [0.9, 0.85],
                  "candidate_tfs": ["KLF2,KLF4,SP1", "GATA2,GATA3"], "posneg": ["pos", "pos"]}).to_csv(
        tmp_path / "motif_annotation.tsv", sep="\t", index=False)
    rows = []
    for index, gene in enumerate(GENES):
        tss = 1000 * (index + 1)
        rows.append(("chr1", tss - 100, tss - 90, "+", "cluster_1", "GATA2", "pos", 0.5))
        if gene in ("G0", "G1", "G2", "G3", "G4", "KLF4") or index % 7 == 0:
            n = 4 if gene in ("G0", "G1", "G2", "G3", "G4", "KLF4") else 1
            rows += [("chr1", tss - 200 + 12 * j, tss - 190 + 12 * j, "+", "cluster_0", "KLF-SP_0", "pos", 0.5)
                     for j in range(n)]
    hits = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "motif_id", "motif_label", "posneg", "score"])
    with gzip.open(tmp_path / "motif_hits_crispri.tsv.gz", "wt") as handle:
        handle.write("#")
        hits.to_csv(handle, sep="\t", index=False)
    argv = base_argv(tmp_path, out_dir, run) + [
        "--motif_source", "finemo", "--motif_element_types", "promoter",
        "--finemo_instances", str(tmp_path / "motif_hits_crispri.tsv.gz"),
        "--finemo_annotation", str(tmp_path / "motif_annotation.tsv")]
    rme.main(argv)
    folder = out_dir / run / "Evaluation" / "3_0_2"
    results = pd.read_csv(folder / "3_motif_enrichment.txt", sep="\t")
    assert set(results["tf"]) == {"KLF-SP_0", "GATA_1"}
    assert dict(zip(results["tf"], results["motif_family"])) == {"KLF-SP_0": "KLF-SP", "GATA_1": "GATA"}
    assert results.loc[(results["program"] == 1) & (results["tf"] == "KLF-SP_0"), "significant"].iloc[0]
    candidates = pd.read_csv(folder / "3_candidate_tfs.txt", sep="\t")
    klf = candidates[(candidates["tf"] == "KLF-SP_0") & (candidates["program"] == 1)]
    assert set(klf["tf_gene_symbol"]) == {"KLF2", "KLF4", "SP1"}


def test_missing_site_specific_inputs_exit_with_env_var_hint(run_dir, monkeypatch):
    """No built-in resource paths: FIMO scanning without --genome_fasta / --motif_file (and the env vars unset)
    exits naming the flags and their environment variables."""
    tmp_path, out_dir, run = run_dir
    monkeypatch.setattr(rme, "DEFAULT_GENOME_FASTA", None)
    monkeypatch.setitem(rme.MOTIF_DATABASE_FILES, "hocomoco_v11", None)
    argv = base_argv(tmp_path, out_dir, run) + ["--motif_element_types", "promoter", "--genome_fasta", ""]
    with pytest.raises(SystemExit) as error:
        rme.main(argv)
    message = str(error.value)
    assert "--genome_fasta (or $PERTURBNMF_GENOME_FASTA)" in message
    assert "--motif_file (or $PERTURBNMF_HOCOMOCO_V11_MEME)" in message
