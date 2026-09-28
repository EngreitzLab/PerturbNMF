"""Unit tests for Stage2_Evaluation/A_Metrics/Slurm_Version/merge_motif_tables_at_peak_condition.py.

Test strategy
  peak condition:   table entry wins; missing / non-run condition falls back to the highest mean score;
                    neither given -> exit; program without a peak -> exit
  merge:            condition-independent (fimo) rows from the first run only; per-condition (finemo) rows
                    from each program's peak run; peak_condition column; by-condition table keeps all runs
  side files:       logos unioned; pattern names copied; config records the peak and how it was chosen
"""

import importlib.util
import json
import os

import pandas as pd
import pytest

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "Stage2_Evaluation", "A_Metrics",
                           "Slurm_Version", "merge_motif_tables_at_peak_condition.py")
spec = importlib.util.spec_from_file_location("merge_motif_tables_at_peak_condition", SCRIPT_PATH)
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)

COLUMNS = ["program", "element_type", "tf", "fdr", "enrichment", "significant", "motif_source"]


def write_run(directory, condition, logo_motif):
    directory.mkdir(parents=True)
    rows = [(program, "promoter", "KLF_0", 0.01, 2.0, True, "fimo") for program in (0, 1)]
    rows += [(program, "enhancer", f"GATA_{condition}", 0.02, 1.5, True, "finemo") for program in (0, 1)]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(directory / "3_motif_enrichment.txt", sep="\t", index=False)
    candidates = pd.DataFrame([(0, "enhancer", f"GATA_{condition}", "GATA2", "motif+expressed", "finemo"),
                               (1, "promoter", "KLF_0", "KLF4", "motif+regulator", "fimo")],
                              columns=["program", "element_type", "tf", "tf_gene_symbol", "evidence_tier", "motif_source"])
    candidates.to_csv(directory / "3_candidate_tfs.txt", sep="\t", index=False)
    with open(directory / "3_motif_logos.json", "w") as handle:
        json.dump({"logos": {"finemo": {logo_motif: {"matrix": [[0.25] * 4]}}}}, handle)
    pd.DataFrame({"pattern_id": ["p0"], "tf": ["GATA_0"]}).to_csv(directory / "3_finemo_pattern_names.tsv",
                                                                   sep="\t", index=False)
    with open(directory / "3_motif_enrichment_config.yml", "w") as handle:
        json.dump({"arguments": {"motif_method": "ttest", "n_top": 300}}, handle)


@pytest.fixture
def runs(tmp_path):
    write_run(tmp_path / "ctrl", "ctrl", "GATA_ctrl")
    write_run(tmp_path / "stim", "stim", "GATA_stim")
    pd.DataFrame({"program": [0, 1], "condition": ["stim", "unassigned"]}).to_csv(tmp_path / "peaks.csv", index=False)
    pd.DataFrame({"program": [0, 0, 1, 1], "condition": ["ctrl", "stim", "ctrl", "stim"],
                  "mean_score": [0.9, 0.1, 0.7, 0.2]}).to_csv(tmp_path / "activity.csv", index=False)
    return tmp_path


def test_peak_table_wins_and_activity_fills_the_rest(runs):
    peaks, rules = merge.read_peak_conditions(["ctrl", "stim"], str(runs / "peaks.csv"), str(runs / "activity.csv"))
    assert peaks == {0: "stim", 1: "ctrl"}, "program 0 from the table; program 1 ('unassigned') from the top mean score"
    assert rules == {0: "table", 1: "max_mean_score"}


def test_peak_inputs_required(runs):
    with pytest.raises(SystemExit):
        merge.read_peak_conditions(["ctrl", "stim"])


def test_merge_takes_condition_rows_from_the_peak_run(runs):
    out = runs / "merged"
    merge.main(["--K", "3", "--condition_runs", f"ctrl={runs / 'ctrl'}", f"stim={runs / 'stim'}",
                "--peak_conditions", str(runs / "peaks.csv"), "--program_activity", str(runs / "activity.csv"),
                "--out_evaluation_dir", str(out)])
    results = pd.read_csv(out / "3_motif_enrichment.txt", sep="\t", keep_default_na=False)
    fimo = results[results["motif_source"] == "fimo"]
    assert len(fimo) == 2 and set(fimo["peak_condition"]) == {""}, "fimo rows once, from the first run"
    finemo = results[results["motif_source"] == "finemo"].set_index("program")
    assert finemo.loc[0, "tf"] == "GATA_stim" and finemo.loc[0, "peak_condition"] == "stim"
    assert finemo.loc[1, "tf"] == "GATA_ctrl" and finemo.loc[1, "peak_condition"] == "ctrl"
    by_condition = pd.read_csv(out / "3_motif_enrichment_by_condition.txt", sep="\t")
    assert len(by_condition) == 4 and set(by_condition["condition"]) == {"ctrl", "stim"}
    candidates = pd.read_csv(out / "3_candidate_tfs.txt", sep="\t", keep_default_na=False)
    assert sorted(candidates["tf"]) == ["GATA_stim", "KLF_0"], "program 0's finemo candidate from its peak (stim) only"
    logos = json.loads((out / "3_motif_logos.json").read_text())
    assert set(logos["logos"]["finemo"]) == {"GATA_ctrl", "GATA_stim"}
    assert (out / "3_finemo_pattern_names.tsv").exists()
    config = json.loads((out / "3_motif_enrichment_config.yml").read_text())
    assert config["merge"]["peak_condition"] == {"0": "stim", "1": "ctrl"}
    assert config["merge"]["peak_chosen_by"] == {"0": "table", "1": "max_mean_score"}
    assert config["first_run_config"]["arguments"]["n_top"] == 300
    assert config["n_significant"] == {"enhancer_finemo": 2, "promoter_fimo": 2}


def test_program_without_peak_exits(runs, tmp_path):
    pd.DataFrame({"program": [0], "condition": ["stim"]}).to_csv(tmp_path / "partial.csv", index=False)
    with pytest.raises(SystemExit, match="without a peak condition"):
        merge.main(["--K", "3", "--condition_runs", f"ctrl={runs / 'ctrl'}", f"stim={runs / 'stim'}",
                    "--peak_conditions", str(tmp_path / "partial.csv"), "--out_evaluation_dir", str(tmp_path / "o")])
