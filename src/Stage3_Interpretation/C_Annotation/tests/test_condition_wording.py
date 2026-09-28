"""Multi-condition prompts are worded by condition; only an ordered design gets ordering language."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ProgramAnnotatorV3" / "scripts"))
import conditions as condition_config  # noqa: E402
from build_annotation_prompts import build_prompt  # noqa: E402
from build_group_prompts import build_prompt as build_group_prompt  # noqa: E402
from validate_annotation_answers import validate_condition_dependence  # noqa: E402

from test_validate_group_answers import EVIDENCE  # noqa: E402

GENES = [f"GENE{i}" for i in range(40)]
CONDITIONS = [{"label": "A", "description": "donor A"}, {"label": "B", "description": "donor B"}]
TIME_WORDS = ("time course", "differentiat", " day", "trajector", "stage", "temporal", "in order",
              "delay", "before or at the peak", "contiguous block")


def resources(composition: bool = False) -> dict:
    loading = pd.DataFrame({"program_id": 0, "Name": GENES, "Score": [1.0 / (i + 1) for i in range(40)],
                            "UniquenessScore": [float(i) for i in range(40)]})
    regulators = pd.DataFrame([
        {"program_id": 0, "target_gene": "REG1", "log2_fc": -1.0, "adj_pval": 0.001, "significant": True, "condition": "A"},
        {"program_id": 0, "target_gene": "REG1", "log2_fc": -0.2, "adj_pval": 0.5, "significant": False, "condition": "B"},
    ])
    screen = {
        "positional": {}, "gene_biotypes": {"protein_coding": 30}, "symbol_families": {}, "marker_sets": {},
        "cis_targets_in_top_genes": {"n_overlap": 0, "genes": []}, "regulators": {"n_tested": 2, "n_significant": 1},
    }
    if composition:
        screen["condition_composition"] = {
            "condition_order": ["A", "B"], "share_of_total": {"A": 0.7, "B": 0.3}, "peak_condition": "A",
            "peak_share": 0.7, "peak_over_second": 2.3, "peak_over_lowest": 2.3,
            "marker_overlap": {c: {"description": c, "genes": [], "n_overlap": 0, "expected": 0.1, "p_value": None}
                               for c in ("A", "B")},
        }
    return {
        "loading": loading, "regulators": regulators,
        "enrichment": pd.DataFrame(columns=["program_id", "category", "description", "fdr", "inputGenes"]),
        "coordinates": {}, "screens": {"0": screen}, "ncbi": {}, "excluded_pmids": frozenset(),
        "conditions": condition_config.normalise_conditions(CONDITIONS),
        "activity": pd.DataFrame({"program_id": 0, "condition": ["A", "B"], "mean_score": [0.7, 0.3]}),
    }


def settings(**extra) -> dict:
    return {"dataset_name": "test", "cell_system": "primary cells", "assay": "Perturb-seq", "k": 10,
            "significance_label": "adj p < 0.05", "annotation_role": "cell biologist", **extra}


def prompt_text(request: dict) -> str:
    return request["params"]["system"] + "\n" + request["params"]["messages"][0]["content"]


def test_unordered_prompt_has_no_time_course_wording():
    text = prompt_text(build_prompt(0, resources(), settings(condition_design="unordered", condition_variable="donor")))
    assert "MULTI-CONDITION SCREEN" in text and "condition variable: donor" in text.lower()
    assert '"condition_dependence"' in text and "Peak condition: A" in text
    lowered = text.lower()
    assert not [word for word in TIME_WORDS if word in lowered]


def test_ordered_prompt_adds_ordering_language():
    text = prompt_text(build_prompt(0, resources(composition=True), settings(condition_design="ordered",
                                                                            condition_variable="timepoint")))
    assert "Conditions, in order" in text and "before or at the peak condition" in text
    assert "condition_composition" in text and "one contiguous block of conditions" in text


def test_deprecated_keys_still_work(capsys):
    condition_config.warned.clear()
    old = [{"label": "A", "stage": "donor A"}, {"label": "B", "stage": "donor B"}]
    assert condition_config.normalise_conditions(old) == CONDITIONS
    assert condition_config.read_condition_design({"condition_design": "time_course"}) == "ordered"
    assert condition_config.read_condition_design({"condition_design": "groups"}) == "unordered"
    assert condition_config.read_setting({"differentiation_delay_description": "x"},
                                         "shared_state_shift_description", "differentiation_delay_description") == "x"
    assert capsys.readouterr().err.count("deprecated") == 4


def test_unordered_group_prompt_has_no_delay_wording():
    request = build_group_prompt(EVIDENCE, settings(condition_variable="donor"), CONDITIONS, 298, True)
    text = prompt_text(request)
    assert "shared_state_shift" in text and "condition variable: donor" in text
    lowered = text.lower()
    assert not [word for word in TIME_WORDS if word in lowered]


def test_group_prompt_state_shift_override_and_ordered_default():
    ordered = prompt_text(build_group_prompt(EVIDENCE, settings(condition_design="ordered"), CONDITIONS, 298, True))
    assert "progression through the ordered conditions" in ordered
    custom = prompt_text(build_group_prompt(
        EVIDENCE, settings(shared_state_shift_description="all members shift the cells to state X"), CONDITIONS, 298, True))
    assert "all members shift the cells to state X" in custom


@pytest.mark.parametrize("slot, key", [("condition_dependence", "regulator_pattern"),
                                       ("temporal_window", "regulator_timing")])
def test_condition_dependence_gate(slot, key):
    prompt = prompt_text(build_prompt(0, resources(), settings(condition_design="unordered")))
    answer = {"interpretation": {slot: {"claim": "x", "peak_condition": "A",
                                        key: [{"symbol": "REG1", "conditions": ["A", "B"]}]}}}
    problems, warnings = validate_condition_dependence(0, answer, prompt)
    assert problems == [] and any("REG1 placed in ['B']" in w for w in warnings)
    answer["interpretation"][slot]["peak_condition"] = "B"
    problems, _ = validate_condition_dependence(0, answer, prompt)
    assert any("prompt says A" in p for p in problems)
