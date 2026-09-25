"""The group gate on a prompt built by build_group_prompts.py and hand-written answers."""
import copy
import json

import pytest

from build_group_prompts import build_prompt
from validate_group_answers import validate

EVIDENCE = {
    "group_id": 7, "stability": 0.8, "mean_raw_r": 0.5, "mean_corrected_r": 0.9,
    "strength_tiers": {"weak": 1, "medium": 1, "strong": 1},
    "members": [
        {"gene": g, "role": role, "stability": 0.8, "r_to_centroid": 0.8, "reliability": 0.6,
         "strength_tier": "strong", "n_significant": 10, "connected_to": [],
         "promoter": {"decision": "clear", "reasons": [], "shared_locus_with": []}, "summary": ""}
        for g, role in (("DGCR8", "core"), ("DROSHA", "core"), ("ZNF999", "peripheral"))
    ],
    "excluded": [{"gene": "C5orf22", "reasons": ["shares a promoter with DROSHA"]}],
    "signature": [{"feature": "P29|D0", "program_id": 29, "condition": "D0", "mean_log2fc": -1.4,
                   "members_significant_same_direction": 3, "members": 3, "program_label": "Pluripotency"}],
    "string_edges": [], "ppi_enrichment": None, "enrichment": [], "complexes": [],
    "reference_pool": [{"pmid": "34319763", "year": "2021", "genes": ["DGCR8", "DROSHA"], "sentence": "Microprocessor."}],
}
SETTINGS = {"cell_system": "hiPSCs", "dataset_name": "test", "k": 50}

GOOD = {
    "group_id": 7, "label": "Microprocessor", "label_family": "miRNA biogenesis", "label_distinguisher": "",
    "brief_summary": "DGCR8 and DROSHA lower pluripotency program 29.",
    "confounder_assessment": [{"confounder": c, "status": "ruled_out", "evidence": "x"} for c in
                              ("generic_fitness_or_stress", "differentiation_delay", "promoter_neighbour", "weak_effect_noise")],
    "shared_function": {"claim": "Microprocessor", "support_members": ["DGCR8", "DROSHA"], "pmids": ["34319763"]},
    "why_here": {"claim": "x", "programs": [{"program_id": 29, "condition": "D0", "direction": "down"}], "pmids": []},
    "regulators": [
        {"symbol": "DGCR8", "role": "core_explained", "confidence": "high", "hypothesis": "", "what_would_test_it": ""},
        {"symbol": "DROSHA", "role": "core_explained", "confidence": "high", "hypothesis": "", "what_would_test_it": ""},
        {"symbol": "ZNF999", "role": "unexplained", "confidence": "low", "hypothesis": "binds pri-miRNA loci",
         "what_would_test_it": "CLIP"},
    ],
    "label_evidence": {"regulators": [{"symbol": "DGCR8", "why": "x"}], "complexes": [], "terms": []},
    "competing_readings": [{"reading": "a"}, {"reading": "b"}],
    "coherence": "strong", "citations": [{"pmid": "PMID:34319763"}], "open_questions": [],
}


@pytest.fixture
def run(tmp_path):
    request = build_prompt(EVIDENCE, SETTINGS, [{"label": "D0", "stage": "hiPSC"}, {"label": "D1", "stage": "PS"}], 298, True)
    directory = tmp_path / "rg_p7"
    directory.mkdir()
    (directory / "prompt.md").write_text(request["params"]["system"] + "\n\n" + request["params"]["messages"][0]["content"])

    def check(answer):
        (directory / "answer.json").write_text(json.dumps(answer))
        return validate(7, directory)
    return check


def test_good_answer_passes(run):
    problems, warnings = run(GOOD)
    assert problems == [] and warnings == []


@pytest.mark.parametrize("mutate, expected", [
    (lambda a: a["regulators"].pop(), "members without a role: ZNF999"),
    (lambda a: a["regulators"].append({"symbol": "C5orf22", "role": "consistent", "confidence": "low"}),
     "excluded member(s) interpreted: C5orf22"),
    (lambda a: a["shared_function"]["support_members"].append("MYC"), "non-member(s)"),
    (lambda a: a["citations"].append({"pmid": "11111111"}), "not in the reference pool: 11111111"),
    (lambda a: a["regulators"][2].update(hypothesis=""), "unexplained member ZNF999 needs a hypothesis"),
    (lambda a: a.update(label="Heterogeneous regulators cluster"), "banned word"),
    (lambda a: a["confounder_assessment"].pop(), "confounders not assessed: weak_effect_noise"),
])
def test_bad_answers_fail(run, mutate, expected):
    answer = copy.deepcopy(GOOD)
    mutate(answer)
    problems, _ = run(answer)
    assert any(expected in p for p in problems), problems


def test_program_outside_signature_warns(run):
    answer = copy.deepcopy(GOOD)
    answer["why_here"]["programs"].append({"program_id": 3})
    problems, warnings = run(answer)
    assert problems == [] and any("program 3" in w for w in warnings)
