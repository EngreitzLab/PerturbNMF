"""settings.effect_label in the ProgramAnnotatorV3 prompts: the default ("log2FC", or no key) gives
byte-identical prompts; another label replaces only the word log2FC in text the builder writes."""
import json

from test_annotation_prompt_motif_section import SETTINGS, prompts, resources, with_motifs  # noqa: F401

LABEL = "Calibrated t-statistic vs controls"


def build_all(resources, settings):
    return [prompts.build_prompt(pid, resources, settings) for pid in (0, 1)]


def single_condition(resources):
    regulators = resources["regulators"]
    return {key: value for key, value in resources.items() if key not in ("conditions", "activity")} | {
        "regulators": regulators[regulators["condition"] == "D0"].drop(columns="condition")}


def test_default_label_gives_byte_identical_prompts(resources, tmp_path):
    for variant in (resources, with_motifs(resources, tmp_path), single_condition(resources)):
        plain = json.dumps(build_all(variant, SETTINGS))
        assert "log2FC" in plain, "the fixture must exercise the label"
        for settings in ({**SETTINGS, "effect_label": "log2FC"}, {**SETTINGS, "effect_label": ""}):
            assert json.dumps(build_all(variant, settings)) == plain, \
                f"effect_label={settings['effect_label']!r} must leave the prompts byte-identical"


def test_label_replaces_only_the_effect_name(resources, tmp_path):
    for variant in (with_motifs(resources, tmp_path), single_condition(resources)):
        plain = build_all(variant, SETTINGS)
        labelled = build_all(variant, {**SETTINGS, "effect_label": LABEL})
        text = json.dumps(labelled)
        assert "log2FC" not in text, "every printed effect name follows effect_label"
        assert LABEL in labelled[0]["params"]["system"], "system rule 4 names the effect"
        assert f"{LABEL}=" in labelled[0]["params"]["messages"][0]["content"], "regulator lines name the effect"
        assert '"log2fc": <float>' in labelled[0]["params"]["messages"][0]["content"], \
            "the answer schema keeps its log2fc fields"
        assert text.replace(LABEL, "log2FC") == json.dumps(plain), "nothing but the label may change"


def test_motif_candidate_line_uses_label(resources, tmp_path):
    user = prompts.build_prompt(0, with_motifs(resources, tmp_path), {**SETTINGS, "effect_label": LABEL})
    assert f"KLF4 (motif+regulator; knockdown {LABEL}=-0.80, adj p=1.0e-04)" in user["params"]["messages"][0]["content"]


def test_label_is_not_applied_to_literature_evidence(resources):
    ncbi = {**resources["ncbi"]}
    ncbi["0"] = {**ncbi["0"], "evidence_snippets": {"POU5F1": ["OCT4 loss gave log2FC -2 (PMID:12345678)"]}}
    user = prompts.build_prompt(0, {**resources, "ncbi": ncbi}, {**SETTINGS, "effect_label": LABEL})
    assert "log2FC -2" in user["params"]["messages"][0]["content"], "quoted evidence must stay verbatim"
