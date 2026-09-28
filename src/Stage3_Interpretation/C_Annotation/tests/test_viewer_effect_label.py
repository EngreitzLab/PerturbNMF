"""settings.effect_label names the regulator effect in both viewers (axis, headers, tooltips).

Dimensions: setting absent / set; program viewer / regulator-group viewer. Behaviour: absent ->
"log2FC"; set -> the page carries the custom label and no hard-coded "log2FC" anywhere, because
every effect label in the page template reads META.effect_label.
"""
import json
import re
import sys

import build_group_viewer as group_viewer
from test_build_annotation_viewer import build_program_zero, viewer
from viewer_common import DEFAULT_EFFECT_LABEL, read_effect_label

CUSTOM = "Calibrated t-statistic"
HARD_CODED_FOLD_CHANGE = re.compile(r"log2 ?FC|log₂ ?FC|[Ff]old change")  # data keys (log2fc) are not labels


def render_program_viewer(tmp_path, monkeypatch, settings_update: dict) -> str:
    """Build the one-program fixture screen with the given settings and return the written page."""
    config_path = tmp_path / "config.json"
    build_program_zero(tmp_path, {})
    config = json.loads(config_path.read_text())
    config["settings"] = {**config["settings"], **settings_update}
    config_path.write_text(json.dumps(config))
    output = tmp_path / "viewer.html"
    monkeypatch.setattr(sys, "argv", ["build_annotation_viewer.py", "--config", str(config_path),
                                      "--dispatch", str(tmp_path / "dispatch"), "--arm", "v3",
                                      "--output", str(output)])
    assert viewer.main() == 0
    return output.read_text()


def meta_of(page: str) -> dict:
    return json.loads(re.search(r"const META = (\{.*?\});\n", page).group(1))


def test_default_effect_label_is_log2fc():
    assert read_effect_label({}) == DEFAULT_EFFECT_LABEL == "log2FC"
    assert read_effect_label({"effect_label": ""}) == "log2FC", "an empty label falls back to the default"


def test_program_viewer_default_renders_log2fc(tmp_path, monkeypatch):
    page = render_program_viewer(tmp_path, monkeypatch, {})
    assert meta_of(page)["effect_label"] == "log2FC"


def test_program_viewer_custom_label_replaces_every_effect_label(tmp_path, monkeypatch):
    page = render_program_viewer(tmp_path, monkeypatch, {"effect_label": CUSTOM})
    assert meta_of(page)["effect_label"] == CUSTOM
    leftovers = HARD_CODED_FOLD_CHANGE.findall(page)
    assert not leftovers, f"page still hard-codes a fold-change label: {leftovers}"


def test_group_viewer_template_has_no_hard_coded_effect_label():
    leftovers = HARD_CODED_FOLD_CHANGE.findall(group_viewer.PAGE)
    assert not leftovers, f"group viewer template hard-codes a fold-change label: {leftovers}"
    assert "const EFFECT = META.effect_label" in group_viewer.PAGE
    assert "read_effect_label(settings)" in open(group_viewer.__file__).read(), \
        "the group viewer META must take effect_label from the settings"
