"""Unit tests for the label rules (1.Annotate/validate_program.check_label) and the set-level label collision pass
(2.Synthesis/resolve_label_collisions). No model calls."""

import json
import sys
from pathlib import Path

import pytest

C_ANNOTATION = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "Stage3_Interpretation"
    / "C_Annotation"
)
sys.path[:0] = [
    str(C_ANNOTATION / "2.Synthesis"),
    str(C_ANNOTATION / "api"),
    str(C_ANNOTATION / "0.Prepare"),
    str(C_ANNOTATION / "1.Annotate"),
]
from validate_program import capitalize_distinguisher, check_label  # noqa: E402

resolve = pytest.importorskip("resolve_label_collisions")

TOP_GENES = [
    "CCNB1",
    "CDK1",
    "TOP2A",
    "CENPF",
    "MKI67",
    "AURKA",
    "PLK1",
    "BUB1",
    "KIF20A",
    "CDC20",
    "COL4A1",
]


def write_program(d, label, regulators=("E2F1",)):
    d.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{i + 1}\t{g}\t0.01\t0.01" for i, g in enumerate(TOP_GENES))
    (d / "genes_top300.tsv").write_text(
        f"rank\tgene\tloading\tdistinctive_score\n{rows}\n"
    )
    regs = "\n".join(f"{r}\tEC\t1.0\t1e-5\tactivator" for r in regulators)
    (d / "regulators_sig.tsv").write_text(
        f"regulator\tcondition\tlog2fc\tadj_p\trole\n{regs}\n"
    )
    (d / "label.json").write_text(json.dumps(label))
    return d


def make_label(name, family, distinguisher, **extra):
    return {
        "program_name": name,
        "label_family": family,
        "label_distinguisher": distinguisher,
        "distinguisher_evidence": "CCNB1, CDK1 among the top genes",
        "function_sentence": "Mitotic genes such as CCNB1, CDK1 and TOP2A.",
        "anchor_genes": ["CCNB1", "CDK1", "TOP2A"],
        "basis": "genes_top300.tsv",
        **extra,
    }


@pytest.mark.parametrize(
    "name, family, distinguisher",
    [
        ("Cell cycle - G2M", "Cell cycle", "G2M"),
        ("Cell cycle", "Cell cycle", ""),
        ("Lysosome, OXPHOS, cholesterol synthesis", "", ""),
        ("Translation - tRNA charging", "Translation", "tRNA charging"),
    ],
)
def test_check_label_accepts(tmp_path, name, family, distinguisher):
    assert (
        check_label(write_program(tmp_path, make_label(name, family, distinguisher)))
        == []
    )


@pytest.mark.parametrize(
    "name, family, distinguisher, expected",
    [
        ("Cell cycle - mitosis", "Cell cycle", "mitosis", "capitalize"),
        ("Cell cycle - CDK1", "Cell cycle", "CDK1", "bare gene symbol"),
        ("Cell cycle - E2F1", "Cell cycle", "E2F1", "bare gene symbol"),
        ("Cell cycle program - G2M", "Cell cycle program", "G2M", "banned word"),
        ("Cell cycle (CDK1)", "Cell cycle (CDK1)", "", "parenthesized"),
        ("Cell cycle - G2M", "Cell cycle", "Mitosis", "must be"),
        ("Mitosis", "", "Mitosis", "without a label_family"),
        (
            "Cell cycle - Late G2M phase mitotic spindle",
            "Cell cycle",
            "Late G2M phase mitotic spindle",
            "more than",
        ),
    ],
)
def test_check_label_rejects(tmp_path, name, family, distinguisher, expected):
    errors = check_label(
        write_program(tmp_path, make_label(name, family, distinguisher))
    )
    assert any(expected in e for e in errors), errors


def test_check_label_needs_family_fields(tmp_path):
    label = make_label("Cell cycle - G2M", "Cell cycle", "G2M")
    del label["label_family"]
    assert any("label_family" in e for e in check_label(write_program(tmp_path, label)))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Extracellular matrix - degradation", "Extracellular matrix - Degradation"),
        ("Cell migration - Focal adhesion", "Cell migration - Focal adhesion"),
        ("Translation - tRNA charging", "Translation - tRNA charging"),
        ("Signaling - mTOR pathway", "Signaling - mTOR pathway"),
        ("Lysosome, OXPHOS", "Lysosome, OXPHOS"),
    ],
)
def test_capitalize_distinguisher(name, expected):
    assert capitalize_distinguisher(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Cell cycle - G2M", ("Cell cycle", "G2M")),
        ("Cell cycle", ("Cell cycle", "")),
        ("Lysosome, OXPHOS", ("", "")),
    ],
)
def test_split_label(name, expected):
    assert resolve.split_label(name) == expected


def test_find_collisions_groups_by_family_and_near_name():
    labels = {
        "P1": make_label("Cell cycle - G2M", "Cell cycle", "G2M"),
        "P2": make_label("Cell cycle - G1S", "Cell cycle", "G1S"),
        "P3": make_label("Interferon response", "Interferon response", ""),
        "P4": make_label("Interferon alpha response", "Interferon alpha response", ""),
        "P5": make_label("Golgi trafficking", "Golgi trafficking", ""),
    }
    groups = sorted(sorted(g) for g in resolve.find_collisions(labels, 0.6))
    assert groups == [["P1", "P2"], ["P3", "P4"]]


def resolved(pid, label, **kw):
    fields = dict(
        program_id=pid,
        label=label,
        distinguisher=label.split(" - ")[-1],
        evidence="CENPF distinctive",
        belongs_to_family=True,
        kept_verbatim=False,
        used_bare_number=False,
    )
    return resolve.ResolvedLabel(**{**fields, **kw})


def test_apply_label_rewrites_downstream_files_and_keeps_the_agent_name(tmp_path):
    d = write_program(tmp_path, make_label("Cell cycle - G2M", "Cell cycle", "G2M"))
    (d / "interpretation.json").write_text(json.dumps({"label": "Cell cycle - G2M"}))
    (d / "evidence.json").write_text(json.dumps({"label": "Cell cycle - G2M"}))
    (d / "driver_genes.json").write_text(
        json.dumps({"annotation": "Cell cycle - G2M", "genes": {}})
    )
    label = json.loads((d / "label.json").read_text())
    resolve.apply_label(
        d,
        label,
        "Cell cycle - Mitotic spindle",
        resolved("P1", "Cell cycle - Mitotic spindle"),
    )

    out = json.loads((d / "label.json").read_text())
    assert out["program_name"] == "Cell cycle - Mitotic spindle"
    assert (out["label_family"], out["label_distinguisher"]) == (
        "Cell cycle",
        "Mitotic spindle",
    )
    assert out["program_name_before_disambiguation"] == "Cell cycle - G2M"
    assert out["before_disambiguation"]["label_distinguisher"] == "G2M"
    assert (
        json.loads((d / "interpretation.json").read_text())["label"]
        == "Cell cycle - Mitotic spindle"
    )
    assert (
        json.loads((d / "evidence.json").read_text())["label"]
        == "Cell cycle - Mitotic spindle"
    )
    assert (
        json.loads((d / "driver_genes.json").read_text())["annotation"]
        == "Cell cycle - Mitotic spindle"
    )
    assert check_label(d) == []


def test_apply_label_twice_keeps_the_original_agent_fields(tmp_path):
    d = write_program(tmp_path, make_label("Cell cycle - G2M", "Cell cycle", "G2M"))
    for new in ("Cell cycle - Mitotic spindle", "Cell cycle - Chromosome segregation"):
        label = json.loads((d / "label.json").read_text())
        resolve.apply_label(d, label, new, resolved("P1", new))
    out = json.loads((d / "label.json").read_text())
    assert out["program_name_before_disambiguation"] == "Cell cycle - G2M"
    assert resolve.agent_name(out) == "Cell cycle - G2M"
    assert resolve.agent_field(out, "label_family") == "Cell cycle"


def test_collisions_are_found_from_agent_names_after_a_rewrite(tmp_path):
    """Detection must read the agent's own fields, so a rerun finds the same groups and its cache key holds."""
    labels = {
        "P1": make_label("Cell cycle - G2M", "Cell cycle", "G2M"),
        "P2": make_label("Cell cycle - G1S", "Cell cycle", "G1S"),
    }
    before = sorted(sorted(g) for g in resolve.find_collisions(labels, 0.6))
    for pid, new in (
        ("P1", "Mitosis - Spindle"),
        ("P2", "DNA replication - Origin licensing"),
    ):
        d = write_program(tmp_path / pid, labels[pid])
        resolve.apply_label(d, labels[pid], new, resolved(pid, new))
        labels[pid] = json.loads((d / "label.json").read_text())
    assert sorted(sorted(g) for g in resolve.find_collisions(labels, 0.6)) == before


def test_check_resolution_flags_missing_members_and_rule_breaks():
    result = resolve.GroupResolution(
        family="Cell cycle",
        programs=[
            resolved("P1", "Cell cycle - G2M program"),
            resolved("P3", "Cell cycle (CDK1)"),
        ],
    )
    errors = resolve.check_resolution(result, ["P1", "P2"])
    assert any("covers" in e for e in errors)
    assert any("program" in e for e in errors)
    assert any("parenthesized" in e for e in errors)
