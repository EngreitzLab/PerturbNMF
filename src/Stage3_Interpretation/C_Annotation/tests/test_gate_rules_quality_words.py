"""The quality-word gate must not fire on the hnRNP family name."""
from gate_rules import label_problems, summary_problems


def test_hnrnp_family_name_is_not_a_quality_comment():
    summary = "Mitotic genes alongside heterogeneous nuclear ribonucleoproteins (HNRNPA1, HNRNPH1)."
    assert summary_problems("P1", summary) == []
    assert label_problems("P1", "Heterogeneous nuclear ribonucleoprotein splicing", set()) == []


def test_quality_words_still_caught():
    assert summary_problems("P1", "A heterogeneous program of mixed genes.")
    assert summary_problems("P1", "hnRNPs (heterogeneous nuclear ribonucleoproteins) and a heterogeneous rest")
    assert label_problems("P1", "Heterogeneous stress genes", set())
