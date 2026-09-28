"""Citation gate: tokeniser spaces before punctuation, and an id slip to a ref beyond the list."""
import json

from validate_citation_answers import is_verbatim, normalise, validate_program

SENTENCE = "The familial form of CCM has been linked to three genes: KRIT1 / CCM1, MGC4607 / CCM2 , and PDCD10 / CCM3 ."


def test_quote_without_tokeniser_space_is_verbatim():
    quote = "linked to three genes: KRIT1 / CCM1, MGC4607 / CCM2, and PDCD10 / CCM3"
    assert is_verbatim(normalise(quote).lower(), normalise(SENTENCE).lower())
    assert not is_verbatim(normalise("linked to four genes").lower(), normalise(SENTENCE).lower())


def write_program(tmp_path, ref):
    directory = tmp_path / "cite_p1"
    directory.mkdir()
    claim = {"claim_id": "R1", "symbol": "PDCD10", "database": [], "literature": [
        {"pmid": "111", "sentence": "Unrelated sentence.", "title": "A", "year": "2000"},
        {"pmid": "222", "sentence": SENTENCE, "title": "B", "year": "2001"},
    ]}
    (directory / "candidates.json").write_text(json.dumps({"claims": [claim]}))
    answer = {"claims": [{"claim_id": "R1", "supports": [
        {"ref": ref, "type": "literature", "role": "context", "pmid": "222",
         "quote": "linked to three genes", "strength": "direct"}]}]}
    (directory / "answer.json").write_text(json.dumps(answer))
    return directory


def test_ref_beyond_the_list_resolves_as_an_id_slip(tmp_path):
    warnings = []
    from collections import Counter
    assert validate_program(write_program(tmp_path, "L3"), Counter(), warnings) == []
    assert any("resolved to L2" in w for w in warnings)


def test_ref_beyond_the_list_without_a_match_fails(tmp_path):
    from collections import Counter
    directory = write_program(tmp_path, "L3")
    answer = json.loads((directory / "answer.json").read_text())
    answer["claims"][0]["supports"][0]["pmid"] = "999"
    (directory / "answer.json").write_text(json.dumps(answer))
    assert any("was not offered" in p for p in validate_program(directory, Counter(), []))
