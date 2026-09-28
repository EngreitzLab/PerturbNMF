"""Unit tests for find_regulatory_resources.py (ranking + manifest), with mocked portal JSON."""

import os
import sys

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src", "Stage2_Evaluation", "A_Metrics", "src"))

import pandas as pd
import pytest

import find_regulatory_resources as frr


# ---------------------------------------------------------------------------
# Query parsing / normalization
# ---------------------------------------------------------------------------

def test_normalize_ontology_id_variants():
    assert frr.normalize_ontology_id("CL:1000413") == "CL:1000413"
    assert frr.normalize_ontology_id("CL_1000413") == "CL:1000413"
    assert frr.normalize_ontology_id("/sample-terms/CL_1000413/") == "CL:1000413"


def test_parse_query_requires_something():
    with pytest.raises(ValueError):
        frr.parse_query(None, None)


EXAMPLE_SYNONYMS = os.path.join(
    PIPELINE_ROOT, "src", "Stage2_Evaluation", "A_Metrics", "src", "resources", "cell_type_synonyms.example.json"
)
EXAMPLE_QUERY = "immortalized hepatocyte line"


def example_query(ontology_id=None):
    return frr.parse_query(EXAMPLE_QUERY, ontology_id, frr.load_matching_rules(EXAMPLE_SYNONYMS))


def test_example_synonyms_file_loads():
    rules = frr.load_matching_rules(EXAMPLE_SYNONYMS)
    assert "immortalized hepatocyte" in rules.synonyms
    assert [label for _, _, label in rules.tier_rules][0] == "hepatocyte"
    assert rules.tissue_rule is not None


def test_no_synonyms_file_means_no_curated_terms():
    query = frr.parse_query(EXAMPLE_QUERY, None)
    assert query.curated_terms() == []


def test_curated_terms_match_example_alias():
    terms = example_query().curated_terms()
    assert ("hepatocyte", "CL:0000182") in terms


def test_tier_score_outside_band_is_rejected(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('{"synonyms": {}, "tier_rules": [{"pattern": "x", "score": 95}]}')
    with pytest.raises(ValueError):
        frr.load_matching_rules(str(path))


# ---------------------------------------------------------------------------
# Scoring tiers
# ---------------------------------------------------------------------------

def make_candidate(**kwargs) -> frr.Candidate:
    defaults = dict(
        portal="ENCODE", resource_type=frr.RESOURCE_E2G, dataset_accession="ENCSR000AAA",
        file_accession="ENCFF000AAA", biosample_term_name="", ontology_id="",
        assembly="GRCh38", file_format="bed", output_type="thresholded element gene links",
        size=100, download_url="https://example.org/x.bed",
    )
    defaults.update(kwargs)
    return frr.Candidate(**defaults)


def test_score_ontology_id_exact_beats_term_name():
    query = frr.parse_query("hepatocyte", "CL:0000182")
    exact_id = make_candidate(biosample_term_name="something else", ontology_id="CL:0000182")
    exact_name = make_candidate(biosample_term_name="hepatocyte", ontology_id="CL:9999999")
    score_id, reason_id = frr.score_biosample_match(query, exact_id)
    score_name, _ = frr.score_biosample_match(query, exact_name)
    assert score_id > score_name
    assert "ontology_id exact" in reason_id


def test_score_curated_synonym_for_example_alias():
    candidate = make_candidate(biosample_term_name="hepatocyte", ontology_id="CL:0000182")
    score, reason = frr.score_biosample_match(example_query(), candidate)
    assert score >= 80.0
    assert "curated synonym" in reason


def test_curated_synonym_without_tier_rules_scores_flat_80(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('{"synonyms": {"my line": [["hepatocyte", "CL:0000182"]]}}')
    query = frr.parse_query("my line", None, frr.load_matching_rules(str(path)))
    score, reason = frr.score_biosample_match(query, make_candidate(biosample_term_name="hepatocyte"))
    assert int(score) == 80
    assert reason.startswith("curated synonym -> hepatocyte")


# ---------------------------------------------------------------------------
# Curated sub-tiers from the example file: hepatocyte > liver progenitor >
# generic epithelial cell > liver tissue > no match.
# ---------------------------------------------------------------------------

def test_tier_rules_order_candidates_by_specificity():
    query = example_query()
    kinds = frr.RESOURCE_MOTIF_INSTANCES
    candidates = {
        "exact": make_candidate(resource_type=kinds, biosample_term_name="hepatocyte"),
        "progenitor": make_candidate(resource_type=kinds, biosample_term_name="hepatoblast"),
        "generic": make_candidate(resource_type=kinds, biosample_term_name="epithelial cell"),
        "tissue": make_candidate(resource_type=kinds, biosample_term_name="right lobe of liver", is_tissue=True),
    }
    scored = {label: frr.score_biosample_match(query, c)[0] for label, c in candidates.items()}
    ordered = sorted(scored, key=lambda label: scored[label], reverse=True)
    assert ordered == ["exact", "progenitor", "generic", "tissue"], scored


def test_tier_rules_use_cell_slims_when_term_name_is_uninformative():
    candidate = make_candidate(
        resource_type=frr.RESOURCE_MOTIF_INSTANCES, biosample_term_name="unusual biosample name",
        cell_slims=("hepatocyte",), organ_slims=("liver",),
    )
    score, reason = frr.score_biosample_match(example_query(), candidate)
    assert score >= 80.0, f"Expected a curated tier via cell_slims, got {score} ({reason})"


def test_tissue_rule_does_not_match_other_cell_type_named_after_tissue():
    """Regression: a different cell type whose name mentions the tissue (a fibroblast line from
    liver) must not be mistaken for a tissue sample -- it scores 0 (no match)."""
    fibroblast = make_candidate(
        resource_type=frr.RESOURCE_MOTIF_INSTANCES, biosample_term_name="fibroblast of liver",
        cell_slims=("fibroblast",), organ_slims=("liver",), is_tissue=False,
    )
    score, reason = frr.score_biosample_match(example_query(), fibroblast)
    assert score == 0.0, f"Expected 0 (no match), got {score} ({reason})"


def test_score_ancestry_match():
    candidate = make_candidate(
        biosample_term_name="unrelated specific subtype",
        ontology_id="CL:9999999",
        ancestor_terms=("liver",),
    )
    score, reason = frr.score_biosample_match(example_query(), candidate)
    assert 50.0 <= score < 80.0
    assert "ancestry" in reason


def test_score_substring_match_lowest_nonzero_tier():
    query = frr.parse_query("hepatocyte progenitor cell", None)
    candidate = make_candidate(biosample_term_name="human hepatocyte progenitor cell primary", ontology_id="")
    score, reason = frr.score_biosample_match(query, candidate)
    assert 20.0 <= score < 50.0
    assert "substring" in reason


def test_score_no_match_is_zero():
    candidate = make_candidate(biosample_term_name="keratinocyte", ontology_id="CL:0000312")
    score, reason = frr.score_biosample_match(example_query(), candidate)
    assert score == 0.0
    assert reason == "no match"


def test_tie_break_prefers_cell_line_over_tissue():
    query = frr.parse_query("hepatocyte", "CL:0000182")
    cell_line = make_candidate(biosample_term_name="hepatocyte", ontology_id="CL:0000182", is_tissue=False)
    tissue = make_candidate(biosample_term_name="hepatocyte", ontology_id="CL:0000182", is_tissue=True)
    score_cell_line, _ = frr.score_biosample_match(query, cell_line)
    score_tissue, _ = frr.score_biosample_match(query, tissue)
    assert score_cell_line > score_tissue
    # Tie-break bonus must never cross a tier boundary.
    assert int(score_cell_line) == int(score_tissue) == 100


def test_score_e2g_prefers_bedpe_over_tsv_sidecar_at_same_score():
    """A bedpe (the actual links) and a same-dataset tsv (a QC/summary sidecar) both pass the
    file-type filter and share the same biosample -- the bedpe must rank first."""
    query = frr.parse_query("hepatocyte", "CL:0000182")
    bedpe = make_candidate(
        resource_type=frr.RESOURCE_E2G, file_format="bedpe",
        biosample_term_name="hepatocyte", ontology_id="CL:0000182",
    )
    tsv = make_candidate(
        resource_type=frr.RESOURCE_E2G, file_format="tsv",
        biosample_term_name="hepatocyte", ontology_id="CL:0000182",
    )
    score_bedpe, _ = frr.score_biosample_match(query, bedpe)
    score_tsv, _ = frr.score_biosample_match(query, tsv)
    assert score_bedpe > score_tsv
    assert int(score_bedpe) == int(score_tsv) == 100, "bonus must not cross a scoring tier"


def test_rank_candidates_sorts_descending_and_drops_zero_score():
    good = make_candidate(biosample_term_name="hepatocyte", ontology_id="CL:0000182")
    bad = make_candidate(biosample_term_name="keratinocyte", ontology_id="CL:0000312")
    ranked = frr.rank_candidates(example_query(), [bad, good])
    assert ranked == [good]


def test_tf_chip_bpnet_and_other_cell_type_do_not_outrank_chrombpnet():
    """Neither a single-TF ChIP BPNet model for the same biosample nor a ChromBPNet model of a
    different cell type named after the tissue may outrank the matching ChromBPNet candidate."""
    chrombpnet = make_candidate(
        resource_type=frr.RESOURCE_MOTIF_INSTANCES, dataset_accession="ENCSR000AAB",
        biosample_term_name="hepatocyte", is_tf_chip_bpnet=False,
    )
    ctcf_chip = make_candidate(
        resource_type=frr.RESOURCE_MOTIF_INSTANCES, dataset_accession="ENCSR000AAC",
        biosample_term_name="hepatocyte", is_tf_chip_bpnet=True,
    )
    fibroblast = make_candidate(
        resource_type=frr.RESOURCE_MOTIF_INSTANCES, dataset_accession="ENCSR000AAD",
        biosample_term_name="fibroblast of liver", cell_slims=("fibroblast",), organ_slims=("liver",),
    )
    ranked = [c.dataset_accession for c in frr.rank_candidates(example_query(), [ctcf_chip, fibroblast, chrombpnet])]
    assert ranked[0] == "ENCSR000AAB", ranked
    assert "ENCSR000AAD" not in ranked, ranked


# ---------------------------------------------------------------------------
# Portal parsing (mocked JSON, no network)
# ---------------------------------------------------------------------------

class FakeHttp:
    """Stand-in for CachedHttp: returns canned responses keyed by exact URL match on substring."""

    def __init__(self, responses: dict):
        self.responses = responses

    def get_json(self, url, body=None, headers=None):
        for key, value in self.responses.items():
            if key in url:
                return value
        return None


IGVF_E2G_SEARCH_RESPONSE = {
    "@graph": [
        {
            "accession": "IGVFDS0000AAA",
            "summary": "element-gene links prediction using scE2G v1.2 in virtual Homo sapiens hepatocyte",
            "file_set_type": "element-gene links",
            "assembly": "GRCh38",
            "samples": [
                {"sample_terms": [{"term_name": "hepatocyte", "@id": "/sample-terms/CL_0000182/"}]}
            ],
            "files": [
                {
                    "accession": "IGVFFI0000AAA", "file_format": "bedpe", "output_type": "element gene links",
                    "file_size": 12345, "href": "/prediction-sets/IGVFDS0000AAA/@@download/IGVFFI0000AAA.bedpe.gz",
                    "assembly": "GRCh38",
                }
            ],
        }
    ]
}


def test_igvf_candidates_for_e2g_parses_biosample_and_file():
    http = FakeHttp({"element-gene+links": IGVF_E2G_SEARCH_RESPONSE})
    candidates = frr.igvf_candidates_for_e2g(http)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.dataset_accession == "IGVFDS0000AAA"
    assert candidate.file_accession == "IGVFFI0000AAA"
    assert candidate.biosample_term_name == "hepatocyte"
    assert candidate.ontology_id == "CL:0000182"
    assert candidate.download_url.endswith("IGVFFI0000AAA.bedpe.gz")


ENCODE_E2G_SEARCH_RESPONSE = {
    "@graph": [
        {
            "accession": "ENCSR270IBQ",
            "biosample_ontology": {
                "term_name": "HepG2", "term_id": "EFO:0001187", "classification": "cell line",
                "organ_slims": ["liver"], "cell_slims": ["epithelial cell"], "system_slims": [],
            },
            "date_released": "2023-01-01",
        }
    ]
}

ENCODE_E2G_OBJECT_RESPONSE = {
    "files": [
        {
            "accession": "ENCFF671UPB", "file_format": "bed", "output_type": "thresholded element gene links",
            "file_size": 999, "href": "/files/ENCFF671UPB/@@download/ENCFF671UPB.bed.gz", "assembly": "GRCh38",
        },
        {
            "accession": "ENCFFIGNOREME", "file_format": "bed", "output_type": "element gene links",
            "file_size": 111, "href": "/files/ENCFFIGNOREME/@@download/ENCFFIGNOREME.bed.gz", "assembly": "hg19",
        },
    ]
}


def test_encode_candidates_for_e2g_keeps_thresholded_file_only(tmp_path, monkeypatch):
    http = FakeHttp({"ENCSR270IBQ/?format=json": ENCODE_E2G_OBJECT_RESPONSE})
    monkeypatch.setattr(
        frr, "encode_search_get", lambda url, cache_dir: ENCODE_E2G_SEARCH_RESPONSE
    )
    query = frr.parse_query("HepG2", None)
    candidates = frr.encode_candidates_for_e2g(http, query, tmp_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.file_accession == "ENCFF671UPB"
    assert candidate.biosample_term_name == "HepG2"
    assert candidate.ontology_id == "EFO:0001187"
    assert candidate.is_tissue is False


ENCODE_MOTIF_SEARCH_RESPONSE = {
    "@graph": [
        {
            "accession": "ENCSR646VRX",
            "biosample_ontology": {
                "term_name": "HepG2", "term_id": "EFO:0001187", "classification": "cell line",
                "organ_slims": [], "cell_slims": [], "system_slims": [],
            },
            "date_released": "2024-01-01",
            "description": (
                "ChromBPNet models trained on DNase-seq data in HepG2 "
                "(ENCSR646VRX)"
            ),
        }
    ]
}

ENCODE_MOTIF_OBJECT_RESPONSE = {
    "files": [
        {"accession": "ENCFF679ALA", "file_format": "tar", "output_type": "sequence motifs instances",
         "file_size": 1, "href": "/files/ENCFF679ALA/@@download/ENCFF679ALA.tar.gz", "assembly": "GRCh38"},
        {"accession": "ENCFF174VNP", "file_format": "bigBed", "output_type": "sequence motifs instances",
         "file_size": 2, "href": "/files/ENCFF174VNP/@@download/ENCFF174VNP.bigBed", "assembly": "GRCh38"},
        {"accession": "ENCFF452XTP", "file_format": "tar", "output_type": "sequence motifs",
         "file_size": 3, "href": "/files/ENCFF452XTP/@@download/ENCFF452XTP.tar.gz", "assembly": "GRCh38"},
        {"accession": "ENCFF220FDD", "file_format": "tar", "output_type": "sequence motifs report",
         "file_size": 4, "href": "/files/ENCFF220FDD/@@download/ENCFF220FDD.tar.gz", "assembly": "GRCh38"},
    ]
}


def test_encode_candidates_for_motifs_splits_by_output_type(tmp_path, monkeypatch):
    http = FakeHttp({"ENCSR646VRX/?format=json": ENCODE_MOTIF_OBJECT_RESPONSE})
    monkeypatch.setattr(
        frr, "encode_search_get", lambda url, cache_dir: ENCODE_MOTIF_SEARCH_RESPONSE
    )
    query = frr.parse_query("HepG2", None)
    candidates = frr.encode_candidates_for_motifs(http, query, tmp_path)
    by_type = {c.resource_type for c in candidates}
    assert by_type == {frr.RESOURCE_MOTIF_INSTANCES, frr.RESOURCE_MOTIF_ANNOTATION, frr.RESOURCE_MOTIF_REPORT}
    instances = [c for c in candidates if c.resource_type == frr.RESOURCE_MOTIF_INSTANCES]
    assert {c.file_accession for c in instances} == {"ENCFF679ALA", "ENCFF174VNP"}
    assert all(c.is_tf_chip_bpnet is False for c in candidates), "ChromBPNet-model candidates must not be flagged as TF-ChIP BPNet"
    assert all(c.model_input_assay == "DNase-seq" for c in candidates), (
        f"Expected model_input_assay parsed from description, got {[c.model_input_assay for c in candidates]}"
    )
    assert all(c.source_experiment == "ENCSR646VRX" for c in candidates), (
        f"Expected source_experiment parsed from description, got {[c.source_experiment for c in candidates]}"
    )


def test_encode_candidates_for_motifs_searches_chrombpnet_only_by_default(tmp_path, monkeypatch):
    seen_urls = []

    def fake_search_get(url, cache_dir):
        seen_urls.append(url)
        return {"@graph": []}

    monkeypatch.setattr(frr, "encode_search_get", fake_search_get)
    query = frr.parse_query("HepG2", None)
    frr.encode_candidates_for_motifs(FakeHttp({}), query, tmp_path)
    assert any("annotation_type=ChromBPNet-model" in u for u in seen_urls), (
        f"Expected a ChromBPNet-model search, got {seen_urls}"
    )
    assert not any("annotation_type=BPNet-model" in u for u in seen_urls), (
        f"BPNet-model (single-TF ChIP) must not be searched by default, got {seen_urls}"
    )


def test_encode_candidates_for_motifs_includes_tf_chip_bpnet_when_flagged(tmp_path, monkeypatch):
    seen_urls = []

    def fake_search_get(url, cache_dir):
        seen_urls.append(url)
        return {"@graph": []}

    monkeypatch.setattr(frr, "encode_search_get", fake_search_get)
    query = frr.parse_query("HepG2", None)
    frr.encode_candidates_for_motifs(FakeHttp({}), query, tmp_path, include_tf_chip_bpnet=True)
    assert any("annotation_type=BPNet-model" in u for u in seen_urls), (
        f"Expected a BPNet-model search when include_tf_chip_bpnet=True, got {seen_urls}"
    )


# ---------------------------------------------------------------------------
# parse_chrombpnet_description
# ---------------------------------------------------------------------------

def test_parse_chrombpnet_description_extracts_assay_and_experiment():
    assay, experiment = frr.parse_chrombpnet_description(
        "ChromBPNet models trained on DNase-seq data in hepatocyte "
        "(ENCSR000EOG)"
    )
    assert assay == "DNase-seq"
    assert experiment == "ENCSR000EOG"


def test_parse_chrombpnet_description_handles_atac_seq_and_missing_accession():
    assay, experiment = frr.parse_chrombpnet_description(
        "ChromBPNet models trained on ATAC-seq data in HepG2"
    )
    assert assay == "ATAC-seq"
    assert experiment == ""


def test_parse_chrombpnet_description_empty_input():
    assay, experiment = frr.parse_chrombpnet_description("")
    assert assay == ""
    assert experiment == ""


# ---------------------------------------------------------------------------
# encode_search_get: 404-as-empty-result, cached, no retry storm
# ---------------------------------------------------------------------------

def test_encode_search_get_treats_404_as_empty_without_retry(tmp_path, monkeypatch):
    import urllib.error

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(frr.urllib.request, "urlopen", fake_urlopen)
    result = frr.encode_search_get("https://example.org/search/?x=y", tmp_path)
    assert result == {"@graph": []}
    assert len(calls) == 1, f"Expected exactly one attempt for a 404, got {len(calls)} — a 404 is a real empty result, not a transient failure to retry"


def test_encode_search_get_caches_to_disk(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def read(self):
            return b'{"@graph": [{"accession": "X"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(frr.urllib.request, "urlopen", fake_urlopen)
    url = "https://example.org/search/?x=y"
    first = frr.encode_search_get(url, tmp_path)
    second = frr.encode_search_get(url, tmp_path)
    assert first == second == {"@graph": [{"accession": "X"}]}
    assert len(calls) == 1, f"Expected the second call to hit the on-disk cache, got {len(calls)} network calls"


def test_encode_search_get_does_not_cache_transient_failure(tmp_path, monkeypatch):
    """A network blip must not be remembered forever as "no results" -- only a real success or a
    confirmed 404 may be written to the cache."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        raise TimeoutError("simulated network timeout")

    monkeypatch.setattr(frr.urllib.request, "urlopen", fake_urlopen)
    url = "https://example.org/search/?x=y"
    result = frr.encode_search_get(url, tmp_path)
    assert result == {"@graph": []}
    assert len(calls) == 2, f"Expected the bounded retry (2 attempts) to run, got {len(calls)}"
    cache_path = frr.encode_search_cache_path(tmp_path)
    assert not cache_path.exists(), "Transient failure must not be persisted to the on-disk cache"


# ---------------------------------------------------------------------------
# Manifest / top suggestions
# ---------------------------------------------------------------------------

def test_top_suggestion_per_type_picks_rank_one():
    manifest = pd.DataFrame([
        {**{c: "" for c in frr.MANIFEST_COLUMNS}, "resource_type": frr.RESOURCE_E2G, "rank": 1,
         "portal": "ENCODE", "dataset_accession": "A", "file_accession": "F1", "match_score": 90.0},
        {**{c: "" for c in frr.MANIFEST_COLUMNS}, "resource_type": frr.RESOURCE_E2G, "rank": 2,
         "portal": "ENCODE", "dataset_accession": "A", "file_accession": "F2", "match_score": 20.0},
    ])
    top = frr.top_suggestion_per_type(manifest)
    assert top[frr.RESOURCE_E2G]["file_accession"] == "F1"


def test_main_writes_manifest_with_user_override(tmp_path, monkeypatch):
    def fake_build_manifest(query, http, cache_dir, top_n=10, include_tf_chip_bpnet=False):
        return pd.DataFrame(columns=frr.MANIFEST_COLUMNS)

    monkeypatch.setattr(frr, "build_manifest", fake_build_manifest)
    monkeypatch.setattr(frr, "build_cached_http_client", lambda cache_dir: FakeHttp({}))

    exit_code = frr.main([
        "--cell-type", "HepG2",
        "--out-dir", str(tmp_path),
        "--e2g-links", "/path/to/links.bedpe",
    ])
    assert exit_code == 0
    manifest = pd.read_csv(tmp_path / "regulatory_resources_manifest.tsv", sep="\t")
    override_row = manifest[manifest["resource_type"] == frr.RESOURCE_E2G].iloc[0]
    assert override_row["portal"] == "user"
    assert override_row["download_url"] == "/path/to/links.bedpe"
