"""Find enhancer-gene link and motif-call files on the IGVF and ENCODE portals for a cell type.

Given a free-text cell-type description and/or ontology ID (e.g. ``"<cell type>"``,
``hepatocyte``, ``CL:0000182``), queries both portals for:

  - enhancer-gene links   (ENCODE-rE2G, ABC, IGVF scE2G)
  - motif instance calls  (ChromBPNet / TF-MoDISco / Fi-NeMo hit calls; "sequence motifs instances")
  - motif annotation      ("sequence motifs", "sequence motifs report" -- pattern/TF annotation)

and ranks the candidate files by how well the file's biosample matches the requested cell type.
Writes a long manifest TSV (one row per candidate file) and prints the top-ranked file per
resource type. A user-supplied path (``--e2g-links`` / ``--motif-hits`` / ``--motif-annotation``)
bypasses the search entirely for that resource type.

Ranking (``score_biosample_match``), documented so the numbers can be defended: an exact
ontology-ID match to the query (100) beats an exact term-name match (90), which beats a curated
synonym (80), which beats an ontology-ancestry match (50; query maps, via synonym, to a term that
is an ancestor/slim of the candidate, or vice versa), which beats a plain substring match (20).
Ties within the same tier are broken by preferring a cell line/primary cell biosample over a
tissue, GRCh38 over other assemblies, an untreated biosample over a treated one, and a more
recent release date. A tie-break penalty keeps a single-TF ChIP-seq BPNet-model candidate from
outranking a ChromBPNet-model candidate at the same tier (see
``ENCODE_MOTIF_TF_CHIP_BPNET_ANNOTATION_TYPE``).

Curated synonyms and sub-tiers are data, not code: pass ``--synonyms <file.json>`` (format and a
worked example in ``resources/cell_type_synonyms.example.json``). Without it no curated synonyms
exist, so ranking uses only the ontology-ID, term-name, ancestry and substring tiers. The file's
optional ``tier_rules`` split the flat curated-synonym tier (e.g. the exact cell type > a related
progenitor > a generic parent cell type), matched against the candidate's own term_name first and
then its ``cell_slims``, so a cell-type slim always outranks a mere organ_slims/tissue-name
mention; ``tissue_rule`` adds a lower tier for tissue biosamples only (see ``curated_tier_score``).

Only ``requests``/``urllib`` are used for HTTP; JSON responses are cached to disk via
``annotator_core.http_cache.CachedHttp`` (or a bundled fallback, see ``build_cached_http_client``) so a
rerun for the same cell type is free.

CLI:
    python find_regulatory_resources.py --cell-type "<cell type>" --out-dir OUT_DIR
    python find_regulatory_resources.py --cell-type "<cell type>" --ontology-id CL:0000182 \\
        --synonyms resources/cell_type_synonyms.example.json --out-dir OUT_DIR --top-n 5
    python find_regulatory_resources.py --cell-type "<cell type>" --e2g-links /path/to/my_links.bedpe \\
        --out-dir OUT_DIR   # bypasses the E2G search; motif search still runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

IGVF_PORTAL_BASE = "https://api.data.igvf.org"
ENCODE_PORTAL_BASE = "https://www.encodeproject.org"

RESOURCE_E2G = "e2g_links"
RESOURCE_MOTIF_INSTANCES = "motif_instances"
RESOURCE_MOTIF_ANNOTATION = "motif_annotation"
RESOURCE_MOTIF_REPORT = "motif_report"
ALL_RESOURCE_TYPES = (
    RESOURCE_E2G, RESOURCE_MOTIF_INSTANCES, RESOURCE_MOTIF_ANNOTATION, RESOURCE_MOTIF_REPORT,
)

MANIFEST_COLUMNS = [
    "rank", "portal", "resource_type", "dataset_accession", "file_accession",
    "biosample_term_name", "ontology_id", "match_reason", "match_score",
    "assembly", "file_format", "output_type", "model_input_assay", "source_experiment",
    "size", "download_url", "local_path",
]

# ---------------------------------------------------------------------------
# Curated cell-type matching rules (loaded from --synonyms; none by default)
# ---------------------------------------------------------------------------

@dataclass
class MatchingRules:
    """Curated synonyms and curated-tier sub-rules, loaded by :func:`load_matching_rules`.

    ``synonyms``: query alias (lowercased, matched as substring of the user's --cell-type) ->
    canonical (term_name, ontology_id) pairs the portals actually use.
    ``tier_rules``: (compiled pattern, score, label), tried in order inside the curated band.
    ``tissue_rule``: (compiled pattern, score, label) for tissue biosamples only, or None.
    """
    synonyms: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    tier_rules: list[tuple[re.Pattern, float, str]] = field(default_factory=list)
    tissue_rule: Optional[tuple[re.Pattern, float, str]] = None


def load_matching_rules(path: Optional[str]) -> MatchingRules:
    """Read a ``--synonyms`` JSON file (see ``resources/cell_type_synonyms.example.json``).
    ``None``/empty path -> no curated rules (neutral ranking)."""
    if not path:
        return MatchingRules()
    data = json.loads(Path(path).read_text())
    synonyms = {
        normalize(alias): [(str(term), str(ontology or "")) for term, ontology in pairs]
        for alias, pairs in (data.get("synonyms") or {}).items()
    }
    tier_rules = [
        (re.compile(rule["pattern"]), float(rule["score"]), str(rule.get("label", rule["pattern"])))
        for rule in data.get("tier_rules") or []
    ]
    tissue = data.get("tissue_rule")
    tissue_rule = (
        (re.compile(tissue["pattern"]), float(tissue["score"]), str(tissue.get("label", tissue["pattern"])))
        if tissue else None
    )
    for _, score, label in tier_rules + ([tissue_rule] if tissue_rule else []):
        if not 50.0 < score < 90.0:
            raise ValueError(f"{path}: tier score for {label!r} must be between 50 and 90, got {score}")
    return MatchingRules(synonyms=synonyms, tier_rules=tier_rules, tissue_rule=tissue_rule)


def normalize(text: str) -> str:
    return re.sub(r"[\s_-]+", " ", text.strip().lower())


def normalize_ontology_id(text: str) -> str:
    """``CL:1000413``, ``CL_1000413``, and ``/sample-terms/CL_1000413/`` all normalize to ``CL:1000413``."""
    match = re.search(r"([A-Za-z]+)[:_](\d+)", text)
    if not match:
        return normalize(text)
    return f"{match.group(1).upper()}:{match.group(2)}"


@dataclass
class CellTypeQuery:
    raw_text: str
    ontology_ids: list[str] = field(default_factory=list)
    rules: MatchingRules = field(default_factory=MatchingRules)

    @property
    def text_norm(self) -> str:
        return normalize(self.raw_text)

    def curated_terms(self) -> list[tuple[str, str]]:
        """Canonical (term_name, ontology_id) pairs from the curated synonyms matching this query."""
        terms = []
        for alias, canonical in self.rules.synonyms.items():
            if alias in self.text_norm:
                terms.extend(canonical)
        return terms


def parse_query(
    cell_type_text: Optional[str], ontology_id: Optional[str], rules: Optional[MatchingRules] = None
) -> CellTypeQuery:
    if not cell_type_text and not ontology_id:
        raise ValueError("Provide --cell-type and/or --ontology-id.")
    query = CellTypeQuery(raw_text=cell_type_text or "", rules=rules or MatchingRules())
    if ontology_id:
        query.ontology_ids.append(normalize_ontology_id(ontology_id))
    return query


# ---------------------------------------------------------------------------
# HTTP client (cached)
# ---------------------------------------------------------------------------

def build_cached_http_client(cache_dir: Path):
    """Reuse annotator_core's disk-cached HTTP client; fall back to a minimal local copy.

    The shared client lives under Stage3_Interpretation/C_Annotation/annotator_core, four levels
    up from this file's ``src`` directory. Importing it directly (rather than duplicating the
    caching logic) keeps cache format and retry behavior consistent across the pipeline.
    """
    pipeline_src = Path(__file__).resolve().parents[3]
    annotator_core = pipeline_src / "Stage3_Interpretation" / "C_Annotation" / "annotator_core"
    if annotator_core.is_dir() and str(annotator_core) not in sys.path:
        sys.path.insert(0, str(annotator_core))
    try:
        from http_cache import CachedHttp  # type: ignore
        return CachedHttp(cache_dir)
    except ImportError:
        return FallbackCachedHttp(cache_dir)


class FallbackCachedHttp:
    """Minimal disk-cached GET, used only if annotator_core.http_cache is unavailable."""

    def __init__(self, cache_dir: Path, pause: float = 0.2):
        import json
        self.json = json
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.cache_file = cache_dir / "http_cache_fallback.json"
        self.cache = (
            self.json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        )
        self.dirty = 0

    def get_json(self, url: str, body=None, headers=None):
        import time
        import urllib.request

        if url in self.cache:
            return self.cache[url]
        request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
        result = None
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = self.json.loads(response.read().decode() or "null")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    print(f"  giving up on {url[:100]} ({exc})")
                time.sleep(2 * attempt)
        time.sleep(self.pause)
        if result is None:
            return None
        self.cache[url] = result
        self.dirty += 1
        if self.dirty % 25 == 0:
            self.save()
        return result

    def save(self):
        self.cache_file.write_text(self.json.dumps(self.cache))


# ---------------------------------------------------------------------------
# Candidate file record
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    portal: str  # "IGVF" | "ENCODE"
    resource_type: str
    dataset_accession: str
    file_accession: str
    biosample_term_name: str
    ontology_id: str
    assembly: str
    file_format: str
    output_type: str
    size: Optional[int]
    download_url: str
    is_tissue: bool = False
    is_treated: bool = False
    released_date: str = ""
    ancestor_terms: tuple[str, ...] = ()  # term names/ids from organ/cell/system slims or ancestors
    cell_slims: tuple[str, ...] = ()
    organ_slims: tuple[str, ...] = ()
    # ChromBPNet/BPNet FileSets only encode these two facts as free text in ``description`` (no
    # structured field exists for either) -- see ``parse_chrombpnet_description``.
    model_input_assay: str = ""  # e.g. "DNase-seq" or "ATAC-seq"
    source_experiment: str = ""  # ENCODE experiment accession the model was trained on
    is_tf_chip_bpnet: bool = False  # single-TF ChIP-seq BPNet model, e.g. CTCF -- see module docstring
    match_score: float = 0.0
    match_reason: str = ""

    def as_row(self, rank: int, local_path: str = "") -> dict:
        return {
            "rank": rank,
            "portal": self.portal,
            "resource_type": self.resource_type,
            "dataset_accession": self.dataset_accession,
            "file_accession": self.file_accession,
            "biosample_term_name": self.biosample_term_name,
            "ontology_id": self.ontology_id,
            "match_reason": self.match_reason,
            "match_score": self.match_score,
            "assembly": self.assembly,
            "file_format": self.file_format,
            "output_type": self.output_type,
            "model_input_assay": self.model_input_assay,
            "source_experiment": self.source_experiment,
            "size": self.size if self.size is not None else "",
            "download_url": self.download_url,
            "local_path": local_path,
        }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def curated_tier_score(query: CellTypeQuery, candidate: Candidate) -> Optional[tuple[float, str]]:
    """Fine-grained ranking within the "curated synonym" band from the ``tier_rules`` /
    ``tissue_rule`` of the ``--synonyms`` file (see the module docstring). Runs only when the
    query itself matched a curated alias; otherwise returns ``None`` so the caller falls through
    to the flat curated/ancestry/substring tiers. Scores in the example file are spaced 3 points
    apart -- wider than the <=1.0 tie-break bonus, so a bonus never crosses a sub-tier.

    The tissue rule requires ``candidate.is_tissue`` so that a different cell type whose name
    happens to mention the tissue (e.g. "fibroblast of <organ>") is never mistaken for a tissue
    sample -- it correctly falls through to "no match" instead.
    """
    if not query.curated_terms():
        return None
    haystacks = [normalize(candidate.biosample_term_name), normalize(" ".join(candidate.cell_slims))]
    for haystack in haystacks:
        for pattern, score, label in query.rules.tier_rules:
            if pattern.search(haystack):
                return score, f"curated synonym -> tier: {label} ({candidate.biosample_term_name!r})"
    tissue_rule = query.rules.tissue_rule
    if tissue_rule and candidate.is_tissue and tissue_rule[0].search(haystacks[0]):
        return tissue_rule[1], f"curated synonym -> tier: {tissue_rule[2]} ({candidate.biosample_term_name!r})"
    return None


def score_biosample_match(query: CellTypeQuery, candidate: Candidate) -> tuple[float, str]:
    """Score how well ``candidate``'s biosample matches ``query``. Higher is better.

    Tiers (documented in the module docstring): ontology-ID exact (100) > term-name exact (90) >
    curated synonym (80, or a ``tier_rules`` score -- see ``curated_tier_score``) > ontology
    ancestry (50) > substring (20) > no match (0). A tie-break bonus (< 1 point total, so it
    never crosses a tier) prefers cell line/primary cell over tissue, GRCh38, untreated, a more
    recent release, a ChromBPNet-model over a single-TF ChIP BPNet-model at the same tier, and --
    for E2G links -- a bedpe/bed link-coordinate file over a same-dataset tsv sidecar (QC tables,
    gene lists, etc. that pass the file-type filter but are not the links themselves).
    """
    candidate_term_norm = normalize(candidate.biosample_term_name)
    candidate_ontology_norm = normalize_ontology_id(candidate.ontology_id) if candidate.ontology_id else ""

    score, reason = 0.0, "no match"

    if query.ontology_ids and candidate_ontology_norm and candidate_ontology_norm in query.ontology_ids:
        score, reason = 100.0, f"ontology_id exact ({candidate_ontology_norm})"
    elif query.text_norm and candidate_term_norm and query.text_norm == candidate_term_norm:
        score, reason = 90.0, f"term_name exact ({candidate.biosample_term_name!r})"
    else:
        curated_tier = curated_tier_score(query, candidate)
        if curated_tier is not None:
            score, reason = curated_tier
        else:
            for term_name, ontology_id in query.curated_terms():
                # An empty ``ontology_id`` (some curated synonyms have none) must never be
                # compared -- otherwise it spuriously matches any candidate whose own
                # ontology_id is also unset, e.g. "" == "".
                if (ontology_id and normalize_ontology_id(ontology_id) == candidate_ontology_norm) or (
                    term_name and normalize(term_name) == candidate_term_norm
                ):
                    score, reason = 80.0, f"curated synonym -> {term_name} ({ontology_id})"
                    break
        if score == 0.0:
            curated_norms = {normalize(t) for t, _ in query.curated_terms() if t} | {
                normalize_ontology_id(o) for _, o in query.curated_terms() if o
            }
            ancestor_norms = {normalize(a) for a in candidate.ancestor_terms if a} | {
                normalize_ontology_id(a) for a in candidate.ancestor_terms if a
            }
            hit = (curated_norms | {query.text_norm} | set(query.ontology_ids)) & ancestor_norms
            if hit:
                score, reason = 50.0, f"ontology ancestry match ({sorted(hit)[0]})"
            elif query.text_norm and candidate_term_norm and (
                query.text_norm in candidate_term_norm or candidate_term_norm in query.text_norm
            ):
                score, reason = 20.0, f"substring match ({candidate.biosample_term_name!r})"

    if score > 0.0:
        bonus = 0.0
        if not candidate.is_tissue:
            bonus += 0.4
        if candidate.assembly == "GRCh38":
            bonus += 0.3
        if not candidate.is_treated:
            bonus += 0.2
        if candidate.released_date:
            # Newer release -> slightly higher; string-sortable ISO dates, capped contribution.
            bonus += min(0.1, int(candidate.released_date[:4] or 0) / 100000.0)
        if candidate.resource_type == RESOURCE_E2G and candidate.file_format in ("bedpe", "bed"):
            # A biosample match ties across every file in the same dataset (QC/summary tsvs
            # included); prefer the actual link-coordinate file over a same-dataset tsv sidecar.
            bonus += 0.05
        if candidate.is_tf_chip_bpnet:
            # Single-TF ChIP BPNet-model (e.g. CTCF) must never outrank a ChromBPNet-model
            # candidate at the same biosample-match tier -- the tiers are spaced >=3 points apart
            # so this can only affect ties within a tier, never cross one.
            bonus -= 0.5
        score += bonus
    return score, reason


def rank_candidates(query: CellTypeQuery, candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        candidate.match_score, candidate.match_reason = score_biosample_match(query, candidate)
    scored = [c for c in candidates if c.match_score > 0.0]
    scored.sort(key=lambda c: c.match_score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# IGVF portal
# ---------------------------------------------------------------------------

IGVF_E2G_FIELDS = [
    "accession", "@id", "summary", "file_set_type", "lab.title", "status",
    "samples.sample_terms.term_name", "samples.sample_terms.@id",
    "samples.classifications", "samples.disease_terms",
    "assembly", "files.accession", "files.file_format", "files.output_type",
    "files.content_type", "files.file_size", "files.href", "files.assembly",
]


def igvf_search_url(file_set_type: str, extra_type: str = "PredictionSet") -> str:
    params = [("type", extra_type), ("file_set_type", file_set_type), ("limit", "all"), ("format", "json")]
    for f in IGVF_E2G_FIELDS:
        params.append(("field", f))
    return f"{IGVF_PORTAL_BASE}/search/?{urllib.parse.urlencode(params)}"


def igvf_candidates_for_e2g(http) -> list[Candidate]:
    """PredictionSets with file_set_type='element-gene links' (IGVF scE2G/ABC E2G predictions)."""
    data = http.get_json(igvf_search_url("element-gene links"))
    if not data:
        return []
    candidates = []
    for item in data.get("@graph", []):
        samples = item.get("samples") or []
        sample_terms = []
        for sample in samples:
            sample_terms.extend(sample.get("sample_terms") or [])
        if not sample_terms:
            # No embedded biosample -- fall back to parsing the free-text summary.
            sample_terms = [{"term_name": item.get("summary", ""), "@id": ""}]
        for term in sample_terms:
            term_name = term.get("term_name", "") or ""
            ontology_id = normalize_ontology_id(term.get("@id", "")) if term.get("@id") else ""
            for f in item.get("files", []) or []:
                if not any(kw in (f.get("output_type") or "").lower() for kw in ("link", "prediction")) and (
                    f.get("file_format") not in ("bedpe", "bed", "tsv")
                ):
                    continue
                candidates.append(Candidate(
                    portal="IGVF",
                    resource_type=RESOURCE_E2G,
                    dataset_accession=item.get("accession", ""),
                    file_accession=f.get("accession", ""),
                    biosample_term_name=term_name,
                    ontology_id=ontology_id,
                    assembly=f.get("assembly", item.get("assembly", "")) or "",
                    file_format=f.get("file_format", "") or "",
                    output_type=f.get("output_type", "") or "",
                    size=f.get("file_size"),
                    download_url=(IGVF_PORTAL_BASE + f["href"]) if f.get("href") else "",
                ))
    return candidates


def igvf_candidates_for_motifs(http) -> list[Candidate]:
    """IGVF currently has no known finemo/ChromBPNet hit-call FileSets; kept as a documented no-op
    stub so a future IGVF submission is picked up automatically once it exists (searches
    ModelSet/AnalysisSet for a 'motif' summary keyword)."""
    data = http.get_json(igvf_search_url("motif enrichment", extra_type="AnalysisSet")) or {}
    candidates = []
    for item in data.get("@graph", []):
        summary = (item.get("summary") or "").lower()
        if "motif" not in summary and "chrombpnet" not in summary and "modisco" not in summary:
            continue
        samples = item.get("samples") or []
        sample_terms = [t for s in samples for t in (s.get("sample_terms") or [])]
        for term in sample_terms or [{"term_name": summary, "@id": ""}]:
            for f in item.get("files", []) or []:
                candidates.append(Candidate(
                    portal="IGVF",
                    resource_type=RESOURCE_MOTIF_INSTANCES,
                    dataset_accession=item.get("accession", ""),
                    file_accession=f.get("accession", ""),
                    biosample_term_name=term.get("term_name", ""),
                    ontology_id=normalize_ontology_id(term.get("@id", "")) if term.get("@id") else "",
                    assembly=f.get("assembly", item.get("assembly", "")) or "",
                    file_format=f.get("file_format", "") or "",
                    output_type=f.get("output_type", "") or "",
                    size=f.get("file_size"),
                    download_url=(IGVF_PORTAL_BASE + f["href"]) if f.get("href") else "",
                ))
    return candidates


# ---------------------------------------------------------------------------
# ENCODE portal
# ---------------------------------------------------------------------------

ENCODE_BIOSAMPLE_FIELDS = [
    "biosample_ontology.term_name", "biosample_ontology.term_id",
    "biosample_ontology.classification", "biosample_ontology.organ_slims",
    "biosample_ontology.cell_slims", "biosample_ontology.system_slims",
]
# ENCODE's annotation_type facet values for motif-model FileSets (confirmed against a live
# search). ChromBPNet-model is the primary source: genome-wide DNase-seq/ATAC-seq accessibility
# models with a full multi-TF Fi-NeMo motif panel (1565 entries as of 2026-09). BPNet-model holds
# single-TF ChIP-seq models (e.g. CTCF) -- single-motif, and only searched when explicitly
# requested (``--include-tf-chip-bpnet-models``); see ``ENCODE_MOTIF_TF_CHIP_BPNET_ANNOTATION_TYPE``
# and the tie-break penalty in ``score_biosample_match`` that keeps it from outranking a
# ChromBPNet-model candidate.
ENCODE_MOTIF_CHROMBPNET_ANNOTATION_TYPE = "ChromBPNet-model"
ENCODE_MOTIF_TF_CHIP_BPNET_ANNOTATION_TYPE = "BPNet-model"
ENCODE_E2G_ANNOTATION_TYPE = "element gene regulatory interaction predictions"


def encode_search_url(
    annotation_type: str, extra_params: list[tuple[str, str]], limit: str = "10"
) -> str:
    """Search Annotation FileSets. Always pass a ``biosample_ontology.term_name`` filter in
    ``extra_params`` -- an unfiltered pull of one annotation_type is 1000s of records and prone to
    ENCODE's under-load 504s; a per-biosample-term query is small and fast (verified live)."""
    params = [("type", "Annotation"), ("annotation_type", annotation_type), ("limit", limit),
              ("format", "json"), ("field", "accession"), ("field", "assembly"),
              ("field", "status"), ("field", "date_released"), ("field", "description")]
    params += [("field", f) for f in ENCODE_BIOSAMPLE_FIELDS]
    params += extra_params
    return f"{ENCODE_PORTAL_BASE}/search/?{urllib.parse.urlencode(params)}"


def encode_search_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "encode_search_cache.json"


def encode_search_get(url: str, cache_dir: Path) -> dict:
    """GET an ENCODE search URL, cached to disk, tolerant of ENCODE's 404-for-empty-facet quirk.

    ENCODE returns HTTP 404 (not a 200 with an empty ``@graph``) when a facet filter combination
    has no matches at all -- e.g. ``biosample_ontology.term_name=<cell line alias>`` (not a real ENCODE
    biosample term) or a valid term name paired with an annotation_type that has no such record.
    Verified live: this 404 comes back immediately and is a real "no results", not a transient
    failure -- retrying it (as ``CachedHttp.get_json`` does for every exception) wastes 30-150s per
    query for nothing. Anything else (timeout, 5xx -- ENCODE does 504 under load) gets a short
    bounded retry.

    ``Accept-Encoding: identity`` is requested explicitly -- verified live: without it, urllib
    intermittently raises ``http.client.IncompleteRead`` on ENCODE's chunked+gzip responses
    (curl on the same URL never reproduces this), which would otherwise burn through the bounded
    retry above on a request that was never going to succeed compressed.
    """
    import json
    import time

    cache_path = encode_search_cache_path(cache_dir)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if url in cache:
        return cache[url]

    result: dict = {"@graph": []}
    resolved = False  # True only for a real success or a confirmed 404 -- never for exhausted
    # retries after a transient failure, which must stay uncached so the next run retries them
    # instead of a network blip being remembered forever as "no results".
    for attempt in range(1, 3):
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "Accept-Encoding": "identity"}
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                result = json.loads(response.read().decode() or "null") or {"@graph": []}
            resolved = True
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                resolved = True
                break  # no results for this facet combination -- not an error, don't retry
            if attempt == 2:
                print(f"  giving up on {url[:100]} (HTTP {exc.code})")
            time.sleep(2 * attempt)
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"  giving up on {url[:100]} ({exc})")
            time.sleep(2 * attempt)

    if resolved:
        cache[url] = result
        cache_path.write_text(json.dumps(cache))
    return result


def candidate_biosample_term_names(query: CellTypeQuery) -> list[str]:
    """Biosample term names worth querying ENCODE for: curated synonyms plus the raw query text."""
    names = [name for name, _ in query.curated_terms()]
    if query.raw_text.strip():
        names.append(query.raw_text.strip())
    seen: set[str] = set()
    unique = []
    for name in names:
        key = normalize(name)
        if key and key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def encode_biosample_from_item(
    item: dict,
) -> tuple[str, str, bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Returns (term_name, ontology_id, is_tissue, ancestor_terms, cell_slims, organ_slims).

    ``cell_slims``/``organ_slims`` are also broken out separately from the combined
    ``ancestor_terms`` because ``curated_tier_score`` needs to check them in priority order
    (cell_slims before organ_slims -- see module docstring).
    """
    ontology = item.get("biosample_ontology") or {}
    term_name = ontology.get("term_name", "") or ""
    ontology_id = ontology.get("term_id", "") or ""
    classification = (ontology.get("classification") or "").lower()
    is_tissue = classification == "tissue"
    organ_slims = tuple(ontology.get("organ_slims", []) or [])
    cell_slims = tuple(ontology.get("cell_slims", []) or [])
    system_slims = tuple(ontology.get("system_slims", []) or [])
    ancestors = organ_slims + cell_slims + system_slims
    return term_name, ontology_id, is_tissue, ancestors, cell_slims, organ_slims


def parse_chrombpnet_description(description: str) -> tuple[str, str]:
    """Parse the two facts ChromBPNet/BPNet FileSet descriptions encode as free text (ENCODE has
    no structured field for either), e.g. "ChromBPNet models trained on DNase-seq data in
    <cell type> (ENCSR000AAA)" -> ("DNase-seq", "ENCSR000AAA")."""
    assay_match = re.search(r"trained on (DNase-seq|ATAC-seq)", description or "", re.IGNORECASE)
    accession_match = re.search(r"\b(ENCSR[0-9A-Z]{6})\b", description or "")
    input_assay = assay_match.group(1) if assay_match else ""
    source_experiment = accession_match.group(1) if accession_match else ""
    return input_assay, source_experiment


def encode_fetch_files(http, accession: str) -> list[dict]:
    """Full object view (not search) -- embeds file objects with output_type/file_format/href.

    Passes ``Accept-Encoding: identity`` for the same reason as ``encode_search_get`` -- urllib
    intermittently raises ``http.client.IncompleteRead`` on ENCODE's chunked+gzip responses.
    """
    data = http.get_json(
        f"{ENCODE_PORTAL_BASE}/{accession}/?format=json", headers={"Accept-Encoding": "identity"}
    )
    files = (data or {}).get("files") or []
    return [f for f in files if isinstance(f, dict)]


def encode_candidates_for_e2g(http, query: CellTypeQuery, cache_dir: Path) -> list[Candidate]:
    candidates = []
    seen_datasets: set[str] = set()
    for term_name in candidate_biosample_term_names(query):
        url = encode_search_url(
            ENCODE_E2G_ANNOTATION_TYPE, [("biosample_ontology.term_name", term_name)]
        )
        data = encode_search_get(url, cache_dir)
        if not data:
            continue
        for item in data.get("@graph", []):
            accession = item.get("accession", "")
            if accession in seen_datasets:
                continue
            seen_datasets.add(accession)
            item_term_name, ontology_id, is_tissue, ancestors, cell_slims, organ_slims = (
                encode_biosample_from_item(item)
            )
            for f in encode_fetch_files(http, accession):
                output_type = (f.get("output_type") or "").lower()
                if "thresholded" not in output_type:
                    continue
                candidates.append(Candidate(
                    portal="ENCODE",
                    resource_type=RESOURCE_E2G,
                    dataset_accession=accession,
                    file_accession=f.get("accession", ""),
                    biosample_term_name=item_term_name,
                    ontology_id=ontology_id,
                    assembly=f.get("assembly", item.get("assembly", "")) or "",
                    file_format=f.get("file_format", "") or "",
                    output_type=f.get("output_type", "") or "",
                    size=f.get("file_size"),
                    download_url=ENCODE_PORTAL_BASE + f.get("href", "") if f.get("href") else "",
                    is_tissue=is_tissue,
                    released_date=item.get("date_released", "") or "",
                    ancestor_terms=ancestors,
                    cell_slims=cell_slims,
                    organ_slims=organ_slims,
                ))
    return candidates


ENCODE_MOTIF_OUTPUT_TO_RESOURCE_TYPE = {
    "sequence motifs instances": RESOURCE_MOTIF_INSTANCES,
    "sequence motifs": RESOURCE_MOTIF_ANNOTATION,
    "sequence motifs report": RESOURCE_MOTIF_REPORT,
}


def encode_candidates_for_motifs(
    http, query: CellTypeQuery, cache_dir: Path, include_tf_chip_bpnet: bool = False
) -> list[Candidate]:
    """ChromBPNet-model FileSets are the primary source (genome-wide accessibility models with a
    full multi-TF Fi-NeMo motif panel). BPNet-model FileSets (single-TF ChIP-seq models, e.g.
    CTCF) are searched too only when ``include_tf_chip_bpnet=True``, and are tagged
    ``is_tf_chip_bpnet=True`` so ``score_biosample_match`` can keep them from outranking a
    ChromBPNet-model candidate at the same biosample-match tier."""
    annotation_types = [ENCODE_MOTIF_CHROMBPNET_ANNOTATION_TYPE]
    if include_tf_chip_bpnet:
        annotation_types.append(ENCODE_MOTIF_TF_CHIP_BPNET_ANNOTATION_TYPE)

    candidates = []
    seen_datasets: set[str] = set()
    for term_name in candidate_biosample_term_names(query):
        for annotation_type in annotation_types:
            url = encode_search_url(
                annotation_type, [("biosample_ontology.term_name", term_name)]
            )
            data = encode_search_get(url, cache_dir)
            if not data:
                continue
            for item in data.get("@graph", []):
                accession = item.get("accession", "")
                dataset_key = (annotation_type, accession)
                if dataset_key in seen_datasets:
                    continue
                seen_datasets.add(dataset_key)
                item_term_name, ontology_id, is_tissue, ancestors, cell_slims, organ_slims = (
                    encode_biosample_from_item(item)
                )
                model_input_assay, source_experiment = parse_chrombpnet_description(
                    item.get("description", "") or ""
                )
                is_tf_chip_bpnet = annotation_type == ENCODE_MOTIF_TF_CHIP_BPNET_ANNOTATION_TYPE
                for f in encode_fetch_files(http, accession):
                    output_type = (f.get("output_type") or "").lower()
                    resource_type = ENCODE_MOTIF_OUTPUT_TO_RESOURCE_TYPE.get(output_type)
                    if resource_type is None:
                        continue
                    candidates.append(Candidate(
                        portal="ENCODE",
                        resource_type=resource_type,
                        dataset_accession=accession,
                        file_accession=f.get("accession", ""),
                        biosample_term_name=item_term_name,
                        ontology_id=ontology_id,
                        assembly=f.get("assembly", item.get("assembly", "")) or "",
                        file_format=f.get("file_format", "") or "",
                        output_type=f.get("output_type", "") or "",
                        size=f.get("file_size"),
                        download_url=ENCODE_PORTAL_BASE + f.get("href", "") if f.get("href") else "",
                        is_tissue=is_tissue,
                        released_date=item.get("date_released", "") or "",
                        ancestor_terms=ancestors,
                        cell_slims=cell_slims,
                        organ_slims=organ_slims,
                        model_input_assay=model_input_assay,
                        source_experiment=source_experiment,
                        is_tf_chip_bpnet=is_tf_chip_bpnet,
                    ))
    return candidates


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(
    query: CellTypeQuery, http, cache_dir: Path, top_n: int = 10, include_tf_chip_bpnet: bool = False
) -> "pd.DataFrame":
    import pandas as pd

    all_candidates = (
        igvf_candidates_for_e2g(http) + igvf_candidates_for_motifs(http)
        + encode_candidates_for_e2g(http, query, cache_dir)
        + encode_candidates_for_motifs(http, query, cache_dir, include_tf_chip_bpnet=include_tf_chip_bpnet)
    )
    rows = []
    for resource_type in ALL_RESOURCE_TYPES:
        ranked = rank_candidates(query, [c for c in all_candidates if c.resource_type == resource_type])
        for rank, candidate in enumerate(ranked[:top_n], start=1):
            rows.append(candidate.as_row(rank))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def top_suggestion_per_type(manifest: "pd.DataFrame") -> dict[str, dict]:
    top = {}
    for resource_type in ALL_RESOURCE_TYPES:
        subset = manifest[(manifest["resource_type"] == resource_type) & (manifest["rank"] == 1)]
        if not subset.empty:
            top[resource_type] = subset.iloc[0].to_dict()
    return top


def print_top_suggestions(manifest: "pd.DataFrame") -> None:
    top = top_suggestion_per_type(manifest)
    if not top:
        print("No candidates found for any resource type.")
        return
    for resource_type in ALL_RESOURCE_TYPES:
        row = top.get(resource_type)
        if row is None:
            print(f"[{resource_type}] no candidate found")
            continue
        print(
            f"[{resource_type}] {row['portal']} {row['dataset_accession']}/{row['file_accession']} "
            f"biosample={row['biosample_term_name']!r} ({row['ontology_id']}) "
            f"score={row['match_score']:.1f} reason={row['match_reason']} "
            f"assembly={row['assembly']} url={row['download_url']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-type", default="", help="Free-text cell type description, e.g. \"hepatocyte\"")
    parser.add_argument("--ontology-id", default="", help="Ontology ID, e.g. CL:0000182")
    parser.add_argument(
        "--synonyms", default=None,
        help=(
            "JSON file of curated synonyms + optional tier rules for the curated-synonym band "
            "(format: resources/cell_type_synonyms.example.json). Default: none (neutral ranking)."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write regulatory_resources_manifest.tsv")
    parser.add_argument("--cache-dir", default=None, help="HTTP cache dir (default: <out-dir>/.http_cache)")
    parser.add_argument("--top-n", type=int, default=10, help="Candidates to keep per resource type")
    parser.add_argument("--e2g-links", default=None, help="User-provided E2G links file; bypasses search")
    parser.add_argument("--motif-hits", default=None, help="User-provided motif-instances file; bypasses search")
    parser.add_argument("--motif-annotation", default=None, help="User-provided motif-annotation file; bypasses search")
    parser.add_argument(
        "--include-tf-chip-bpnet-models", action="store_true",
        help=(
            "Also search single-TF ChIP-seq BPNet-model FileSets (e.g. CTCF) as motif "
            "candidates; never ranked above a ChromBPNet-model candidate for the same biosample."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / ".http_cache"

    overrides = {
        RESOURCE_E2G: args.e2g_links,
        RESOURCE_MOTIF_INSTANCES: args.motif_hits,
        RESOURCE_MOTIF_ANNOTATION: args.motif_annotation,
    }

    query = parse_query(args.cell_type or None, args.ontology_id or None, load_matching_rules(args.synonyms))
    http = build_cached_http_client(cache_dir)
    manifest = build_manifest(
        query, http, cache_dir, top_n=args.top_n,
        include_tf_chip_bpnet=args.include_tf_chip_bpnet_models,
    )
    if hasattr(http, "save"):
        http.save()

    # User overrides win: prepend a synthetic rank-1 row and drop the searched candidates for
    # that resource type so the manifest reflects what will actually be used downstream.
    import pandas as pd

    override_rows = []
    for resource_type, path in overrides.items():
        if not path:
            continue
        manifest = manifest[manifest["resource_type"] != resource_type]
        override_rows.append({
            "rank": 1, "portal": "user", "resource_type": resource_type,
            "dataset_accession": "", "file_accession": "", "biosample_term_name": "",
            "ontology_id": "", "match_reason": "user override", "match_score": float("inf"),
            "assembly": "", "file_format": Path(path).suffix.lstrip("."), "output_type": "",
            "model_input_assay": "", "source_experiment": "",
            "size": "", "download_url": path, "local_path": path,
        })
    if override_rows:
        manifest = pd.concat([pd.DataFrame(override_rows, columns=MANIFEST_COLUMNS), manifest], ignore_index=True)

    manifest_path = out_dir / "regulatory_resources_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"Wrote {len(manifest)} candidate row(s) to {manifest_path}")
    print_top_suggestions(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
