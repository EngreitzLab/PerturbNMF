#!/usr/bin/env python3
"""STRING functional enrichment, PPI enrichment and high-confidence partners of a ranked gene list, and the STRING
links among the regulators (each labeled by its evidence type, with the regulators' own PPI enrichment).

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md). Also importable:
0.Prepare/prepare_programs.py and 0.Prepare/refresh_string.py call enrich_program() directly for every program.
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import request_with_retry  # noqa: E402

STRING_API = "https://version-12-0.string-db.org/api/json"
CALLER = "perturbnmf_annotation"
CURATED = ("Process", "Function", "Component", "KEGG", "RCTM", "WikiPathways")
CHANNELS = {"escore": "experiments", "dscore": "databases", "tscore": "text mining", "ascore": "co-expression",
            "nscore": "neighborhood", "fscore": "gene fusion", "pscore": "co-occurrence"}
TERM_COLUMNS = ("category", "term", "description", "fdr", "p_value", "n_genes", "n_background", "genes")
DEFAULTS = {"taxid": 9606, "categories": list(CURATED), "max_fdr": 0.05, "required_score": 700, "max_partners": 15}
MIN_CHANNEL = 0.3   # a channel is listed for an edge when its own score reaches this
INTERACTION_TYPES = ("experimental", "curated", "predicted")


def edge_channels(edge):
    """{channel: score} of one STRING network edge, channels with score >= MIN_CHANNEL only."""
    return {name: round(float(edge.get(k) or 0), 3) for k, name in CHANNELS.items()
            if float(edge.get(k) or 0) >= MIN_CHANNEL}


def interaction_type(channels):
    """experimental (an assay detected the interaction), else curated (a pathway / complex database lists it), else
    predicted (only text mining, co-expression or genomic context)."""
    if "experiments" in channels:
        return "experimental"
    if "databases" in channels:
        return "curated"
    return "predicted"


def string_post(endpoint, **data):
    """One STRING API call; STRING asks for at most one call per second."""
    time.sleep(1.0)
    resp = request_with_retry("post", f"{STRING_API}/{endpoint}", data={**data, "caller_identity": CALLER})
    return resp.json()


def map_string_ids(genes, taxid, cache_path=None, chunk=2000):
    """{gene: STRING id} (best match per name); names tried before are read from cache_path when given."""
    cache_path = Path(cache_path) if cache_path else None
    full = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    cache = full.get(str(taxid), {})
    todo = [g for g in dict.fromkeys(genes) if g not in cache]
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        hits = string_post("get_string_ids", identifiers="\r".join(part), species=taxid, limit=1, echo_query=1)
        found = {h["queryItem"]: h["stringId"] for h in hits}
        cache.update({g: found.get(g) for g in part})   # None = tried, not in STRING
    if todo and cache_path:
        full[str(taxid)] = cache
        cache_path.write_text(json.dumps(full))
    return {g: cache[g] for g in genes if cache.get(g)}


def regulator_network(regulators, ids, edges, p):
    """STRING links among the regulators (combined score >= required_score), each labeled by interaction_type, and
    the regulators' PPI enrichment against STRING's genome (as the top genes'; STRING computes no PPI enrichment
    against a background above 2000 proteins, so a spectra background is not possible)."""
    mapped = [r for r in dict.fromkeys(regulators) if r in ids]
    name_of = {ids[r]: r for r in mapped}
    links = {}
    for e in edges:
        a, b = name_of.get(e["stringId_A"]), name_of.get(e["stringId_B"])
        if a and b and a != b:
            ch = edge_channels(e)
            a, b = sorted((a, b))
            links[(a, b)] = {"a": a, "b": b, "score": round(float(e["score"]), 3), "type": interaction_type(ch),
                             "channels": list(ch), "channel_scores": ch}
    links = sorted(links.values(), key=lambda x: (-x["score"], x["a"], x["b"]))
    ppi = {}
    if len(mapped) >= 2:
        ppi = (string_post("ppi_enrichment", identifiers="\r".join(dict.fromkeys(ids[r] for r in mapped)),
                           species=p["taxid"], required_score=p["required_score"]) or [{}])[0]
    return {"n_regulators": len(list(dict.fromkeys(regulators))), "n_mapped": len(mapped),
            "unmapped": [r for r in dict.fromkeys(regulators) if r not in ids],
            "background": "genome",
            "ppi_enrichment": {k: ppi.get(k) for k in ("number_of_nodes", "number_of_edges",
                                                       "expected_number_of_edges", "average_node_degree", "p_value")}
                              | ({"error": ppi.get("message") or ppi["error"]} if ppi.get("error") else {}),
            "n_edges_by_type": {t: sum(x["type"] == t for x in links) for t in INTERACTION_TYPES},
            "edges": links}


def enrich_program(genes, ids, background_ids=None, candidates=None, params=None, regulators=None):
    """Every enriched term (FDR <= max_fdr, uncapped), the PPI enrichment of genes, each candidate's
    high-confidence partners among genes, and the links among the regulators (regulator_network). genes: ranked
    names; ids: map_string_ids output; candidates: {gene: role} whose partners are listed (default: every gene in
    genes); regulators: names (default: the candidates with role "regulator")."""
    p = {**DEFAULTS, **(params or {})}
    by_id = {}
    for g, s in ids.items():
        by_id.setdefault(s, g)
    mapped = [g for g in genes if g in ids]
    top_ids = list(dict.fromkeys(ids[g] for g in mapped))
    rank = {g: i + 1 for i, g in enumerate(genes)}
    candidates = candidates or {g: "gene" for g in genes}
    if regulators is None:
        regulators = [g for g, role in candidates.items() if role == "regulator"]

    query = {"identifiers": "\r".join(top_ids), "species": p["taxid"]}
    if background_ids:
        query["background_string_identifiers"] = "\r".join(background_ids)
    terms = []
    for t in string_post("enrichment", **query):
        if t["category"] not in p["categories"] or t["fdr"] > p["max_fdr"]:
            continue
        terms.append({"category": t["category"], "term": t["term"], "description": t["description"],
                      "fdr": t["fdr"], "p_value": t["p_value"], "n_genes": t["number_of_genes"],
                      "n_background": t["number_of_genes_in_background"],
                      "genes": [by_id.get(s, n) for s, n in zip(t["inputGenes"], t["preferredNames"])]})
    terms.sort(key=lambda t: (t["fdr"], t["category"], t["term"]))

    ppi = string_post("ppi_enrichment", identifiers="\r".join(top_ids), species=p["taxid"],
                      required_score=p["required_score"])
    ppi = ppi[0] if ppi else {}
    net_ids = list(dict.fromkeys(top_ids + [ids[g] for g in [*candidates, *regulators] if g in ids]))
    edges = string_post("network", identifiers="\r".join(net_ids), species=p["taxid"],
                        required_score=p["required_score"])
    top_set, partners = set(mapped), {}
    for e in edges:
        a, b = by_id.get(e["stringId_A"], e["preferredName_A"]), by_id.get(e["stringId_B"], e["preferredName_B"])
        channels = list(edge_channels(e))
        for x, y in ((a, b), (b, a)):
            if y in top_set and x != y:
                partners.setdefault(x, {})[y] = (round(float(e["score"]), 3), channels)
    gene_rows = {}
    for g, role in candidates.items():
        ps = sorted(partners.get(g, {}).items(), key=lambda kv: (-kv[1][0], kv[0]))
        gene_rows[g] = {"role": role, "rank": rank.get(g), "in_string": g in ids, "n_partners": len(ps),
                        "partners": [{"gene": y, "score": s, "channels": c} for y, (s, c) in ps[:p["max_partners"]]],
                        "terms": [t["term"] for t in terms if g in t["genes"]]}
    return {"params": p, "n_genes": len(genes), "n_mapped": len(mapped), "unmapped": [g for g in genes if g not in ids],
            "n_background": len(background_ids) if background_ids else "genome",
            "ppi_enrichment": {k: ppi.get(k) for k in ("number_of_nodes", "number_of_edges",
                                                       "expected_number_of_edges", "average_node_degree", "p_value")},
            "terms": terms, "genes": gene_rows,
            "regulator_network": regulator_network(regulators, ids, edges, p)}


def write_tables(result, out_dir):
    """string_sig.tsv (one row per term) and string_genes.json (PPI, per-gene partners, regulator network) into
    out_dir."""
    out_dir = Path(out_dir)
    with (out_dir / "string_sig.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(TERM_COLUMNS)
        for t in result["terms"]:
            w.writerow([*(t[k] for k in TERM_COLUMNS[:-1]), ",".join(t["genes"])])
    (out_dir / "string_genes.json").write_text(json.dumps(
        {k: result[k] for k in ("params", "n_genes", "n_mapped", "unmapped", "n_background", "ppi_enrichment", "genes",
                                "regulator_network")},
        indent=2))


def fail(code, message):
    return {"ok": False, "error": {"code": code, "message": message}, "warnings": []}


def main():
    try:
        req = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as e:
        return fail("invalid_json", str(e))
    genes = req.get("genes")
    if not isinstance(genes, list) or not genes or not all(isinstance(g, str) for g in genes):
        return fail("invalid_input", "`genes` must be a non-empty list of gene symbols, ranked")
    params = {k: req[k] for k in DEFAULTS if k in req}
    try:
        regs = req.get("regulators")
        if regs is not None and (not isinstance(regs, list) or not all(isinstance(g, str) for g in regs)):
            return fail("invalid_input", "`regulators` must be a list of gene symbols")
        names = genes + list(req.get("candidates") or {}) + list(regs or []) + list(req.get("background_genes") or [])
        ids = map_string_ids(names, params.get("taxid", DEFAULTS["taxid"]), req.get("id_cache"))
        background = sorted({ids[g] for g in req.get("background_genes") or [] if g in ids}) or None
        result = enrich_program(genes, ids, background, req.get("candidates"), params, regs)
    except Exception as e:   # network / API errors are reported, not raised
        return fail("network_error", f"{type(e).__name__}: {e}")
    if req.get("write_to"):
        write_tables(result, req["write_to"])
    max_terms = int(req.get("max_terms", 25))
    rn = result["regulator_network"]
    return {"ok": True, "source": STRING_API, "n_genes": result["n_genes"], "n_mapped": result["n_mapped"],
            "n_terms": len(result["terms"]), "ppi_enrichment": result["ppi_enrichment"],
            "regulator_network": {**{k: rn[k] for k in ("n_regulators", "n_mapped", "background", "ppi_enrichment",
                                                        "n_edges_by_type")},
                                  "edges": rn["edges"][:max_terms]},
            "terms": [{k: t[k] for k in ("category", "description", "fdr", "n_genes")} for t in result["terms"][:max_terms]],
            "written_to": req.get("write_to"), "warnings": []}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
