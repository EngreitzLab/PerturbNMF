"""Shared helpers for the gene-database scripts (MyGene.py, NCBI.py, UniProt.py).

Each script adds its own source-labeled block to every gene of a program bundle:

    {"gene": "KIAA1429", "gene_info": {"MyGene": {...}, "NCBI": {...}, "UniProt": {...}}}

All scripts write to one shared output folder; a script reads '<out_dir>/P<k>.json'
when it already exists (so earlier sources are kept) and '<bundle_dir>/P<k>.json'
otherwise. The original bundles are never modified.
"""
import copy
import json
import time
from pathlib import Path

import requests

TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
OUT_FOLDER = "Gene_info_extended_PerturbNMF_Info"
GENE_LISTS = ("program_genes", "distinctive_genes")
NOT_FOUND = {"found": False}


# loaders
def _check_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return path


def default_out_dir(bundle_dir):
    return Path(bundle_dir).resolve().parent / OUT_FOLDER


def load_program_JSON(bundle_dir, out_dir, programs):
    """{label: bundle} for the requested programs only; '<out_dir>/P<k>.json' is read
    when it exists (already extended by another source), else '<bundle_dir>/P<k>.json'."""
    bundle_dir, out_dir = _check_file(bundle_dir), Path(out_dir)
    programs = list(dict.fromkeys(programs))  # dedupe, keep order
    paths = {}
    for pid in programs:
        extended = out_dir / f"P{pid}.json"
        paths[pid] = extended if extended.exists() else bundle_dir / f"P{pid}.json"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No bundle for {len(missing)} of {len(programs)} programs: {missing}")
    n_ext = sum(p.parent == out_dir for p in paths.values())
    print(f"[load] {len(paths)} bundle(s): {n_ext} from {out_dir}, {len(paths) - n_ext} from {bundle_dir}")
    bundles = {}
    for pid, p in paths.items():
        bundles[f"P{pid}"] = json.loads(p.read_text())
        validate_bundle(bundles[f"P{pid}"], p)
    return bundles


def validate_bundle(bundle, path):
    """Raise if the bundle lacks the gene lists or has malformed gene entries."""
    def _valid_entry(e):
        return isinstance(e, str) or (isinstance(e, dict) and isinstance(e.get("gene"), str)
                                      and isinstance(e.get("gene_info", {}), dict))

    for key in GENE_LISTS:
        if not isinstance(bundle.get(key), list):
            raise ValueError(f"{path}: '{key}' missing or not a list")
        bad = [e for e in bundle[key] if not _valid_entry(e)]
        if bad:
            raise ValueError(f"{path}: '{key}' has {len(bad)} entries that are neither a gene "
                             f"name nor {{'gene': ...}}: {bad[:3]}")
    regulators = bundle.get("perturbation_regulators", {})
    if not isinstance(regulators, dict):
        raise ValueError(f"{path}: 'perturbation_regulators' must be {{condition: [entries]}}")
    for cond, regs in regulators.items():
        if not isinstance(regs, list):
            raise ValueError(f"{path}: perturbation_regulators['{cond}'] is not a list")
        bad = [r for r in regs if not (isinstance(r, dict) and _valid_entry(r))]
        if bad:
            raise ValueError(f"{path}: perturbation_regulators['{cond}'] has {len(bad)} entries "
                             f"without a 'gene' name: {bad[:3]}")


def gene_name(entry):
    """Gene symbol of a plain-string entry (original bundle) or a dict entry (extended)."""
    return str(entry) if isinstance(entry, str) else str(entry["gene"])


def _gene_entries(bundle):
    entries = [e for key in GENE_LISTS for e in bundle.get(key, [])]
    for regs in bundle.get("perturbation_regulators", {}).values():
        entries += regs
    return entries


def collect_genes(bundle):
    """program_genes + distinctive_genes + every regulator gene, deduped in order."""
    return list(dict.fromkeys(gene_name(e) for e in _gene_entries(bundle)))


def split_cached(bundles, source, overwrite=False):
    """(all genes, {gene: existing gene_info[source]}, genes still to query).

    A gene that already carries a gene_info[source] block (found or not) in any loaded
    bundle is not queried again unless overwrite is set."""
    genes = list(dict.fromkeys(g for b in bundles.values() for g in collect_genes(b)))
    cached = {}
    if not overwrite:
        for b in bundles.values():
            for e in _gene_entries(b):
                if isinstance(e, dict) and source in e.get("gene_info", {}):
                    cached.setdefault(gene_name(e), e["gene_info"][source])
    query = [g for g in genes if g not in cached]
    print(f"[{source}] {len(genes)} unique genes: {len(cached)} already have {source} info "
          f"(skipped{'' if cached else '; none'}), {len(query)} to query")
    return genes, cached, query


# query
def request_with_retry(method, url, retries=3, **kwargs):
    """requests.<method> with exponential backoff on transient HTTP status codes and
    dropped connections / timeouts (e.g. NCBI cutting off a large response)."""
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in TRANSIENT_STATUS and attempt < retries:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# bundle
def attach_source(bundle, source, info):
    """Copy of the bundle with info[gene] stored under gene_info[source] on every gene
    entry; plain-string genes become {"gene": g}. Other sources' blocks are kept."""
    ext = copy.deepcopy(bundle)
    for key in GENE_LISTS:
        ext[key] = [e if isinstance(e, dict) else {"gene": e} for e in ext.get(key, [])]
    for e in _gene_entries(ext):
        e.setdefault("gene_info", {})[source] = info.get(gene_name(e), NOT_FOUND)
    return ext


def report_query(source, genes, info):
    """Print found / renamed / not-found counts of the queried genes; return (renamed, not_found)."""
    if not genes:
        print(f"[{source}] nothing to query")
        return {}, []
    not_found = [g for g in genes if g not in info]
    renamed = {g: i["symbol"] for g, i in info.items() if i.get("symbol") and i["symbol"] != g}
    print(f"[{source}] found {len(info)}/{len(genes)} queried; renamed {len(renamed)}")
    if not_found:
        print(f"  [warn] {len(not_found)} genes not found in {source}: {not_found}")
    return renamed, not_found


def write_outputs(out_dir, bundles, source, info, meta):
    """Write '<out_dir>/P<k>.json' with this source attached and '<out_dir>/meta_<source>.json'."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, bundle in bundles.items():
        (out_dir / f"{label}.json").write_text(json.dumps(attach_source(bundle, source, info), indent=2))
        genes = collect_genes(bundle)
        print(f"  {label}: genes={len(genes)} found={sum(info.get(g, NOT_FOUND)['found'] for g in genes)}")
    meta_path = out_dir / f"meta_{source}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(bundles)} bundle(s) -> {out_dir}; meta -> {meta_path}")
