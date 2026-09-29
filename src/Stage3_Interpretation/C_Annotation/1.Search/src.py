"""Shared helpers for the literature scripts (search_generif.py, llm_query_agent.py,
run_literature_search.py).

Every step reads the 1.0.Search_database output ('Gene_info_extended_PerturbNMF_Info/P<k>.json')
and writes an extended copy into a sibling 'Literature_info_extended_PerturbNMF_Info/' folder:

    '<out_dir>/P<k>.json' is read when it exists (an earlier literature step already extended it),
    '<info_dir>/P<k>.json' otherwise. The 1.0 bundles are never modified.
"""
import copy
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENETIC_DIR = HERE.parent
OUT_FOLDER = "Literature_info_extended_PerturbNMF_Info"
GENE_LISTS = ("program_genes", "distinctive_genes")
GENE_INFO_SOURCES = ("MyGene", "NCBI", "UniProt")


# loaders
def load_env(path=AGENETIC_DIR / ".env"):
    """Read KEY=VALUE lines from AGeneTic/.env into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def default_out_dir(info_dir):
    return Path(info_dir).resolve().parent / OUT_FOLDER


def load_program_JSON(info_dir, out_dir, programs):
    """{label: bundle} for the requested programs; '<out_dir>/P<k>.json' wins over '<info_dir>/P<k>.json'."""
    info_dir, out_dir = Path(info_dir), Path(out_dir)
    if not info_dir.exists():
        raise FileNotFoundError(f"Input not found: {info_dir}")
    paths = {}
    for pid in dict.fromkeys(programs):
        extended = out_dir / f"P{pid}.json"
        paths[f"P{pid}"] = extended if extended.exists() else info_dir / f"P{pid}.json"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No bundle for {len(missing)} of {len(paths)} programs: {missing}")
    n_ext = sum(p.parent == out_dir for p in paths.values())
    print(f"[load] {len(paths)} bundle(s): {n_ext} from {out_dir}, {len(paths) - n_ext} from {info_dir}")
    return {label: json.loads(p.read_text()) for label, p in paths.items()}


def write_bundles(out_dir, bundles):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, bundle in bundles.items():
        (out_dir / f"{label}.json").write_text(json.dumps(bundle, indent=2))


def write_meta(out_dir, name, meta):
    path = Path(out_dir) / f"meta_{name}.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"[done] meta -> {path}")


def make_handler(model):
    """Anthropic API handler (api/LLM_interface.claude_api); needs ANTHROPIC_API_KEY."""
    sys.path.insert(0, str(AGENETIC_DIR / "api"))
    from LLM_interface import claude_api
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (export it or add it to AGeneTic/.env)")
    return claude_api(api_key=key, model=model)


# genes
def iter_gene_entries(bundle):
    """Every gene entry dict: program_genes, distinctive_genes, then regulators of every condition."""
    for key in GENE_LISTS:
        for e in bundle.get(key, []):
            if isinstance(e, dict):
                yield e
    for regs in bundle.get("perturbation_regulators", {}).values():
        yield from regs


def gene_entries(bundle):
    """{gene: first entry} over program genes, distinctive genes and regulators."""
    entries = {}
    for e in iter_gene_entries(bundle):
        entries.setdefault(e["gene"], e)
    return entries


def regulator_stats(bundle):
    """{regulator: {condition: {log2fc, adj_pval}}}."""
    stats = {}
    for cond, regs in bundle.get("perturbation_regulators", {}).items():
        for r in regs:
            stats.setdefault(r["gene"], {})[cond] = {"log2fc": r.get("log2fc"), "adj_pval": r.get("adj_pval")}
    return stats


def gene_aliases(gene, entry):
    """[query name, official symbols, aliases] from the gene_info blocks, deduped in order."""
    info = (entry or {}).get("gene_info", {})
    names = [gene]
    for src in GENE_INFO_SOURCES:
        block = info.get(src, {})
        if block.get("found"):
            names += [block.get("symbol")] + list(block.get("alias") or [])
    return list(dict.fromkeys(n for n in names if n))


def describe_gene(gene, entry):
    """Short text of a gene's names and function from its gene_info blocks."""
    info = (entry or {}).get("gene_info", {})
    lines = []
    mg, up = info.get("MyGene", {}), info.get("UniProt", {})
    if mg.get("name"):
        lines.append(f"  name: {mg['name']}")
    summary = mg.get("summary") or up.get("function") or info.get("NCBI", {}).get("summary")
    if summary:
        lines.append(f"  function: {summary[:600]}")
    if mg.get("go_bp"):
        lines.append(f"  GO BP: {', '.join(mg['go_bp'][:6])}")
    return "\n".join([f"- {gene} (names/aliases: {', '.join(gene_aliases(gene, entry))})"] + lines)


def attach_gene_block(bundle, source, info):
    """Copy of the bundle with info[gene] stored under gene_info[source] on every entry of that gene."""
    ext = copy.deepcopy(bundle)
    for e in iter_gene_entries(ext):
        if e["gene"] in info:
            e.setdefault("gene_info", {})[source] = info[e["gene"]]
    return ext
