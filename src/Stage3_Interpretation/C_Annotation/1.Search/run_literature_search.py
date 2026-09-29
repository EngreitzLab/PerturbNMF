"""Claude-mediated literature search, driven by a query plan or by OmniPath's not-found pairs.

A Claude Agent SDK agent calls the paper searchers (search_NCBI, search_OpenAlex, search_Crossref,
search_Nature, ...) as tools. Open-access / subscribed PDFs land in one folder per task; web hits
(built-in WebSearch) are kept as JSON. Two modes:

  --mode plan      one task per literature_plan query (llm_query_agent.py): the agent runs the
                   query's specificity ladder in order and reports the rung that found evidence.
                   Folder: <query id> with ':' -> '__' (e.g. brief__Q1).
  --mode omnipath  one task per gene_interactions.OmniPath pair with found: false; the agent
                   writes its own queries. Folder: evidence_cache.pair_key (e.g. ACVR2B__SMAD2).
  --mode auto      (default) plan when the bundle has a literature_plan, else omnipath.

    <out_dir>/<task>/*.pdf              PDFs, read later by 2.Evidence_curation (paper-qa)
    <out_dir>/<task>/search_log.json    tool calls, agent result, PDFs on disk
    <out_dir>/<task>/web_hits.json      web search hits (title, url, snippet)
    <out_dir>/meta.json

A task shared by several programs is searched once. The bundles are not modified.

Env: ANTHROPIC_API_KEY (or a Claude Code login), NCBI_EMAIL / NCBI_API_KEY, UNPAYWALL_EMAIL,
OPENALEX_API_KEY (only for the openalex source); read from AGeneTic/.env when present.
"""
import argparse
import asyncio
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENETIC_DIR = HERE.parent
sys.path[:0] = [str(HERE), str(HERE / "1.1.Search_literature"), str(AGENETIC_DIR / "2.Evidence_curation")]

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,  # noqa: E402
                              ToolUseBlock, create_sdk_mcp_server, query, tool)
from evidence_cache import pair_key  # noqa: E402
from src import default_out_dir, describe_gene, gene_entries, load_env, load_program_JSON  # noqa: E402

SERVER = "literature"
# source -> (module, function); pubmed / arxiv need langchain_community
SOURCES = {
    "ncbi": ("search_ncbi", "search_NCBI"),
    "openalex": ("search_openalex", "search_OpenAlex"),
    "crossref": ("search_crossref", "search_Crossref"),
    "nature": ("search_nature", "search_Nature"),
    "pubmed": ("search_pubmed", "search_PubMed"),
    "arxiv": ("search_arxiv", "search_Arxiv"),
}

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "rung_reached": {"type": ["integer", "null"]},   # plan mode: 1-based ladder rung that found evidence
        "papers": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "pmid": {"type": ["string", "null"]},
                "doi": {"type": ["string", "null"]},
                "url": {"type": ["string", "null"]},
                "pdf_path": {"type": ["string", "null"]},
                "relevance": {"type": "string"},
            },
            "required": ["title", "relevance"],
        }},
        "web_hits": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "url": {"type": "string"},
                           "snippet": {"type": "string"}},
            "required": ["title", "url"],
        }},
        "notes": {"type": "string"},
    },
    "required": ["queries", "papers", "web_hits", "notes"],
}

SYSTEM_PROMPT = """You are a literature-search specialist for CRISPR perturbation screens.
A regulator was knocked down/out and a gene program changed; the program's gene below has no
known interaction with that regulator in OmniPath. Find published evidence that relates the
two genes (direct binding, modification, transcriptional or post-transcriptional regulation,
shared pathway / complex, shared phenotype), ideally in the given cell type.

How to search:
- Call the search tools yourself; every tool downloads available PDFs automatically.
- Start with both genes together (official symbols AND aliases), then broaden with the
  program's biology (GO terms, cell type) or the regulator's known function.
- Use search_Nature for journal-specific searches and web search for anything the paper
  searchers miss (reviews, database pages, preprints).
- Stop once further queries return nothing new, or after about {max_calls} tool calls.

Report ONLY papers and pages that the tools returned; never invent a title, PMID, DOI, URL
or PDF path. For each paper give a one-sentence relevance note. Put web results in web_hits.
"""

PLAN_SYSTEM_PROMPT = """You are a literature-search specialist for CRISPR perturbation screens.
You are given one hypothesis about a gene program and a ladder of prepared queries, from most
specific to broadest. Find published evidence for or against the hypothesis.

How to search:
- Call the search tools yourself; every tool downloads available PDFs automatically.
- Run the rungs in order: the PubMed query with search_NCBI, the plain-text query with the other
  paper searchers. Stop descending at the first rung that returns relevant papers, then refine
  around those hits (aliases, the regulator's pathway, the cell type) if useful.
- Contradicting or negative evidence is as valuable as supporting evidence; report it.
- Use web search for anything the paper searchers miss (reviews, database pages, preprints).
- Stop once further queries return nothing new, or after about {max_calls} tool calls.

Report ONLY papers and pages that the tools returned; never invent a title, PMID, DOI, URL
or PDF path. For each paper give a one-sentence relevance note (supports / contradicts / context).
Set rung_reached to the 1-based rung that first returned relevant papers (null if none did).
Put web results in web_hits.
"""


# loaders
def not_found_pairs(bundle):
    omnipath = bundle.get("gene_interactions", {}).get("OmniPath")
    if omnipath is None:
        raise ValueError(f"{bundle.get('program_id')}: no gene_interactions.OmniPath block "
                         "(run 1.0.Search_database/search_gene_interaction/search_OmniPath.py first)")
    return [p for p in omnipath["pairs"].values() if not p["found"]]


def plan_queries(bundle):
    """Every query of every literature_plan in the bundle (llm_query_agent.py), in plan order."""
    return [q for plan in bundle.get("literature_plan", {}).values() for q in plan.get("queries", [])]


def task_dir_name(query_id):
    return query_id.replace(":", "__")


def build_plan_prompt(q, entries, bundle):
    ladder = "\n".join(f"  rung {i}: {r['scope']}\n    PubMed: {r['pubmed_query']}\n    text: {r['text_query']}"
                       for i, r in enumerate(q["ladder"], 1))
    return "\n".join([
        f"Program {bundle.get('program_id')} ({bundle.get('organism', 'human')}, "
        f"cell type: {bundle.get('cell_type') or 'unspecified'}).",
        f"Program GO terms: {', '.join(bundle.get('GO', [])[:8]) or 'none'}",
        "",
        f"Hypothesis ({q['question_type']}): {q['hypothesis']}",
        "",
        "Genes involved:",
        *[describe_gene(g, entries.get(g)) for g in q["genes"]],
        "",
        "Query ladder:",
        ladder,
    ])


def build_prompt(pair, entries, bundle):
    gene, reg = pair["gene"], pair["regulator"]
    stats = "; ".join(f"{c}: log2FC={s.get('log2fc')}, adj_p={s.get('adj_pval')}"
                      for c, s in pair.get("regulator_stats", {}).items())
    return "\n".join([
        f"Program {bundle.get('program_id')} ({bundle.get('organism', 'human')}, "
        f"cell type: {bundle.get('cell_type') or 'unspecified'}).",
        f"Program GO terms: {', '.join(bundle.get('GO', [])[:8]) or 'none'}",
        "",
        f"Program gene ({pair.get('query_category')}):",
        describe_gene(gene, entries.get(gene)),
        "",
        f"Perturbed regulator ({stats}):",
        describe_gene(reg, entries.get(reg)),
        "",
        f"Find literature relating {gene} and {reg}.",
    ])


# tools
def load_search_functions(sources):
    funcs = {}
    for s in sources:
        module, name = SOURCES[s]
        try:
            funcs[name] = getattr(importlib.import_module(module), name)
        except ImportError as e:
            raise ImportError(f"source '{s}' ({module}.py) cannot be imported: {e}") from e
    return funcs


def list_pdfs(pair_dir):
    return {p.name for p in Path(pair_dir).glob("*.pdf")}


def make_tools(funcs, pair_dir, max_papers, calls):
    """One agent tool per search function; each downloads into pair_dir and logs to calls."""
    schema = {"type": "object",
              "properties": {"query": {"type": "string"},
                             "max_results": {"type": "integer", "minimum": 1, "maximum": 15}},
              "required": ["query"]}
    tools = []
    for name, fn in funcs.items():
        def _make(name=name, fn=fn):
            @tool(name, (fn.__doc__ or name).strip(), schema)
            async def _run(args):
                n = min(int(args.get("max_results") or max_papers), max_papers)
                before = list_pdfs(pair_dir)
                try:
                    text = await asyncio.to_thread(fn, args["query"], n, None, str(pair_dir))
                except Exception as e:  # a failing source must not end the agent run
                    text = f"Error during {name}: {e}"
                new = sorted(list_pdfs(pair_dir) - before)
                calls.append({"tool": name, "query": args["query"], "max_results": n,
                              "new_pdfs": new, "error": text.startswith("Error")})
                return {"content": [{"type": "text", "text": str(text)}]}
            return _run
        tools.append(_make())
    return tools


# search
async def search_task(prompt, system_prompt, pair_dir, funcs, args, task_info):
    """Run one agent search into pair_dir; task_info (pair or query fields) is stored in the log."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    calls, web_queries = [], []
    server = create_sdk_mcp_server(SERVER, tools=make_tools(funcs, pair_dir, args.max_papers, calls))
    allowed = [f"mcp__{SERVER}__{n}" for n in funcs] + (["WebSearch"] if args.web_search else [])
    options = ClaudeAgentOptions(
        system_prompt=system_prompt.format(max_calls=args.max_tool_calls),
        mcp_servers={SERVER: server},
        tools=["WebSearch"] if args.web_search else [],   # no Bash / Read / Write
        allowed_tools=allowed,
        permission_mode="dontAsk",
        setting_sources=[],                                # ignore user/project settings
        model=args.model,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        output_format={"type": "json_schema", "schema": RESULT_SCHEMA},
        cwd=str(pair_dir),
    )
    result, final = None, None
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and block.name == "WebSearch":
                    web_queries.append(block.input.get("query"))
        elif isinstance(msg, ResultMessage):
            final, result = msg, msg.structured_output

    result = result or {"queries": [], "papers": [], "web_hits": [], "notes": ""}
    pdfs = sorted(list_pdfs(pair_dir))
    # keep the agent's paper list, but flag PDF paths that do not exist on disk
    for paper in result.get("papers", []):
        path = paper.get("pdf_path")
        paper["pdf_exists"] = bool(path) and Path(path).name in pdfs
    log = {
        "created": datetime.now().isoformat(timespec="seconds"),
        **task_info,
        "rung_reached": result.get("rung_reached"),
        "model": args.model,
        "tool_calls": calls,
        "web_search_queries": web_queries,
        "result": result,
        "pdfs": pdfs,
        "n_pdfs": len(pdfs),
        "num_turns": getattr(final, "num_turns", None),
        "total_cost_usd": getattr(final, "total_cost_usd", None),
        "is_error": getattr(final, "is_error", True),
        "stop_reason": getattr(final, "subtype", None),
    }
    (pair_dir / "web_hits.json").write_text(json.dumps(result.get("web_hits", []), indent=2))
    (pair_dir / "search_log.json").write_text(json.dumps(log, indent=2))
    return log


def build_parser():
    p = argparse.ArgumentParser(description="Claude Agent SDK literature search for literature_plan queries or OmniPath not-found gene-regulator pairs.")

    # IO
    p.add_argument("--info_dir", required=True, help="Gene_info_extended_PerturbNMF_Info folder from 1.0.Search_database.")
    p.add_argument("--lit_dir", default=None, help="Literature_info_extended_PerturbNMF_Info folder (P<k>.json there win over --info_dir). Default: <info_dir>/../Literature_info_extended_PerturbNMF_Info.")
    p.add_argument("--out_dir", default=None, help="Search output folder. Default: <lit_dir>/Literature_search.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3).")
    p.add_argument("--mode", default="auto", choices=["auto", "plan", "omnipath"], help="plan: literature_plan queries (llm_query_agent.py); omnipath: OmniPath not-found pairs; auto: plan when the bundle has one.")
    p.add_argument("--max_queries", type=int, default=None, help="plan mode: max queries searched per program (in plan order). Default: all.")
    p.add_argument("--max_pairs", type=int, default=None, help="omnipath mode: max not-found pairs searched per program (in OmniPath's stored order). Default: all.")

    # search
    p.add_argument("--sources", nargs="+", default=["ncbi", "crossref", "nature"], choices=list(SOURCES), help="Paper searchers given to the agent (openalex needs OPENALEX_API_KEY; pubmed/arxiv need langchain_community).")
    p.add_argument("--max_papers", type=int, default=5, help="Max papers per tool call (searchers cap at 15).")
    p.add_argument("--no_web_search", dest="web_search", action="store_false", help="Do not give the agent the WebSearch tool.")

    # agent
    p.add_argument("--model", default="claude-sonnet-5", help="Claude model for the search agent.")
    p.add_argument("--max_turns", type=int, default=20, help="Max agent turns per task.")
    p.add_argument("--max_tool_calls", type=int, default=8, help="Tool-call budget stated in the agent's instructions.")
    p.add_argument("--max_budget_usd", type=float, default=1.0, help="Hard spend cap per task (USD).")
    p.add_argument("--overwrite", action="store_true", help="Re-search tasks that already have a search_log.json.")
    return p


def program_tasks(bundle, mode, args):
    """[(folder name, prompt, system prompt, task_info)] for one bundle."""
    entries = gene_entries(bundle)
    info = {"program": bundle.get("program_id"), "cell_type": bundle.get("cell_type")}
    if mode == "plan":
        return [(task_dir_name(q["id"]), build_plan_prompt(q, entries, bundle), PLAN_SYSTEM_PROMPT,
                 {**info, "mode": "plan", "query_id": q["id"], "question_type": q["question_type"],
                  "hypothesis": q["hypothesis"], "genes": q["genes"]})
                for q in plan_queries(bundle)[:args.max_queries]]
    return [(pair_key(p["gene"], p["regulator"]), build_prompt(p, entries, bundle), SYSTEM_PROMPT,
             {**info, "mode": "omnipath", "pair": pair_key(p["gene"], p["regulator"]),
              "gene": p["gene"], "regulator": p["regulator"]})
            for p in not_found_pairs(bundle)[:args.max_pairs]]


async def run(args):
    lit_dir = Path(args.lit_dir or default_out_dir(args.info_dir))
    out_dir = Path(args.out_dir or lit_dir / "Literature_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    funcs = load_search_functions(args.sources)
    bundles = load_program_JSON(args.info_dir, lit_dir, args.programs)

    summary, done = {}, set()
    for label, bundle in bundles.items():
        mode = args.mode if args.mode != "auto" else ("plan" if plan_queries(bundle) else "omnipath")
        tasks = program_tasks(bundle, mode, args)
        print(f"[{label}] {len(tasks)} {mode} task(s) to search")
        summary[label] = {"mode": mode, "tasks": []}
        for key, prompt, system_prompt, task_info in tasks:
            task_dir = out_dir / key
            summary[label]["tasks"].append(key)
            if key in done or ((task_dir / "search_log.json").exists() and not args.overwrite):
                print(f"  {key}: already searched; skipped (use --overwrite)")
                continue
            log = await search_task(prompt, system_prompt, task_dir, funcs, args, task_info)
            done.add(key)
            print(f"  {key}: tool calls={len(log['tool_calls'])} web={len(log['web_search_queries'])} "
                  f"PDFs={log['n_pdfs']} rung={log['rung_reached']} cost=${log['total_cost_usd'] or 0:.3f} "
                  f"{'[agent error: ' + str(log['stop_reason']) + ']' if log['is_error'] else ''}")

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"info_dir": str(args.info_dir), "lit_dir": str(lit_dir)},
        "params": {k: getattr(args, k) for k in (
            "mode", "max_queries", "max_pairs", "sources", "max_papers", "web_search", "model",
            "max_turns", "max_tool_calls", "max_budget_usd")},
        "programs": summary,
        "n_searched_this_run": len(done),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(done)} task(s) searched -> {out_dir}; meta -> {out_dir / 'meta.json'}")


def main():
    args = build_parser().parse_args()
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [warn] ANTHROPIC_API_KEY not set (env or AGeneTic/.env); relying on a Claude Code login")
    if "openalex" in args.sources and not os.environ.get("OPENALEX_API_KEY"):
        raise SystemExit("--sources openalex needs OPENALEX_API_KEY")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
