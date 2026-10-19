"""Claude-mediated literature search: find papers, download their PDFs, read and summarize them.

A Claude Agent SDK agent calls the paper searchers (search_NCBI, search_OpenAlex, search_Crossref,
search_Nature, ...) as tools; every searcher downloads open-access PDFs into the task's folder. The
agent then reads those PDFs (read_pdf tool) and returns one summary per paper
(llm_summary_agent.LiteratureSearchResult). Three task modes:

  --mode brief     one task per program, driven by the bundle's research_brief and a digest of its
                   genes / regulators / GO terms (llm_query_agent.bundle_digest). Folder: P<k>__brief.
  --mode plan      one task per literature_plan query (llm_query_agent.py): the agent runs the
                   query's specificity ladder in order and reports the rung that found evidence.
                   Folder: <query id> with ':' -> '__' (e.g. brief__Q1).
  --mode omnipath  one task per gene_interactions.OmniPath pair with found: false; the agent
                   writes its own queries. Folder: src.pair_key (e.g. ACVR2B__SMAD2).
  --mode auto      (default) plan when the bundle has a literature_plan, else brief.

    <out_dir>/<task>/*.pdf              PDFs (also read by paper-qa, 2.Evidence_curation/query_paper.py)
    <out_dir>/<task>/search_log.json    tool calls, agent result, PDFs on disk
    <out_dir>/<task>/web_hits.json      web search hits (title, url, snippet)
    <out_dir>/meta.json
    <lit_dir>/P<k>.json                 bundle + literature_search[<task>]: summary, open_questions and
                                        papers (pdf_path, summary, finding, stance, experimental context)

Programs come from --program_json (bundle paths) or --info_dir + --programs. '<lit_dir>/P<k>.json' is
read instead when it exists (an earlier literature step already extended it); the input bundle is
never modified. A task shared by several programs is searched once; a task already searched is
re-attached to the bundle from its search_log.json.

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
sys.path[:0] = [str(HERE), str(HERE / "1.1.Search_literature")]

import pymupdf  # noqa: E402
from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,  # noqa: E402
                              ToolUseBlock, create_sdk_mcp_server, query, tool)
from llm_query_agent import bundle_digest  # noqa: E402
from llm_summary_agent import LiteratureSearchResult  # noqa: E402
from src import (default_out_dir, describe_gene, gene_entries, load_env,  # noqa: E402
                 load_program_JSON, pair_key, write_bundles)

SERVER = "literature"
SOURCE = "literature_search"   # bundle key
# source -> (module, function); pubmed / arxiv need langchain_community
SOURCES = {
    "ncbi": ("search_ncbi", "search_NCBI"),
    "openalex": ("search_openalex", "search_OpenAlex"),
    "crossref": ("search_crossref", "search_Crossref"),
    "nature": ("search_nature", "search_Nature"),
    "pubmed": ("search_pubmed", "search_PubMed"),
    "arxiv": ("search_arxiv", "search_Arxiv"),
}

RESULT_SCHEMA = LiteratureSearchResult.model_json_schema()
EMPTY_RESULT = {"queries": [], "rung_reached": None, "papers": [], "web_hits": [], "summary": "",
                "open_questions": [], "notes": ""}

READ_RULES = """
How to read and summarize:
- After searching, call read_pdf (no file argument lists the PDFs on disk), then read each PDF
  once from page 1 (abstract, results); read further pages only when the first pages do not
  answer how the paper bears on the task.
- Return every relevant paper the search tools returned. For a downloaded paper set pdf_file to
  its file name and read_from to fulltext; for a paper without a PDF summarize the abstract the
  search tool returned and set read_from to abstract.
- Summaries, findings and experimental context come only from the text you read; return null
  for a field that is not stated.

Report ONLY papers and pages that the tools returned; never invent a title, PMID, DOI, URL or
file name. Put web results in web_hits.
"""

BRIEF_SYSTEM_PROMPT = """You are a literature-search specialist for CRISPR perturbation screens.
You are given a research brief for one gene program and a digest of its genes, perturbed
regulators (with knockdown effects) and enriched GO terms. Find the papers that best establish
the program's biological function and how its regulators act on it.

How to search:
- Call the search tools yourself; every tool downloads available PDFs automatically.
- Search the program's themes (GO terms with its top genes, official symbols AND aliases), the
  regulators with the strongest effects together with program genes, and the cell type.
- Use search_Nature for journal-specific searches and web search for anything the paper
  searchers miss (reviews, database pages, preprints).
- Stop once further queries return nothing new, or after about {max_calls} search calls.
""" + READ_RULES

OMNIPATH_SYSTEM_PROMPT = """You are a literature-search specialist for CRISPR perturbation screens.
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
- Stop once further queries return nothing new, or after about {max_calls} search calls.
""" + READ_RULES

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
- Stop once further queries return nothing new, or after about {max_calls} search calls.
- Set rung_reached to the 1-based rung that first returned relevant papers (null if none did).
""" + READ_RULES


# loaders
def load_bundles(args, lit_dir):
    """{label: bundle}; '<lit_dir>/<label>.json' wins over the given bundle when it exists."""
    if args.info_dir:
        return load_program_JSON(args.info_dir, lit_dir, args.programs)
    bundles = {}
    for path in map(Path, args.program_json):
        extended = lit_dir / path.name
        src_path = extended if extended.exists() else path
        if not src_path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        bundles[path.stem] = json.loads(src_path.read_text())
        print(f"[load] {path.stem} <- {src_path}")
    return bundles


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


def build_brief_prompt(bundle):
    brief = bundle.get("research_brief")
    if not brief:
        raise ValueError(f"{bundle.get('program_id')}: no research_brief; use --mode plan or omnipath")
    return f"Research brief:\n{brief}\n\nProgram digest:\n{bundle_digest(bundle)}"


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


def pdf_text(path, start_page, max_pages, max_chars):
    """Text of pages start_page .. start_page + max_pages - 1 (1-based), capped at max_chars."""
    with pymupdf.open(path) as doc:
        n = doc.page_count
        first = min(max(1, start_page), n)
        last = min(n, first + max_pages - 1)
        text = "\n".join(doc[i].get_text() for i in range(first - 1, last))
    head = f"{path.name}: pages {first}-{last} of {n}"
    if len(text) > max_chars:
        text = text[:max_chars]
        head += f" (truncated to {max_chars} chars; call again with a later start_page to continue)"
    return f"{head}\n\n{text}"


def make_read_tool(pair_dir, args, calls):
    """read_pdf: text of a PDF in pair_dir only (no other file access); no file -> list the PDFs."""
    schema = {"type": "object",
              "properties": {"file": {"type": "string", "description": "PDF file name in this task's folder."},
                             "start_page": {"type": "integer", "minimum": 1},
                             "max_pages": {"type": "integer", "minimum": 1, "maximum": args.max_pdf_pages}}}

    @tool("read_pdf", "Read the text of a downloaded PDF by file name; call without a file to list "
                      "the PDFs downloaded so far.", schema)
    async def _read(a):
        pdfs = sorted(list_pdfs(pair_dir))
        name = Path(a.get("file") or "").name
        if name not in pdfs:
            text = (f"PDFs on disk ({len(pdfs)}): " + (", ".join(pdfs) or "none")
                    if not name else f"No PDF named {name!r}. PDFs on disk: {', '.join(pdfs) or 'none'}")
        else:
            try:
                text = await asyncio.to_thread(pdf_text, Path(pair_dir) / name, int(a.get("start_page") or 1),
                                               min(int(a.get("max_pages") or args.max_pdf_pages), args.max_pdf_pages),
                                               args.max_pdf_chars)
            except Exception as e:  # a broken PDF must not end the agent run
                text = f"Error reading {name}: {e}"
        calls.append({"tool": "read_pdf", "file": name or None, "start_page": a.get("start_page"),
                      "error": text.startswith(("Error", "No PDF"))})
        return {"content": [{"type": "text", "text": text}]}
    return _read


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
    tools = make_tools(funcs, pair_dir, args.max_papers, calls) + [make_read_tool(pair_dir, args, calls)]
    server = create_sdk_mcp_server(SERVER, tools=tools)
    allowed = [f"mcp__{SERVER}__{n}" for n in [*funcs, "read_pdf"]] + (["WebSearch"] if args.web_search else [])
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

    result = {**EMPTY_RESULT, **(result or {})}
    if task_info.get("mode") != "plan":   # only plan tasks have a query ladder
        result["rung_reached"] = None
    pdfs = sorted(list_pdfs(pair_dir))
    # keep the agent's paper list, but only trust PDF file names that exist on disk
    for paper in result["papers"]:
        name = Path(paper.get("pdf_file") or "").name
        paper["pdf_path"] = str((pair_dir / name).resolve()) if name in pdfs else None
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
    (pair_dir / "web_hits.json").write_text(json.dumps(result["web_hits"], indent=2))
    (pair_dir / "search_log.json").write_text(json.dumps(log, indent=2))
    return log


def bundle_entry(log, task_dir):
    """The literature_search[<task>] block stored in the bundle, from a task's search_log.json."""
    result = log.get("result", {})
    keep = ("mode", "query_id", "question_type", "hypothesis", "pair", "gene", "regulator", "genes")
    return {
        **{k: log[k] for k in keep if k in log},
        "pdf_dir": str(task_dir.resolve()),
        "summary": result.get("summary", ""),
        "open_questions": result.get("open_questions", []),
        "rung_reached": log.get("rung_reached"),
        "papers": result.get("papers", []),
        "n_pdfs": log.get("n_pdfs"),
        "model": log.get("model"),
        "total_cost_usd": log.get("total_cost_usd"),
        "is_error": log.get("is_error"),
        "created": log.get("created"),
    }


def build_parser():
    p = argparse.ArgumentParser(description="Claude Agent SDK literature search: find, download, read and summarize papers for a program's research_brief, literature_plan queries or OmniPath not-found pairs.")

    # IO
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--program_json", nargs="+", help="Program bundle(s) P<k>.json, space separated (e.g. .../Gene_info_extended_PerturbNMF_Info/P1.json).")
    inp.add_argument("--info_dir", help="Gene_info_extended_PerturbNMF_Info folder from 1.0.Search_database (use with --programs).")
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids with --info_dir, space separated (e.g. 1 2 3).")
    p.add_argument("--lit_dir", default=None, help="Literature_info_extended_PerturbNMF_Info folder: extended P<k>.json are read from and written here. Default: sibling of the input folder.")
    p.add_argument("--out_dir", default=None, help="PDF / search-log folder. Default: <lit_dir>/Literature_search.")

    # context info
    p.add_argument("--mode", default="auto", choices=["auto", "brief", "plan", "omnipath"], help="brief: one search from the research_brief; plan: literature_plan queries (llm_query_agent.py); omnipath: OmniPath not-found pairs; auto: plan when the bundle has one, else brief.")
    p.add_argument("--max_queries", type=int, default=None, help="plan mode: max queries searched per program (in plan order). Default: all.")
    p.add_argument("--max_pairs", type=int, default=None, help="omnipath mode: max not-found pairs searched per program (in OmniPath's stored order). Default: all.")

    # search
    p.add_argument("--sources", nargs="+", default=["ncbi", "crossref", "nature"], choices=list(SOURCES), help="Paper searchers given to the agent (openalex needs OPENALEX_API_KEY; pubmed/arxiv need langchain_community).")
    p.add_argument("--max_papers", type=int, default=5, help="Max papers per tool call (searchers cap at 15).")
    p.add_argument("--no_web_search", dest="web_search", action="store_false", help="Do not give the agent the WebSearch tool.")

    # reading
    p.add_argument("--max_pdf_pages", type=int, default=6, help="Max PDF pages returned per read_pdf call.")
    p.add_argument("--max_pdf_chars", type=int, default=20000, help="Max characters returned per read_pdf call.")

    # agent
    p.add_argument("--model", default="claude-sonnet-5", help="Claude model for the search agent.")
    p.add_argument("--max_turns", type=int, default=40, help="Max agent turns per task (search + read calls).")
    p.add_argument("--max_tool_calls", type=int, default=8, help="Search-call budget stated in the agent's instructions (read_pdf calls not counted).")
    p.add_argument("--max_budget_usd", type=float, default=2.0, help="Hard spend cap per task (USD).")
    p.add_argument("--overwrite", action="store_true", help="Re-search tasks that already have a search_log.json.")
    return p


def program_tasks(label, bundle, mode, args):
    """[(folder name, prompt, system prompt, task_info)] for one bundle."""
    entries = gene_entries(bundle)
    info = {"program": bundle.get("program_id"), "cell_type": bundle.get("cell_type")}
    if mode == "brief":
        return [(f"{label}__brief", build_brief_prompt(bundle), BRIEF_SYSTEM_PROMPT,
                 {**info, "mode": "brief", "goal": bundle.get("research_brief")})]
    if mode == "plan":
        return [(task_dir_name(q["id"]), build_plan_prompt(q, entries, bundle), PLAN_SYSTEM_PROMPT,
                 {**info, "mode": "plan", "query_id": q["id"], "question_type": q["question_type"],
                  "hypothesis": q["hypothesis"], "genes": q["genes"]})
                for q in plan_queries(bundle)[:args.max_queries]]
    return [(pair_key(p["gene"], p["regulator"]), build_prompt(p, entries, bundle), OMNIPATH_SYSTEM_PROMPT,
             {**info, "mode": "omnipath", "pair": pair_key(p["gene"], p["regulator"]),
              "gene": p["gene"], "regulator": p["regulator"]})
            for p in not_found_pairs(bundle)[:args.max_pairs]]


async def run(args):
    input_dir = Path(args.info_dir) if args.info_dir else Path(args.program_json[0]).parent
    lit_dir = Path(args.lit_dir or default_out_dir(input_dir))
    out_dir = Path(args.out_dir or lit_dir / "Literature_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    funcs = load_search_functions(args.sources)
    bundles = load_bundles(args, lit_dir)

    summary, done = {}, set()
    for label, bundle in bundles.items():
        mode = args.mode if args.mode != "auto" else ("plan" if plan_queries(bundle) else "brief")
        tasks = program_tasks(label, bundle, mode, args)
        print(f"[{label}] {len(tasks)} {mode} task(s) to search")
        summary[label] = {"mode": mode, "tasks": []}
        for key, prompt, system_prompt, task_info in tasks:
            task_dir = out_dir / key
            log_path = task_dir / "search_log.json"
            summary[label]["tasks"].append(key)
            if key in done or (log_path.exists() and not args.overwrite):
                print(f"  {key}: already searched; re-attached from search_log.json (use --overwrite)")
                log = json.loads(log_path.read_text())
            else:
                log = await search_task(prompt, system_prompt, task_dir, funcs, args, task_info)
                done.add(key)
                n_read = sum(c["tool"] == "read_pdf" and not c["error"] for c in log["tool_calls"])
                print(f"  {key}: tool calls={len(log['tool_calls'])} (read_pdf={n_read}) "
                      f"web={len(log['web_search_queries'])} PDFs={log['n_pdfs']} "
                      f"papers={len(log['result']['papers'])} rung={log['rung_reached']} "
                      f"cost=${log['total_cost_usd'] or 0:.3f} "
                      f"{'[agent error: ' + str(log['stop_reason']) + ']' if log['is_error'] else ''}")
            bundle.setdefault(SOURCE, {})[key] = bundle_entry(log, task_dir)
        write_bundles(lit_dir, {label: bundle})   # after every program, so an interrupted run keeps its work
        print(f"[{label}] bundle -> {lit_dir / f'{label}.json'}")

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"program_json": args.program_json, "info_dir": args.info_dir, "lit_dir": str(lit_dir)},
        "params": {k: getattr(args, k) for k in (
            "mode", "max_queries", "max_pairs", "sources", "max_papers", "web_search", "max_pdf_pages",
            "max_pdf_chars", "model", "max_turns", "max_tool_calls", "max_budget_usd")},
        "programs": summary,
        "n_searched_this_run": len(done),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(done)} task(s) searched -> {out_dir}; meta -> {out_dir / 'meta.json'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.info_dir and not args.programs:
        parser.error("--info_dir needs --programs")
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [warn] ANTHROPIC_API_KEY not set (env or AGeneTic/.env); relying on a Claude Code login")
    if "openalex" in args.sources and not os.environ.get("OPENALEX_API_KEY"):
        raise SystemExit("--sources openalex needs OPENALEX_API_KEY")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
