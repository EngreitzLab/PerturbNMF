"""LLM query agent: turn a program's research_brief (or a vague question) into literature queries.

The fixed searchers in 1.1.Search_literature/ (search_NCBI, search_Crossref, search_Nature, ...) need specific
queries. This agent reads a compact digest of a program bundle -- genes with aliases and database
function, GeneRIF context (search_generif.py), GO terms, regulators with log2FC, OmniPath pairs --
plus either the bundle's research_brief or --question, and returns:

  queries             one per hypothesis, each with a specificity ladder of PubMed-syntax and
                      plain-text queries (run by run_literature_search.py --mode plan)
  curation_questions  the questions 2.Evidence_curation puts to paper-qa over the PDFs found

Stored in '<out_dir>/P<k>.json' -> literature_plan[<plan_name>] (other plans kept) + meta_QueryAgent.json.

    python llm_query_agent.py --info_dir .../Gene_info_extended_PerturbNMF_Info --programs 1
    python llm_query_agent.py ... --question "why does VIRMA knockdown reduce translation genes" --plan_name virma

Env: ANTHROPIC_API_KEY (or AGeneTic/.env).
"""
import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src import (default_out_dir, describe_gene, gene_aliases, gene_entries, load_env,  # noqa: E402
                 load_program_JSON, make_handler, regulator_stats, write_bundles, write_meta)

SOURCE = "QueryAgent"

# question types the agent may plan; the key is stored on every query / curation question
QUESTION_TYPES = {
    "A1_direct": "Is program gene X a known direct target of regulator R (binding, modification, transcriptional or post-transcriptional control)?",
    "A2_mechanism": "Given R's known mechanism, does its loss plausibly act on the program's GO theme as a whole (query R's mechanism with the theme, not gene by gene)?",
    "A3_direction": "Does published evidence agree with the sign of R's log2FC on the program (knockdown -> program down/up)?",
    "A4_indirect": "Is there a published R -> intermediate -> X chain where the intermediate is also a program gene or regulator?",
    "B5_coherence": "Do several program genes act together in one pathway, complex or process?",
    "B6_prior_module": "Has this gene module (or its top genes together) been reported in prior scRNA-seq / Perturb-seq / co-expression studies?",
    "C7_cell_type": "Does gene X have a known role in this cell type / tissue, or only elsewhere (target genes whose GeneRIF evidence_in_cell_type is false)?",
    "C8_condition": "Is gene X or regulator R known to respond to the experimental condition(s) in program_specificity?",
    "D9_disease": "Is R or X linked to the disease context (GWAS, endothelial dysfunction, patient data)?",
    "E10_novelty": "Is there any evidence at all for an untested pair / claim, including negative or contradicting evidence (a clean negative marks a candidate novel link)?",
}
QuestionType = Literal[tuple(QUESTION_TYPES)]


class QueryRung(BaseModel):
    scope: str = Field(description="What this rung covers, e.g. 'gene + regulator + cell type'.")
    pubmed_query: str = Field(description="PubMed E-utilities syntax: field tags ([tiab], [mesh]), AND/OR/NOT, aliases OR-ed in parentheses.")
    text_query: str = Field(description="Plain keyword query for Crossref / Nature / OpenAlex / web search (no field tags).")


class PlannedQuery(BaseModel):
    id: str = Field(description="Short unique id, e.g. Q1.")
    question_type: QuestionType
    hypothesis: str = Field(description="The specific, testable statement this query looks for evidence on.")
    genes: list[str] = Field(description="Gene symbols involved, exactly as named in the digest.")
    priority: Literal[1, 2, 3] = Field(description="1 = most informative for the program's function.")
    ladder: list[QueryRung] = Field(description="2-4 rungs from most specific to broadest; searched in order.")


class CurationQuestion(BaseModel):
    id: str = Field(description="Short unique id, e.g. C1.")
    question_type: QuestionType
    question: str = Field(description="A self-contained question answerable from retrieved papers (names genes and cell type explicitly).")
    genes: list[str]
    query_ids: list[str] = Field(description="Ids of the queries whose papers answer this question.")


class LiteraturePlan(BaseModel):
    interpreted_goal: str = Field(description="One or two sentences: what the literature search should establish.")
    queries: list[PlannedQuery]
    curation_questions: list[CurationQuestion]


SYSTEM_PROMPT = """You plan literature searches for gene programs from CRISPR perturbation screens.
You receive a digest of one program and a research goal (a research brief or a user question). Turn the
goal into specific, searchable hypotheses and the queries that test them.

Question types you may use:
{types}

Rules:
- Only name genes that appear in the digest. Put every alias of a gene into its queries (OR-ed), because
  papers often use old symbols (e.g. KIAA1429 for VIRMA).
- Each query's ladder goes from most specific (genes + regulator + cell type / condition) to broadest
  (gene + regulator's pathway or gene family); the searcher stops at the first rung with relevant hits.
- Prefer regulator -> program questions (A*) for regulators with the strongest effect; use C7 for genes
  whose GeneRIF papers are not in this cell type; include at least one E10 question for an untested pair.
- {disease}
- Plan at most {max_queries} queries. Write one curation question per hypothesis worth answering from papers.
Do not assign a program label and do not claim any finding; you are only planning searches."""


# digest
def _fmt_stats(stats):
    return "; ".join(f"{c}: log2FC={s.get('log2fc')}, adj_p={s.get('adj_pval')}" for c, s in stats.items())


def bundle_digest(bundle, max_genes=40):
    entries = gene_entries(bundle)
    regs = regulator_stats(bundle)
    lines = [f"Program {bundle.get('program_id')} | organism: {bundle.get('organism')} | "
             f"cell type: {bundle.get('cell_type')} | conditions: {', '.join(bundle.get('conditions', []))}",
             f"Enriched GO terms: {', '.join(bundle.get('GO', [])) or 'none'}",
             f"Program specificity: {bundle.get('program_specificity', {}).get('per_condition')}", ""]

    def gene_lines(gene):
        entry = entries.get(gene, {})
        text = describe_gene(gene, entry)
        grif = entry.get("gene_info", {}).get("GeneRIF") or {}
        if grif.get("found"):
            text += (f"\n  GeneRIF context (evidence_in_cell_type={grif.get('evidence_in_cell_type')}): "
                     f"{grif.get('context_summary')}")
        return text

    for key, title in (("program_genes", "Program genes (top loadings)"),
                       ("distinctive_genes", "Distinctive genes")):
        genes = [e["gene"] for e in bundle.get(key, []) if isinstance(e, dict)][:max_genes]
        lines += [f"{title}:"] + [gene_lines(g) for g in genes] + [""]
    lines.append("Perturbation regulators (knockdown effect on this program):")
    lines += [f"{gene_lines(r)}\n  effect: {_fmt_stats(s)}" for r, s in regs.items()] + [""]

    omnipath = bundle.get("gene_interactions", {}).get("OmniPath")
    if omnipath:
        pairs = omnipath.get("pairs", {}).values()
        found = [f"{p['regulator']}-{p['gene']}" for p in pairs if p.get("found")]
        missing = [f"{p['regulator']}-{p['gene']}" for p in pairs if not p.get("found")]
        lines += [f"OmniPath regulator-gene pairs with a database interaction: {', '.join(found) or 'none'}",
                  f"OmniPath pairs with NO database interaction: {', '.join(missing) or 'none'}"]
    return "\n".join(lines)


def build_plan(bundle, goal, handler, args):
    system = SYSTEM_PROMPT.format(
        types="\n".join(f"  {k}: {v}" for k, v in QUESTION_TYPES.items()),
        disease=(f"Disease context for D9: {args.disease_context}." if args.disease_context
                 else "No disease context was given: do not plan D9 questions."),
        max_queries=args.max_queries)
    prompt = f"Research goal:\n{goal}\n\nProgram digest:\n{bundle_digest(bundle)}"
    out = handler.get_json(system, prompt, LiteraturePlan, effort=args.effort, max_tokens=args.max_tokens)
    return out.model_dump() if out is not None else None


def from_research_brief(bundle, handler, args):
    brief = bundle.get("research_brief")
    if not brief:
        raise ValueError(f"{bundle.get('program_id')}: no research_brief; pass --question instead")
    return build_plan(bundle, brief, handler, args)


def from_question(question, bundle, handler, args):
    return build_plan(bundle, question, handler, args)


# post-processing (deterministic)
def regulator_strength(genes, regs):
    """max |log2FC| * -log10(adj_p) over the regulators among genes (0 when none)."""
    best = 0.0
    for g in genes:
        for s in regs.get(g, {}).values():
            if s.get("log2fc") is not None and s.get("adj_pval"):
                best = max(best, abs(s["log2fc"]) * -math.log10(max(s["adj_pval"], 1e-300)))
    return best


def finalize_plan(plan, bundle, plan_name, args):
    """Validate genes, add aliases, rank, cap and prefix ids; returns (plan, dropped log)."""
    entries = gene_entries(bundle)
    regs = regulator_stats(bundle)
    lookup = {}
    for gene, entry in entries.items():
        for name in gene_aliases(gene, entry):
            lookup.setdefault(name.upper(), gene)
    dropped = []

    def resolve(genes, owner):
        kept = []
        for g in genes:
            if g.upper() in lookup:
                kept.append(lookup[g.upper()])
            else:
                dropped.append(f"{owner}: unknown gene {g}")
        return list(dict.fromkeys(kept))

    queries = []
    for q in plan["queries"]:
        q["genes"] = resolve(q["genes"], q["id"])
        if not q["genes"] or not q["ladder"]:
            dropped.append(f"{q['id']}: no valid genes or empty ladder; query dropped")
            continue
        if q["question_type"] == "D9_disease" and not args.disease_context:
            dropped.append(f"{q['id']}: D9 without --disease_context; query dropped")
            continue
        q["aliases"] = {g: gene_aliases(g, entries.get(g)) for g in q["genes"]}
        q["regulator_strength"] = round(regulator_strength(q["genes"], regs), 3)
        queries.append(q)
    queries.sort(key=lambda q: (q["priority"], -q["regulator_strength"]))
    for q in queries[args.max_queries:]:
        dropped.append(f"{q['id']}: over --max_queries; query dropped")
    queries = queries[:args.max_queries]

    ids = {q["id"]: f"{plan_name}:{q['id']}" for q in queries}
    for q in queries:
        q["id"] = ids[q["id"]]
    questions = []
    for c in plan["curation_questions"]:
        c["genes"] = resolve(c["genes"], c["id"])
        c["query_ids"] = [ids[i] for i in c["query_ids"] if i in ids]
        if not c["query_ids"] and c["question_type"] != "C7_cell_type":   # C7 can also read gene_pdfs/
            dropped.append(f"{c['id']}: none of its queries kept; question dropped")
            continue
        c["id"] = f"{plan_name}:{c['id']}"
        questions.append(c)
    return {**plan, "queries": queries, "curation_questions": questions}, dropped


def build_parser():
    p = argparse.ArgumentParser(description="Turn a program's research_brief (or --question) into literature queries and curation questions.")

    # IO
    p.add_argument("--info_dir", required=True, help="Gene_info_extended_PerturbNMF_Info folder from 1.0.Search_database.")
    p.add_argument("--out_dir", default=None, help="Literature_info_extended_PerturbNMF_Info folder. Default: <info_dir>/../Literature_info_extended_PerturbNMF_Info.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3).")
    p.add_argument("--question", default=None, help="Free-text question to plan for instead of the research_brief (vague -> specific mode).")
    p.add_argument("--plan_name", default=None, help="Key under literature_plan (other plans are kept). Default: 'brief', or 'question' with --question.")
    p.add_argument("--disease_context", default=None, help="Enables D9 disease questions, e.g. 'coronary artery disease'.")

    # plan
    p.add_argument("--max_queries", type=int, default=12, help="Max queries kept per program (ranked by priority, then regulator |log2FC| x -log10 adj_p).")

    # agent
    p.add_argument("--model", default="claude-sonnet-5", help="Claude model for the query agent.")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"], help="Effort level of the query agent.")
    p.add_argument("--max_tokens", type=int, default=16000, help="Max output tokens per program.")
    p.add_argument("--overwrite", action="store_true", help="Re-plan programs that already have literature_plan[<plan_name>].")
    return p


def main():
    args = build_parser().parse_args()
    load_env()
    plan_name = args.plan_name or ("question" if args.question else "brief")
    out_dir = Path(args.out_dir or default_out_dir(args.info_dir))
    bundles = load_program_JSON(args.info_dir, out_dir, args.programs)
    handler = make_handler(args.model)

    summary = {}
    for label, bundle in bundles.items():
        if plan_name in bundle.get("literature_plan", {}) and not args.overwrite:
            print(f"[{label}] literature_plan['{plan_name}'] exists; skipped (use --overwrite)")
            continue
        plan = (from_question(args.question, bundle, handler, args) if args.question
                else from_research_brief(bundle, handler, args))
        if plan is None:
            print(f"[{label}] [warn] query agent returned nothing; bundle left unchanged")
            summary[label] = {"error": True}
            continue
        plan, dropped = finalize_plan(plan, bundle, plan_name, args)
        plan.update(source="question" if args.question else "research_brief", goal=args.question or bundle.get("research_brief"),
                    model=args.model, created=datetime.now().isoformat(timespec="seconds"), dropped=dropped)
        bundle.setdefault("literature_plan", {})[plan_name] = plan
        types = {}
        for q in plan["queries"]:
            types[q["question_type"]] = types.get(q["question_type"], 0) + 1
        summary[label] = {"n_queries": len(plan["queries"]), "n_questions": len(plan["curation_questions"]),
                          "question_types": types, "n_dropped": len(dropped)}
        print(f"[{label}] {len(plan['queries'])} queries, {len(plan['curation_questions'])} curation questions; "
              f"types {types}; {len(dropped)} dropped")
        for d in dropped:
            print(f"    [drop] {d}")

    write_bundles(out_dir, bundles)
    write_meta(out_dir, SOURCE, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"info_dir": str(args.info_dir)},
        "plan_name": plan_name,
        "params": {k: getattr(args, k) for k in ("question", "disease_context", "max_queries", "model", "effort", "max_tokens")},
        "programs": summary,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
