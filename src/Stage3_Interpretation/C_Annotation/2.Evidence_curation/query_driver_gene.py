"""Ask paper-qa, gene by gene, whether each gene drives its program's final biological annotation.

  python query_driver_gene.py --lit_dir <Literature_info_extended_PerturbNMF_Info> --programs 1 2

The annotation is the program_name + function_sentence of '<label_dir>/P<k>.json'
(3.Generate/program_label.py). For every gene with gene function categorization PDFs
(gene_info.Gene_function_categorization.papers[].pdf_path, from
1.Search/1.1.Search_literature/search_gene_function.py) one question is asked,

    Does <GENE> drive "<program_name>", and why?

over that gene's own PDFs only, so every answer rests on papers retrieved for that gene. paper-qa finds the
evidence (top evidence_k passages) and writes the reasons; Claude (--judge_model) then reads paper-qa's evidence
summaries + answer and gives the verdict (supported|contradicted|mixed|insufficient, the biology in any cell
type) and cell_type_evidence (same|related|other|none). paper-qa's own verdict is kept as paperqa_verdict.

Relabel loop: when fewer than --min_explained genes (n, default 3) explain the label (verdict in
--explain_verdicts, default supported / mixed), Claude rewrites the label from the program evidence
(program_label.build_prompt) plus each gene's function categories and paper-qa verdict, and every gene is
asked again; at most --max_rounds relabels. Round r uses up to --pdf_schedule[r] PDFs per gene (default 3, 5,
7), downloading more with search_gene_function.py before the round. The best round (most genes explained,
earliest on ties) is kept, converged or not; when it is not round 0 its label replaces '<label_dir>/P<k>.json',
with every label tried in its label_history.

    <lit_dir>/P<k>.json                         + literature_evidence.driver_gene: {annotation, genes: {<GENE>:
                                                  answer}, explained, converged, rounds: [...], n_by_verdict}
    <label_dir>/P<k>.json                       final label + label_history (only when the label changed)
    <cache_dir>/P<k>__<GENE>__<key>.json        answer cache, key = sha1(prompt + PDF paths + qa settings)[:12]

A re-run reuses each gene's cached answer until the annotation, its PDFs or the settings change (or
--overwrite). Env: OPENAI_API_KEY (paper-qa's default backend), ANTHROPIC_API_KEY (judge, relabel, gene function summaries), read from C_Annotation/.env.
"""
import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "1.Search"), str(HERE.parent / "1.Search" / "1.1.Search_literature"),
                str(HERE.parent / "api")]

from typing import Literal  # noqa: E402

import yaml  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import qa  # noqa: E402
from budget import add_budget_args, attach, claude_worst_case, embedding_cost  # noqa: E402
from src import GENE_LISTS, load_env, make_handler  # noqa: E402

BLOCK = "literature_evidence"
KEY = "driver_gene"
SOURCE = "Gene_function_categorization"
VERDICTS = ("supported", "mixed", "contradicted", "insufficient")
_CELL_RE = re.compile(r"CELL_TYPE\s*[:=]\s*\**\s*(same|related|other|none)", re.IGNORECASE)


class PaperFinding(BaseModel):
    title: str = Field(description="Paper title exactly as given in the evidence.")
    finding: str = Field(description="At most 20 words: what this paper shows that bears on the annotation (finding + system).")


class GeneVerdict(BaseModel):
    verdict: Literal["supported", "mixed", "contradicted", "insufficient"]
    cell_type_evidence: Literal["same", "related", "other", "none"]
    why: str = Field(description="One sentence, at most 25 words: why the gene does or does not support the annotation (key finding and system).")
    reason: str = Field(description="1-3 sentences: what the evidence shows the gene does and why that does or does not match the annotation.")
    paper_findings: list[PaperFinding] = Field(description="One entry per paper whose evidence bears on the annotation; empty if none.")


JUDGE_VERSION = 2   # bump when GeneVerdict changes: cached answers are then re-judged


JUDGE_SYSTEM = """You are a molecular biologist judging whether literature evidence shows that a gene carries out
the biology named in a gene-program annotation. Judge the biology, not the wording: the papers never use the
annotation's exact phrase. Use only the evidence given."""


# inputs
def gene_roles(bundle):
    """[(gene, role, entry)] over program_genes, distinctive_genes, then regulators; first role wins."""
    out, seen = [], set()
    lists = [(k, bundle.get(k, [])) for k in GENE_LISTS]
    lists.append(("regulator", [r for regs in bundle.get("perturbation_regulators", {}).values() for r in regs]))
    for role, entries in lists:
        for e in entries:
            if isinstance(e, dict) and e["gene"] not in seen:
                seen.add(e["gene"])
                out.append((e["gene"], role.rstrip("s").replace("_", " "), e))
    return out


def gene_papers(gene, entry):
    """The gene's downloaded function PDFs: [{gene, pmid, pdf_path, docname, title, year, journal, doi}]."""
    papers = []
    for p in (entry.get("gene_info", {}).get(SOURCE) or {}).get("papers", []):
        if p.get("pdf_path") and Path(p["pdf_path"]).exists():
            papers.append({"gene": gene, "pmid": p.get("pmid"), "pdf_path": p["pdf_path"],
                           "docname": f"{gene}_{p.get('pmid')}", "title": p.get("title"),
                           "year": p.get("year"), "journal": p.get("journal"), "doi": p.get("doi")})
    return papers


def build_question(gene, label):
    """Only the label's biology: with the cell type or the program's full function sentence in the question,
    paper-qa answers 'insufficient' whenever no paper studied that exact cell type / gene network."""
    return f"Does {gene} drive \"{label['program_name']}\", and why?"


def build_prompt(question, gene, role, label, bundle):
    cell = f"{bundle.get('organism', 'human')} {bundle.get('cell_type') or ''}".strip()
    return "\n".join([
        f"{gene} is a {role} of a gene program annotated as \"{label['program_name']}\", found in {cell}.",
        "",
        question,
        "",
        "Answer based ONLY on the provided papers. Start with exactly two header lines:",
        "VERDICT: <supported|contradicted|mixed|insufficient>",
        "CELL_TYPE: <same|related|other|none>",
        f"VERDICT judges {gene}'s own function in ANY cell type or species: supported = the papers show {gene} "
        f"carries out or controls \"{label['program_name']}\" or a core part of it (for example one of the "
        "processes named); contradicted = they show it does "
        "not; mixed = both; insufficient = they do not address this biology at all. Do NOT answer insufficient "
        f"only because {cell} was not studied; record that in CELL_TYPE instead (same = {cell}, "
        "related = a closely related cell type, other = only unrelated cell types / species, "
        "none = no evidence). Then explain why: what the papers show the gene does, by which assay and in which "
        "cell type / species, and how that does or does not produce the annotated biology.",
    ])


def cache_key(prompt, papers, cfg):
    text = json.dumps({"prompt": prompt, "pdfs": sorted(p["pdf_path"] for p in papers), "qa": cfg.get("qa", {})},
                      sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def tag_citations(result, papers):
    """Add pmid to each citation by matching its DOI / title to the gene's papers."""
    by_doi = {p["doi"].lower(): p for p in papers if p.get("doi")}
    by_title = {(p.get("title") or "").lower().rstrip("."): p for p in papers if p.get("title")}
    for c in result.get("citations", []):
        p = by_doi.get((c.get("doi") or "").lower()) or by_title.get((c.get("title") or "").lower().rstrip("."))
        c["pmid"] = p["pmid"] if p else None


def judge(gene, role, label, bundle, result, handler, args):
    """Claude's verdict on paper-qa's evidence for one gene (dict), or None when the call failed."""
    cell = f"{bundle.get('organism', 'human')} {bundle.get('cell_type') or ''}".strip()
    evidence = "\n\n".join(f"[{i}] {e.get('title') or 'untitled'} (relevance {e.get('score')})\n{e['summary']}"
                             for i, e in enumerate(result.get("evidence") or [], 1)) or "none"
    prompt = "\n".join([
        f"Gene: {gene} ({role} of the program). Program found in: {cell}.",
        f"Annotation: \"{label['program_name']}\". {label['function_sentence']}",
        "",
        f"Evidence summaries paper-qa extracted from {gene}'s own papers:",
        evidence,
        "",
        f"paper-qa's answer:\n{result.get('answer') or 'none'}",
        "",
        f"verdict, judged in ANY cell type or species: supported = the evidence shows {gene} carries out or controls "
        "a core process named in the annotation (for example ER protein folding, chaperoning, survival under "
        "stress); mixed = some evidence for and some against (papers on unrelated topics are not 'against'); "
        f"contradicted = the evidence shows {gene} does not; insufficient = the evidence does not show {gene} doing "
        f"any core process of the annotation. cell_type_evidence: same = {cell}, related = a closely related "
        "cell type, other = only unrelated cell types / species, none = no relevant evidence.",
    ])
    out = handler.get_json(JUDGE_SYSTEM, prompt, GeneVerdict, effort=args.judge_effort)
    return out.model_dump() if out is not None else None


def clean_judge(result):
    """Drop stray tokens Claude sometimes appends after the last sentence ('...cells.eval'); a supported /
    mixed verdict has evidence somewhere, so cell_type_evidence 'none' becomes 'other'."""
    j = result.get("judge") or {}
    for k in ("why", "reason"):
        if j.get(k):
            j[k] = re.sub(r"([.!?])[A-Za-z0-9]{1,8}$", r"\1", j[k].strip())
    if result.get("verdict") in ("supported", "mixed") and result.get("cell_type_evidence") == "none":
        result["cell_type_evidence"] = "other"
    return result


def apply_judge(result, gene, role, label, bundle, handler, args):
    """result with verdict / cell_type_evidence from Claude; paper-qa's own verdict kept as paperqa_verdict."""
    result.setdefault("paperqa_verdict", result.get("verdict"))
    result.setdefault("paperqa_cell_type", result.get("cell_type_evidence"))
    j = judge(gene, role, label, bundle, result, handler, args)
    result["judge"] = {"model": args.judge_model, "version": JUDGE_VERSION, "error": j is None,
                       **({k: j[k] for k in ("why", "reason", "paper_findings")} if j else {"reason": None})}
    result["verdict"] = j["verdict"] if j else result["paperqa_verdict"]   # fall back to paper-qa's verdict
    result["cell_type_evidence"] = j["cell_type_evidence"] if j else result["paperqa_cell_type"]
    return clean_judge(result)


# per gene
async def ask_gene(prog, gene, role, entry, bundle, label, cache_dir, settings, cfg, args, docs_cache, handler):
    """(answer dict, ran paper-qa?) for one gene over its own PDFs, verdict from Claude (judge);
    docs_cache {gene: (Docs, indexed PDF paths)} grows with the gene's papers across rounds."""
    papers = gene_papers(gene, entry)
    question = build_question(gene, label)
    base = {"gene": gene, "role": role, "question": question, "n_pdfs_given": len(papers),
            "pmids_given": [p["pmid"] for p in papers]}
    if not papers:
        return {**base, "verdict": None, "status": "no_literature"}, False

    prompt = build_prompt(question, gene, role, label, bundle)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", gene)
    handler.key = f"{prog}__{safe}"   # names the judge request with --llm_backend session
    cache = cache_dir / f"{prog}__{safe}__{cache_key(prompt, papers, cfg)}.json"
    if cache.exists() and not args.overwrite:
        result = json.loads(cache.read_text())
        j = result.get("judge") or {}
        if j.get("model") != args.judge_model or j.get("version") != JUDGE_VERSION or j.get("error"):
            if not result.get("evidence"):   # answered before evidence summaries were kept: ask paper-qa again
                cache.unlink()
                return await ask_gene(prog, gene, role, entry, bundle, label, cache_dir, settings, cfg, args,
                                      docs_cache, handler)
            result = apply_judge(result, gene, role, label, bundle, handler, args)
            cache.write_text(json.dumps(result, indent=2))
        return clean_judge(result), False

    budget = handler.budget
    if budget is not None:   # paper-qa (fixed estimate) + the judge call (worst case) must fit
        judge_cost = 0.0 if args.llm_backend == "session" else claude_worst_case(args.judge_model, 20000, 4000)
        estimate = args.qa_cost_estimate + judge_cost
        if not budget.allows(estimate, args.budget_reserve_usd):
            budget.skip("driver_gene", f"paper-qa + judge for {gene}", estimate)
            return {**base, "verdict": None, "status": "budget_skipped"}, False
    docs, indexed = docs_cache.get(gene, (None, set()))
    new = [p for p in papers if p["pdf_path"] not in indexed]
    if new or docs is None:
        n_before = sum(len(t.text) for t in docs.texts) if docs is not None else 0
        docs = await qa.build_docs_from_papers(new, settings, docs)
        docs_cache[gene] = (docs, indexed | {p["pdf_path"] for p in new})
        if budget is not None:   # embeddings of the newly indexed text (not in paper-qa's session cost)
            n_new = sum(len(t.text) for t in docs.texts) - n_before
            budget.charge("driver_gene", embedding_cost(settings.embedding, n_new), f"embed {gene} PDFs", chars=n_new)
    result = await qa.answer_question(question, docs, settings, max_citations=args.max_citations, prompt=prompt)
    if budget is not None:
        budget.charge("driver_gene", result.get("cost_usd", 0.0), f"paper-qa {gene}")
    tag_citations(result, papers)
    m = _CELL_RE.search(result["answer"])
    result["cell_type_evidence"] = m.group(1).lower() if m else None
    result["short_excerpt"] = qa.first_sentence(_CELL_RE.sub("", qa._VERDICT_RE.sub("", result["answer"])).strip())
    result = {**base, **result, "status": "answered" if result["n_pdfs"] else "no_parsed_pdf",
              "cache_file": str(cache), "created": datetime.now().isoformat(timespec="seconds")}
    if result["n_pdfs"]:   # judge and cache only when paper-qa indexed at least one PDF
        result = apply_judge(result, gene, role, label, bundle, handler, args)
        cache.write_text(json.dumps(result, indent=2))
    return result, True


def build_parser():
    p = argparse.ArgumentParser(description="paper-qa, one question per gene: does the gene drive its program's final annotation (program_label.py)? Each answer uses only that gene's gene function categorization PDFs.")

    # IO
    p.add_argument("--lit_dir", required=True, type=Path, help="Literature_info_extended_PerturbNMF_Info folder: P<k>.json with gene_info.Gene_function_categorization (search_gene_function.py); updated in place.")
    p.add_argument("--label_dir", default=None, type=Path, help="Program_label folder from 3.Generate/program_label.py. Default: <lit_dir>/../Program_label.")
    p.add_argument("--cache_dir", default=None, type=Path, help="Answer cache. Default: <lit_dir>/driver_gene_paperqa_cache.")
    p.add_argument("--config", default=str(HERE / "config.yaml"), help="paper-qa settings (qa: block).")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3).")
    p.add_argument("--genes", nargs="+", default=None, help="Only ask about these genes, space separated (e.g. HSPA5 PDIA3). Default: every gene with function PDFs.")

    # answer
    p.add_argument("--max_citations", type=int, default=3, help="Top-scoring paper-qa contexts kept as citations per gene.")
    p.add_argument("--overwrite", action="store_true", help="Re-run paper-qa for genes that already have a cached answer.")

    # relabel loop
    p.add_argument("--pdf_schedule", type=int, nargs="+", default=[3, 5, 7], help="Gene function PDFs per gene in round 0, 1, 2, ... (the last value repeats); more are downloaded with search_gene_function.py before a round.")
    p.add_argument("--judge_model", default="claude-sonnet-5", help="Claude model that gives each gene's verdict from paper-qa's evidence (also used for the relabel and the gene function summaries).")
    p.add_argument("--judge_effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"], help="Effort level of the judge call.")
    p.add_argument("--min_explained", type=int, default=3, help="n: stop once at least n genes explain the label; fewer -> Claude rewrites the label from the evidence and every gene is asked again.")
    p.add_argument("--max_rounds", type=int, default=3, help="Max relabels per program (0 = ask once, never relabel).")
    p.add_argument("--explain_verdicts", nargs="+", default=["supported", "mixed"], choices=list(VERDICTS), help="Verdicts that count as the gene explaining the label.")
    p.add_argument("--label_model", default="claude-sonnet-5", help="Claude model that rewrites the label (needs ANTHROPIC_API_KEY).")
    p.add_argument("--label_effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"], help="Effort level of the relabel call.")
    p.add_argument("--top_regulator", type=int, default=6, help="Regulators passed to the relabel prompt (as in program_label.py).")
    p.add_argument("--no_expand_papers", action="store_true", help="Relabel rounds reuse round 0's PDFs instead of downloading and re-summarizing more (--pdf_schedule then only applies to round 0); much cheaper.")

    p.add_argument("--llm_backend", default="api", choices=["api", "session"], help="api = Anthropic API for the judge / relabel / paper search summaries; session = judge prompts written to --session_dir and answered by a Claude Code session (api/session_llm.py; run twice). Session runs use the papers already in the bundle (no new paper search); --judge_model is then the name recorded for the verdicts. paper-qa still uses OPENAI_API_KEY.")
    p.add_argument("--session_dir", default=None, help="Request / answer folder for --llm_backend session.")

    # spend cap
    add_budget_args(p, reserve_default=0.15)
    p.add_argument("--qa_cost_estimate", type=float, default=0.15, help="USD reserved before each paper-qa question (its real cost is charged afterwards); with --budget_dir.")
    return p


# relabel loop
def gene_evidence_lines(bundle, answers, label):
    """Per gene: function categories / summary (search_gene_function.py) + paper-qa verdict on the label."""
    lines = []
    for gene, role, entry in gene_roles(bundle):
        gf = entry.get("gene_info", {}).get(SOURCE) or {}
        a = answers.get(gene)
        if not gf.get("found") and a is None:
            continue
        line = f"- {gene} ({role}): categories {', '.join(gf.get('categories') or []) or 'none'}. {gf.get('function_summary') or ''}"
        if a is not None:
            reason = (a.get("judge") or {}).get("reason") or a.get("short_excerpt") or "no answer"
            line += f"\n  verdict on \"{label['program_name']}\": {a.get('verdict')} ({reason})"
        lines.append(line)
    return lines


def relabel(bundle, label, answers, rounds, handler, args):
    """A new ProgramLabel (dict) from the program evidence + per-gene literature, or None when the call fails."""
    sys.path.insert(0, str(HERE.parent / "3.Generate"))
    import program_label   # noqa: E402  (3.Generate)
    history = "\n".join(f"- \"{r['program_name']}\": {len(r['explained'])} gene(s) explained it "
                         f"({', '.join(r['explained']) or 'none'})" for r in rounds)
    prompt = "\n\n".join([
        program_label.build_prompt(bundle, args.top_regulator),
        "Per-gene literature (each gene's own PubMed papers; categories and summary from those papers, then "
        "whether paper-qa found that the gene explains the current label):\n"
        + "\n".join(gene_evidence_lines(bundle, answers, label)),
        f"Earlier labels, each explained by fewer than {args.min_explained} genes:\n{history}",
        f"Write a NEW label whose biology the per-gene literature above supports for at least {args.min_explained} "
        "genes; stay faithful to the program's genes and GO terms, and do not reuse an earlier label. "
        "Return program_name: a short title-case name of at most 8 words. Return function_sentence: exactly one "
        "sentence (at most 40 words) on the program's biological function in this cell type, naming 2-4 anchor genes.",
    ])
    out = handler.get_json(program_label.SYSTEM_PROMPT, prompt, program_label.ProgramLabel, effort=args.label_effort)
    if out is None:
        return None
    return {**label, "program_name": out.program_name.strip(), "function_sentence": out.function_sentence.strip(),
            "model": args.label_model, "created": datetime.now().isoformat(timespec="seconds")}


def expand_papers(bundle, gene_set, n_pdfs, lit_dir, handler):
    """Bundle with up to n_pdfs gene function PDFs per gene of gene_set (search_gene_function.categorize_gene:
    PubMed search, download, summary agent; its per-gene cache makes an unchanged n_pdfs free)."""
    import search_gene_function as sgf   # 1.Search/1.1.Search_literature
    gargs = sgf.build_parser().parse_args(["--info_dir", str(lit_dir), "--programs", "0", "--n_pdfs", str(n_pdfs)])
    cell_type = bundle.get("cell_type", "")
    terms = sgf.context_terms(cell_type, gargs.context_terms)
    blocks = {}
    for gene, entry in sgf.gene_entries(bundle).items():
        if gene in gene_set:
            block, ran = sgf.categorize_gene(gene, entry, cell_type, terms, handler, Path(lit_dir), gargs)
            if block.get("found"):
                blocks[gene] = block
                print(f"  [papers] {gene}: {sum(bool(p['pdf_path']) for p in block['papers'])} PDFs"
                      f"{'' if ran else ' (cached)'}")
    return sgf.attach_gene_block(bundle, SOURCE, blocks)


async def ask_all_genes(prog, bundle, label, genes, answers, cache_dir, settings, cfg, args, docs_cache, handler):
    """Ask every gene about label; returns (answers, n paper-qa runs). answers holds kept earlier answers."""
    n_ran = 0
    for gene, role, entry in genes:
        result, ran = await ask_gene(prog, gene, role, entry, bundle, label, cache_dir, settings, cfg, args,
                                     docs_cache, handler)
        answers[gene] = result
        n_ran += ran
        print(f"  {gene}: verdict={result.get('verdict')} cell_type={result.get('cell_type_evidence')} "
              f"PDFs={result.get('n_pdfs', 0)}/{result['n_pdfs_given']} ({'paper-qa run' if ran else 'cached'})")
    return answers, n_ran


def verdict_counts(answers):
    counts = {v: sum(a.get("verdict") == v for a in answers.values()) for v in VERDICTS}
    counts["none"] = sum(a.get("verdict") is None for a in answers.values())
    return counts


def write_label(label_path, original, final, rounds, best):
    """Best label into Program_label/P<k>.json; every label tried in label_history (selected one flagged).
    History from an earlier run of this script is kept first, marked source 'earlier run'."""
    history = [{**h, "source": "earlier run", "selected": False} for h in original.get("label_history") or []]
    for r in rounds:
        history.append({"round": r["round"], "program_name": r["program_name"],
                        "function_sentence": r["function_sentence"], "n_pdfs_per_gene": r["n_pdfs_per_gene"],
                        "n_explained": len(r["explained"]), "explained": r["explained"],
                        "selected": r["round"] == best})
    out = {k: v for k, v in final.items() if k != "label_history"}
    out["label_history"] = history
    out["relabeled_by"] = "2.Evidence_curation/query_driver_gene.py"
    label_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


async def run(args, settings, cfg):
    label_dir = args.label_dir or args.lit_dir.resolve().parent / "Program_label"
    cache_dir = args.cache_dir or args.lit_dir / "driver_gene_paperqa_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.llm_backend == "session":   # judge (and relabel) answered from a Claude Code session; no paper search
        from session_llm import session_llm
        handler = session_llm(args.session_dir, "driver_judge", args.judge_model)
    else:
        handler = make_handler(args.judge_model)   # judge, relabel and the gene function summary agent
    n_ran = 0
    for pid in dict.fromkeys(args.programs):
        prog = f"P{pid}"
        bundle_path, label_path = args.lit_dir / f"{prog}.json", label_dir / f"{prog}.json"
        missing = [str(p) for p in (bundle_path, label_path) if not p.exists()]
        if missing:
            print(f"[{prog}] skipped: missing {missing}")
            continue
        bundle, original = json.loads(bundle_path.read_text()), json.loads(label_path.read_text())
        budget = attach(handler, args, prog, "driver_gene")
        genes = [(g, r, e) for g, r, e in gene_roles(bundle)
                 if gene_papers(g, e) and (args.genes is None or g in args.genes)]
        old = bundle.get(BLOCK, {}).get(KEY, {})
        # round 0 keeps other genes' answers (--genes subset) while the annotation is unchanged
        answers = dict(old.get("genes", {})) if old.get("annotation") == original["program_name"] else {}
        gene_set = {g for g, _, _ in genes}
        label, rounds, labels, round_answers, docs_cache = original, [], [], [], {}
        while True:
            r = len(rounds)
            n_pdfs = args.pdf_schedule[min(r, len(args.pdf_schedule) - 1)]
            print(f"[{prog}] round {r}: \"{label['program_name']}\" (up to {n_pdfs} PDFs per gene)")
            round_start = datetime.now().isoformat(timespec="seconds")
            if args.llm_backend == "api" and (r == 0 or not args.no_expand_papers):
                bundle = expand_papers(bundle, gene_set, n_pdfs, args.lit_dir, handler)
            genes = [(g, ro, e) for g, ro, e in gene_roles(bundle) if g in gene_set]
            answers, ran = await ask_all_genes(prog, bundle, label, genes, answers, cache_dir, settings, cfg, args,
                                               docs_cache, handler)
            n_ran += ran
            explained = [g for g, a in answers.items() if a.get("verdict") in args.explain_verdicts]
            rounds.append({"round": r, "program_name": label["program_name"],
                           "function_sentence": label["function_sentence"], "n_pdfs_per_gene": n_pdfs,
                           "explained": explained, "n_by_verdict": verdict_counts(answers)})
            labels.append(label)
            round_answers.append(answers)
            print(f"[{prog}] round {r}: {len(explained)}/{args.min_explained} explained {explained or ''}")
            if len(explained) >= args.min_explained or len(rounds) > args.max_rounds:
                break
            if budget is not None:   # another round costs about what this one did
                round_cost = sum(e["usd"] for e in budget._read()["events"]
                                 if e["step"] == "driver_gene" and e["time"] >= round_start)
                if not budget.allows(1.2 * round_cost, args.budget_reserve_usd):
                    budget.skip("driver_gene", f"relabel round {r + 1}", 1.2 * round_cost)
                    break
            new = relabel(bundle, label, answers, rounds, handler, args)
            if new is None:
                print(f"[{prog}] relabel failed; stopping")
                break
            label, answers = new, {}
        # best round: most genes explained, earliest on ties
        best = max(range(len(rounds)), key=lambda i: (len(rounds[i]["explained"]), -i))
        label, answers = labels[best], round_answers[best]
        converged = len(rounds[best]["explained"]) >= args.min_explained
        bundle.setdefault(BLOCK, {})[KEY] = {
            "annotation": label["program_name"],
            "function_sentence": label["function_sentence"],
            "explained": rounds[best]["explained"],
            "converged": converged,
            "best_round": best,
            "n_by_verdict": rounds[best]["n_by_verdict"],
            "genes": answers,
            "rounds": rounds,
            "params": {"qa": cfg.get("qa", {}), "label_file": str(label_path.resolve()),
                       **{k: getattr(args, k) for k in ("min_explained", "max_rounds", "explain_verdicts",
                                                        "pdf_schedule", "judge_model", "label_model",
                                                        "label_effort", "top_regulator")}},
        }
        bundle_path.write_text(json.dumps(bundle, indent=2))   # after every program, so an interrupted run keeps its work
        if len(rounds) > 1:
            write_label(label_path, original, label, rounds, best)
            print(f"[{prog}] label -> {label_path} (round {best}; original \"{original['program_name']}\")")
        print(f"[{prog}] final \"{label['program_name']}\" (round {best}): {len(rounds[best]['explained'])} "
              f"explained {'(converged)' if converged else '(not converged; best round kept)'}")
    print(f"[done] {n_ran} paper-qa run(s) -> {args.lit_dir} ({BLOCK}.{KEY}); cache -> {cache_dir}")


def main():
    args = build_parser().parse_args()
    if not args.lit_dir.is_dir():
        raise FileNotFoundError(f"--lit_dir not found: {args.lit_dir}")
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    asyncio.run(run(args, qa.build_settings(cfg), cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
