"""LLM summary agents for the literature step.

  Summarize_Agent              free-text summary of one abstract (used by 1.1.Search_literature/search_*.py when a
                               mini_handler is passed)
  Summarize_GeneRIF_Agent      a gene's GeneRIF papers -> claim + experimental context per paper
  Summarize_Interaction_Agent  papers found for a regulator-gene pair -> relationship, direction, stance
  Summarize_Query_Agent        papers found for a literature_plan query -> stance on its hypothesis

The structured agents take papers as {pmid, title, year, journal, study_type, abstract, fulltext}
and a handler from api/LLM_interface.claude_api; they return a dict, or None when the call failed.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


def Summarize_Agent(text, handler):
    """
    Use LLM to summarize the text.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summarized text.
    """
    # no handler: keep the raw text (the literature agent reads abstracts itself)
    if handler is None:
        return text

    ask = getattr(handler, "get_completion", handler)
    return ask("You are a molecular biologist. Summarize this article in one paragraph: the genes studied, what "
               "they were shown to do, and the species, cell types and conditions used. Keep only key findings.",
               text) or ""


# shared
EvidenceType = Literal["knockout", "knockdown", "CRISPR_screen", "overexpression", "mutation", "binding",
                       "structural", "expression_correlation", "genetic_association", "clinical", "review", "other"]
# mirrors 2.Evidence_curation/qa.py CATEGORIES (paper-qa / GeneProgramExplorer vocabulary)
Relationship = Literal["physical_interaction", "enzymatic_modification", "transcriptional_regulation",
                       "same_pathway", "same_cellular_phenotype", "same_disease", "none"]


class ExperimentalContext(BaseModel):
    pmid: str
    evidence_type: EvidenceType
    species: Optional[str] = Field(description="Species studied, e.g. human, mouse; null if not stated.")
    cell_line_or_tissue: Optional[str] = Field(description="Cell lines / primary cells / tissues used; null if not stated.")
    condition_or_stimulus: Optional[str] = Field(description="Treatment, stimulus, disease model or condition tested; null if none.")
    context_match: Literal["same_cell_type", "related_cell_type", "unrelated"]


PAPER_RULES = """Use only the provided text (PubMed abstract, plus full-text excerpts when available): if a field
is not stated, return null rather than guessing. Return one entry per paper, using the paper's PMID exactly as given."""


def _format_papers(papers):
    blocks = []
    for p in papers:
        text = f"PMID {p['pmid']} | {p.get('title')} | {p.get('journal')} {p.get('year')} | {p.get('study_type')}\n"
        text += f"Abstract: {p.get('abstract') or 'not available'}"
        if p.get("fulltext"):
            text += f"\nFull-text excerpt:\n{p['fulltext']}"
        blocks.append(text)
    return f"Papers ({len(papers)}):\n\n" + "\n\n---\n\n".join(blocks)


def _ask(handler, system, prompt, schema, effort, max_tokens):
    out = handler.get_json(system, prompt, schema, effort=effort, max_tokens=max_tokens)
    return out.model_dump() if out is not None else None


# GeneRIF provenance (1.1.Search_literature/search_generif.py)
class GeneRIFPaper(ExperimentalContext):
    claim: str = Field(description="One sentence: what this paper shows the gene does.")
    direction: Optional[str] = Field(description="Effect of the gene, e.g. 'promotes X', 'represses Y', 'required for Z'.")
    partner_genes: list[str] = Field(description="Genes the paper links to this gene (targets, binders, pathway members).")


class GeneRIFSummary(BaseModel):
    papers: list[GeneRIFPaper]
    context_summary: str = Field(description="2-3 sentences: the gene's function in the target cell type vs elsewhere, from these papers only.")
    evidence_in_cell_type: bool = Field(description="True if at least one paper studies the gene in the target cell type or a closely related one.")


GENERIF_SYSTEM = f"""You are a molecular biologist curating gene-function evidence for a CRISPR perturbation screen.
For each paper you receive, extract what it shows about the gene and the experimental context.
{PAPER_RULES}"""


def Summarize_GeneRIF_Agent(gene, gene_desc, papers, cell_type, handler, effort="medium", max_tokens=16000):
    """Extract claim / species / cell line / condition / evidence type for a gene's GeneRIF papers.

    Args:
        gene (str): Gene symbol.
        gene_desc (str): Names/aliases and database function text (src.describe_gene).
        papers (list[dict]): {pmid, title, year, journal, study_type, abstract, fulltext}.
        cell_type (str): The screen's cell type ("same_cell_type" is judged against it).
        handler: api/LLM_interface.claude_api.

    Returns:
        dict | None: GeneRIFSummary as a dict, or None when the call failed.
    """
    prompt = (f"Gene:\n{gene_desc}\n\nTarget cell type: {cell_type or 'unspecified'}\n\n" + _format_papers(papers))
    return _ask(handler, GENERIF_SYSTEM, prompt, GeneRIFSummary, effort, max_tokens)


# gene-gene interactions (papers found for a regulator-gene pair)
class InteractionPaper(ExperimentalContext):
    relationship: Relationship = Field(description="Most direct relationship between the two genes this paper shows; bare co-mention is 'none'.")
    source_gene: Optional[str] = Field(description="Upstream gene when the paper shows a direction, else null.")
    target_gene: Optional[str] = Field(description="Downstream gene when the paper shows a direction, else null.")
    directed: bool
    sign: Literal["activates", "represses", "unclear"] = Field(description="Effect of source on target.")
    mechanism: Optional[str] = Field(description="One sentence: how the two genes are linked (e.g. m6A methylation stabilizes the mRNA).")
    stance: Literal["supports", "contradicts", "context_only"] = Field(description="Does the paper support a link between the two genes, argue against one, or only give background?")


class InteractionSummary(BaseModel):
    papers: list[InteractionPaper]
    best_relationship: Relationship = Field(description="Most direct relationship supported across the papers ('none' if no paper supports one).")
    summary: str = Field(description="2-3 sentences on how the two genes are related, from these papers only.")
    evidence_in_cell_type: bool = Field(description="True if a supporting paper is in the target cell type or a closely related one.")
    consistent_with_perturbation: Literal["yes", "no", "unclear"] = Field(description="Does the literature direction agree with the observed knockdown effect on the gene's program?")


INTERACTION_SYSTEM = f"""You are a molecular biologist curating gene-gene interaction evidence for a CRISPR perturbation screen.
A regulator was knocked down and a program containing the gene changed. For each paper you receive, classify how
the two genes are related, in which direction, by what mechanism, and in which experimental context.
Prefer the most direct relationship the paper actually shows.
{PAPER_RULES}"""


def Summarize_Interaction_Agent(gene, regulator, gene_desc, regulator_desc, papers, cell_type, regulator_stats,
                                handler, effort="medium", max_tokens=16000):
    """Relationship / direction / mechanism / stance for the papers found for a regulator-gene pair.

    Args:
        gene (str): Program gene symbol.
        regulator (str): Perturbed regulator symbol.
        gene_desc, regulator_desc (str): src.describe_gene text of each.
        papers (list[dict]): {pmid, title, year, journal, study_type, abstract, fulltext}.
        cell_type (str): The screen's cell type.
        regulator_stats (dict): {condition: {log2fc, adj_pval}} of the regulator on the program.
        handler: api/LLM_interface.claude_api.

    Returns:
        dict | None: InteractionSummary as a dict, or None when the call failed.
    """
    effect = "; ".join(f"{c}: log2FC={s.get('log2fc')}, adj_p={s.get('adj_pval')}"
                       for c, s in (regulator_stats or {}).items()) or "not given"
    prompt = (f"Program gene:\n{gene_desc}\n\nPerturbed regulator:\n{regulator_desc}\n\n"
              f"Knockdown effect of {regulator} on the program containing {gene}: {effect}\n"
              f"Target cell type: {cell_type or 'unspecified'}\n\n" + _format_papers(papers))
    return _ask(handler, INTERACTION_SYSTEM, prompt, InteractionSummary, effort, max_tokens)


# literature_plan queries (papers found for one hypothesis)
class HypothesisPaper(ExperimentalContext):
    stance: Literal["supports", "contradicts", "mixed", "context_only"] = Field(description="The paper's stance on the hypothesis.")
    finding: str = Field(description="One sentence: what the paper shows that bears on the hypothesis.")
    genes_mentioned: list[str] = Field(description="Genes of the hypothesis that the paper studies.")


class HypothesisSummary(BaseModel):
    papers: list[HypothesisPaper]
    verdict: Literal["supported", "contradicted", "mixed", "insufficient"]
    summary: str = Field(description="2-3 sentences weighing the evidence, from these papers only.")
    open_questions: list[str] = Field(description="Gaps the papers leave open that a follow-up search could target.")


QUERY_SYSTEM = f"""You are a molecular biologist weighing literature evidence for one hypothesis about a gene program
from a CRISPR perturbation screen. For each paper you receive, judge whether it supports or contradicts the hypothesis
and record its experimental context. Contradicting evidence counts as much as supporting evidence; papers that only
give background are context_only. Give the verdict 'insufficient' when no paper tests the hypothesis directly.
{PAPER_RULES}"""


def Summarize_Query_Agent(query, papers, gene_descs, cell_type, handler, effort="medium", max_tokens=16000):
    """Stance / finding / context per paper and an overall verdict for one literature_plan query.

    Args:
        query (dict): A literature_plan query (llm_query_agent.py): hypothesis, question_type, genes.
        papers (list[dict]): {pmid, title, year, journal, study_type, abstract, fulltext}.
        gene_descs (dict): {gene: src.describe_gene text} for the query's genes.
        cell_type (str): The screen's cell type.
        handler: api/LLM_interface.claude_api.

    Returns:
        dict | None: HypothesisSummary as a dict, or None when the call failed.
    """
    genes = "\n".join(gene_descs.get(g, f"- {g}") for g in query.get("genes", []))
    prompt = (f"Hypothesis ({query.get('question_type')}): {query['hypothesis']}\n\nGenes:\n{genes}\n\n"
              f"Target cell type: {cell_type or 'unspecified'}\n\n" + _format_papers(papers))
    return _ask(handler, QUERY_SYSTEM, prompt, HypothesisSummary, effort, max_tokens)
