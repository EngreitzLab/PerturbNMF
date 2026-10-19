"""LLM summary agents for the literature step.

  Summarize_Agent              free-text summary of one abstract (used by 1.1.Search_literature/search_*.py when a
                               mini_handler is passed)
  Summarize_GeneRIF_Agent      a gene's GeneRIF papers -> claim + experimental context per paper
  Summarize_GeneFunction_Agent a gene's PubMed function papers -> claim + role per paper, 1-3 function categories
  LiteratureSearchResult       output schema of the run_literature_search.py agent: the papers it found, each
                               read (PDF via read_pdf, else abstract) and summarized, plus an overall summary

Summarize_GeneRIF_Agent / Summarize_GeneFunction_Agent take papers as {pmid, title, year, journal, study_type, abstract, fulltext}
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


# gene function categorization (1.1.Search_literature/search_gene_function.py)
FunctionCategory = Literal["transcription_regulation", "chromatin_epigenetics", "RNA_processing_translation",
                           "signaling", "cell_cycle_DNA_repair", "apoptosis_survival", "metabolism",
                           "protein_homeostasis", "trafficking_transport", "cytoskeleton_motility", "adhesion_ECM",
                           "immune_inflammation", "development_differentiation", "angiogenesis_vascular",
                           "stress_response", "unknown"]


class GeneFunctionPaper(ExperimentalContext):
    claim: str = Field(description="One sentence: what this paper shows the gene does.")
    function_role: str = Field(description="One sentence: the gene's molecular / cellular role in this paper (e.g. 'kinase phosphorylating SMAD2 downstream of TGF-beta').")


class GeneFunctionSummary(BaseModel):
    papers: list[GeneFunctionPaper]
    categories: list[FunctionCategory] = Field(description="1-3 function categories supported by these papers, most supported first; ['unknown'] if none.")
    primary_category: FunctionCategory = Field(description="The single best-supported category (first of categories).")
    function_summary: str = Field(description="2-3 sentences: the gene's function from these papers only, noting the target cell type when studied.")
    evidence_in_cell_type: bool = Field(description="True if at least one paper studies the gene in the target cell type or a closely related one.")


GENE_FUNCTION_SYSTEM = f"""You are a molecular biologist categorizing gene function for a CRISPR perturbation screen.
For each paper you receive, extract what it shows the gene does and the experimental context; then assign the gene
1-3 function categories that these papers support (use 'unknown' only when they support none).
{PAPER_RULES}"""


def Summarize_GeneFunction_Agent(gene, gene_desc, papers, cell_type, handler, effort="medium", max_tokens=16000):
    """Per-paper claim / role / context and gene-level function categories for a gene's PubMed papers.

    Args:
        gene (str): Gene symbol.
        gene_desc (str): Names/aliases and database function text (src.describe_gene).
        papers (list[dict]): {pmid, title, year, journal, study_type, abstract, fulltext}.
        cell_type (str): The screen's cell type ("same_cell_type" is judged against it).
        handler: api/LLM_interface.claude_api.

    Returns:
        dict | None: GeneFunctionSummary as a dict, or None when the call failed.
    """
    prompt = (f"Gene:\n{gene_desc}\n\nTarget cell type: {cell_type or 'unspecified'}\n\n" + _format_papers(papers))
    return _ask(handler, GENE_FUNCTION_SYSTEM, prompt, GeneFunctionSummary, effort, max_tokens)


# run_literature_search.py agent output (papers found, read and summarized for one search task)
class PaperSummary(ExperimentalContext):
    pmid: Optional[str] = Field(description="PMID exactly as a search tool returned it; null if none.")
    title: str
    doi: Optional[str] = Field(description="DOI as a search tool returned it; null if none.")
    url: Optional[str]
    pdf_file: Optional[str] = Field(description="File name of the downloaded PDF (as listed by read_pdf); null if no PDF.")
    read_from: Literal["fulltext", "abstract"] = Field(description="fulltext if you read the PDF with read_pdf, else abstract.")
    summary: str = Field(description="2-4 sentences: genes studied, what they were shown to do, and how (assay, model).")
    finding: str = Field(description="One sentence: what the paper shows that bears on the search goal.")
    stance: Literal["supports", "contradicts", "mixed", "context_only"] = Field(description="The paper's stance on the hypothesis / goal.")
    genes_mentioned: list[str] = Field(description="Genes from the task that the paper studies.")


class WebHit(BaseModel):
    title: str
    url: str
    snippet: Optional[str]


class LiteratureSearchResult(BaseModel):
    queries: list[str] = Field(description="Every query you ran, in order.")
    rung_reached: Optional[int] = Field(description="plan mode: 1-based ladder rung that first returned relevant papers; null otherwise.")
    papers: list[PaperSummary]
    web_hits: list[WebHit]
    summary: str = Field(description="3-5 sentences weighing what these papers establish for the search goal, from these papers only.")
    open_questions: list[str] = Field(description="Gaps the papers leave open that a follow-up search could target.")
    notes: str
