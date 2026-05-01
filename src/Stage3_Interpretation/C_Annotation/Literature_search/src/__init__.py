"""Literature Search Agent for PerturbNMF program annotation.

Hybrid PubTator3 + LLM-generated query search with self-verification.
"""
from .llm_backend import LLMBackend
from .query_generator import generate_search_queries
from .search_engine import SearchEngine
from .paper_fetcher import PaperFetcher, PaperData, PaperCache, summarize_paper
from .verification import Verifier, VerificationResult
from .output_writer import write_program_json, write_program_markdown, write_program_html
__all__ = [
    "LLMBackend",
    "generate_search_queries",
    "SearchEngine",
    "PaperFetcher",
    "PaperData",
    "PaperCache",
    "summarize_paper",
    "Verifier",
    "VerificationResult",
    "write_program_json",
    "write_program_markdown",
    "write_program_html",
]
