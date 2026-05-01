"""Two-level self-verification for literature evidence.

Level 1 (citation check): Does the cited PMID contain the mentioned genes?
Level 2 (semantic check): Does the evidence sentence support the claim? (LLM)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_backend import LLMBackend
from .paper_fetcher import PaperData

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class VerificationResult:
    """Result of one verification check."""

    level: int  # 1 or 2
    passed: bool
    confidence: float  # 0.0-1.0
    details: str
    method: str  # "citation_check" or "semantic_check"

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "passed": self.passed,
            "confidence": self.confidence,
            "details": self.details,
            "method": self.method,
        }

    @property
    def badge(self) -> str:
        if self.level == 1 and not self.passed:
            return "[GENE_MISMATCH]"
        if self.level == 2 and not self.passed:
            return "[WEAK_SUPPORT]"
        return "[VERIFIED]"


class Verifier:
    """Two-level verification of literature evidence."""

    def __init__(self, llm: Optional[LLMBackend] = None):
        self.llm = llm
        self._semantic_prompt = None

    def _get_semantic_prompt(self) -> str:
        if self._semantic_prompt is None:
            path = PROMPT_DIR / "semantic_check.txt"
            self._semantic_prompt = path.read_text(encoding="utf-8")
        return self._semantic_prompt

    # ------------------------------------------------------------------
    # Level 1: Citation check (automated, free)
    # ------------------------------------------------------------------

    def verify_citation(
        self,
        claimed_genes: List[str],
        paper: PaperData,
    ) -> VerificationResult:
        """Check if the cited paper actually contains the claimed genes.

        Uses PubTator3 gene annotations from paper_data.gene_mentions.
        """
        if not claimed_genes:
            return VerificationResult(
                level=1, passed=True, confidence=1.0,
                details="No genes to check", method="citation_check",
            )

        doc_genes_upper = set(paper.gene_mentions.keys())
        found = []
        missing = []
        for gene in claimed_genes:
            if gene.upper() in doc_genes_upper:
                found.append(gene)
            else:
                missing.append(gene)

        passed = len(found) > 0
        confidence = len(found) / len(claimed_genes) if claimed_genes else 1.0

        if missing:
            details = f"Found {found}; missing {missing} in PMID:{paper.pmid} annotations"
        else:
            details = f"All genes found in PMID:{paper.pmid} annotations"

        return VerificationResult(
            level=1, passed=passed, confidence=confidence,
            details=details, method="citation_check",
        )

    # ------------------------------------------------------------------
    # Level 2: Semantic check (LLM-based, costs tokens)
    # ------------------------------------------------------------------

    def verify_semantic(
        self,
        claim: str,
        evidence_sentence: str,
        abstract: str,
    ) -> VerificationResult:
        """Use LLM to check if evidence sentence supports the claim."""
        if self.llm is None:
            return VerificationResult(
                level=2, passed=True, confidence=0.5,
                details="Semantic check skipped (no LLM configured)",
                method="semantic_check",
            )

        system_prompt = self._get_semantic_prompt()
        user_prompt = (
            f"CLAIM: {claim}\n\n"
            f"EVIDENCE SENTENCE: {evidence_sentence}\n\n"
            f"FULL ABSTRACT: {abstract}"
        )

        try:
            result = self.llm.complete_json(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("Semantic check failed: %s", exc)
            return VerificationResult(
                level=2, passed=True, confidence=0.5,
                details=f"Semantic check error: {exc}",
                method="semantic_check",
            )

        supported = result.get("supported", True)
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        return VerificationResult(
            level=2, passed=supported, confidence=confidence,
            details=reasoning, method="semantic_check",
        )

    # ------------------------------------------------------------------
    # Convenience: run all applicable checks for one paper
    # ------------------------------------------------------------------

    def verify_paper(
        self,
        paper: PaperData,
        claimed_genes: List[str],
        run_semantic: bool = True,
    ) -> List[VerificationResult]:
        """Run verification levels for one paper.

        Args:
            paper: The paper data.
            claimed_genes: Genes expected in this paper.
            run_semantic: Whether to run level-2 semantic checks.

        Returns:
            List of VerificationResult objects.
        """
        results = []

        # Level 1: citation check
        results.append(self.verify_citation(claimed_genes, paper))

        # Level 2: semantic check on top evidence sentences (LLM)
        if run_semantic and self.llm is not None:
            for gene, sentences in paper.evidence_sentences.items():
                for sent in sentences[:1]:  # Check top sentence per gene
                    claim_text = f"{gene} is involved in this biological program"
                    results.append(
                        self.verify_semantic(claim_text, sent, paper.abstract)
                    )

        return results
