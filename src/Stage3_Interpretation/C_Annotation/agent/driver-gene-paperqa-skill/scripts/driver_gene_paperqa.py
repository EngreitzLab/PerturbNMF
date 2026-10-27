#!/usr/bin/env python3
"""Ask paper-qa whether one gene drives a program's annotated biology, over that gene's own PDFs only.

paper-qa reads the PDFs, ranks evidence passages and answers with a VERDICT line (supported / contradicted /
mixed / insufficient) and a CELL_TYPE line (same / related / other / none). The calling agent judges the
returned evidence itself; this script only retrieves and answers. Reads one JSON object on stdin and writes one
JSON object on stdout (see ../SKILL.md).
"""
import asyncio
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib.skill_lib import SkillError, require_str, require_str_list, run  # noqa: E402

CELL_RE = re.compile(r"CELL_TYPE\s*[:=]\s*(same|related|other|none)", re.IGNORECASE)


def build_prompt(gene, role, annotation, cell):
    question = f"Does {gene} drive \"{annotation}\", and why?"
    return question, "\n".join([
        f"{gene} is a {role} of a gene program annotated as \"{annotation}\", found in {cell}.", "", question, "",
        "Answer based ONLY on the provided papers. Start with exactly two header lines:",
        "VERDICT: <supported|contradicted|mixed|insufficient>",
        "CELL_TYPE: <same|related|other|none>",
        f"VERDICT judges {gene}'s own function in ANY cell type or species: supported = the papers show {gene} "
        f"carries out or controls \"{annotation}\" or a core part of it (for example one of the processes named); "
        "contradicted = they show it does not; mixed = both; insufficient = they do not address this biology at all. "
        f"Do NOT answer insufficient only because {cell} was not studied; record that in CELL_TYPE instead (same = "
        f"{cell}, related = a closely related cell type, other = only unrelated cell types / species, none = no "
        "evidence). Then explain why: what the papers show the gene does, by which assay and in which cell type / "
        "species, and how that does or does not produce the annotated biology."])


def handle(req):
    gene = require_str(req, "gene")
    annotation = require_str(req, "annotation")
    pdfs = [p for p in require_str_list(req, "pdfs") if Path(p).is_file()]
    if not pdfs:
        raise SkillError("not_found", "none of the given PDFs exist")
    import yaml
    from _lib.qa import answer_question, build_docs_from_papers, build_settings
    cfg = yaml.safe_load((HERE / "paperqa_config.yaml").read_text())
    settings = build_settings(cfg)
    question, prompt = build_prompt(gene, req.get("role", "gene"), annotation, req.get("cell_type") or "the screen's cells")

    async def go():
        docs = await build_docs_from_papers([{"pdf_path": p, "title": Path(p).stem} for p in pdfs], settings)
        return await answer_question(question, docs, settings, prompt=prompt)
    result = asyncio.run(go())
    m = CELL_RE.search(result.get("answer") or "")
    return {"gene": gene, "annotation": annotation, "model": cfg["qa"]["llm"], "verdict": result["verdict"],
            "cell_type_evidence": m.group(1).lower() if m else None, "answer": result["answer"],
            "citations": result["citations"], "evidence": result.get("evidence", []),
            "n_pdfs": result["n_pdfs"], "cost_usd": result.get("cost_usd")}


if __name__ == "__main__":
    sys.exit(run(handle))
