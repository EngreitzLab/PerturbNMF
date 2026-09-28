"""Build blinded v3 annotation prompts for cNMF gene programs, one per program.

The prompt (system + user) asks for, in order:
  1. a confounder rule-out pass driven by deterministic screens (build_confounder_screens.py),
     before any biology;
  2. a layered biological reading — upstream trigger / co-regulation mechanism / cellular
     output — instead of one forced category;
  3. a plain, distinguishable label (`label_family` + `label_distinguisher`; process terms, never
     a bare gene tag; no quality words; several processes as a comma-separated list);
  4. citations selected by PMID from a supplied reference pool, never recalled.
Evidence: top 30 genes in detail plus all program genes ranked, 30 distinctive genes, every
significant regulator split by sign, STRING enrichment, screens, gene summaries.

Multi-condition screens (timepoints, stimuli, donors, genotypes, cohorts, ...): when the config
has `conditions`, each regulator is shown per condition plus a cross-condition log2FC profile,
with per-condition program activity and its peak, and a `condition_dependence` interpretation
slot. settings.condition_variable names what the conditions vary (e.g. "timepoint");
settings.condition_design = "ordered" (e.g. timepoints, a dose series) adds ordering language, the
default "unordered" adds none. A `condition_composition` confounder is added when the screens were
built with --condition-markers. Without `conditions` the single-condition prompt is produced.

TF motifs (optional): when the config names a `motif_enrichment` table (and optionally
`candidate_tfs`) from Stage 2, section E2 of the user message says how to read motif enrichment
(correlative), then, per element type (promoter / enhancer), lists the top significant motif
families (Stage 2 `motif_family`: MotifCompendium family such as KLF-SP, or the TFClass family for
HOCOMOCO), one line each: the family's best motifs with enrichment / FDR, then its candidate TFs
with extra support and their evidence tier. The system prompt is not touched. Without those keys
the prompts are unchanged.
When the table has several motif sources (`motif_source` column from Stage 2 --motif_source both:
FIMO and Fi-NeMo), each element type x source gets its own block, FIMO first; they are never pooled.

settings.effect_label (default "log2FC", the viewers' setting) replaces the word "log2FC" wherever the
prompt names the regulator effect (system rules, regulator and motif-candidate lines). The answer
schema keeps its `log2fc` fields. With the default the prompts are byte-identical.

Config: see ../configs/example_config.json and ../README.md.

Usage:
    python build_annotation_prompts.py --config my_config.json --output batch_request.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from conditions import normalise_conditions, read_condition_design, read_condition_variable  # noqa: E402
from viewer_common import DEFAULT_EFFECT_LABEL, read_effect_label  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 8192

TOP_LOADING = 30
TOP_UNIQUE = 30
MAX_GENE_SUMMARIES = 40
TOP_ENRICHMENT_PER_CATEGORY = 3
GENES_PER_TERM = 10
MAX_REFERENCES = 40
MOTIF_FAMILIES_PER_ELEMENT_TYPE = 5
MOTIFS_PER_FAMILY = 3
CANDIDATES_PER_FAMILY = 6
MOTIF_ELEMENT_TYPES = ("promoter", "enhancer")
MOTIF_SOURCE_ORDER = ("fimo", "finemo")    # Stage 2 motif_source values, shown in this order
FIMO_DATABASE_LABELS = {"motifcompendium": "FIMO/MotifCompendium", "hocomoco": "FIMO/HOCOMOCO"}
MOTIF_SOURCE_LABELS = {"fimo": FIMO_DATABASE_LABELS["hocomoco"], "finemo": "Fi-NeMo"}
MOTIF_SOURCES_NOTES = {
    "FIMO/MotifCompendium": "FIMO/MotifCompendium = motif scan with MotifCompendium database clusters",
    "FIMO/HOCOMOCO": "FIMO/HOCOMOCO = HOCOMOCO v11 motif scan",
    "Fi-NeMo": "Fi-NeMo = ChromBPNet motif calls in accessible chromatin, named by the matched database cluster",
}
MOTIFCOMPENDIUM_CLUSTER_RE = r"_\d+$"      # KLF-SP_0: a MotifCompendium cluster name
# Candidate TFs shown under each motif family (nominate_candidate_tfs.py tiers, strongest first). A motif
# cluster lists many TFs; the expressed ones say which could act here. motif_only (not expressed) is hidden.
CANDIDATE_TIERS_SHOWN = ("motif+regulator", "motif+expressed_in_program", "motif+expressed")


SYSTEM_PROMPT = """You are a {annotation_role} annotating gene expression programs derived by \
cNMF from single-cell CRISPR perturbation data in {cell_system}.

A program is a set of genes whose expression co-varies across cells. Co-variation has many \
possible causes and only some of them are pathways. So you work in two passes: first rule out \
the confounders that masquerade as biology, then read the remaining biology in layers — what \
triggers the program, what makes these particular genes move together, and what the cells end \
up doing. Those layers are usually all true at once; do not choose between them. Then give one \
plain, evidence-anchored label that reconciles them.

RULES — each of these exists because it was gotten wrong before.

1. EVIDENCE ONLY. Every gene, regulator, term, statistic and PMID you mention must appear in \
the PROGRAM EVIDENCE below. Never add one from memory.

2. NAME YOUR SUPPORT. Every claim names the specific genes, regulators or terms behind it. A \
claim resting on fewer than 2 items is a speculation and must be labeled as one.

3. CITATIONS ARE SELECTED, NOT RECALLED. Cite only PMIDs listed in the REFERENCE POOL, and only \
where the supplied evidence sentence actually states what you claim. Never write a PMID from \
memory. No citation is strictly better than a guessed one: an uncited mechanistic claim is an \
honest hypothesis; a wrong citation is a defect that survives into the paper.

4. PERTURBATION SIGN. log2FC is the effect of KNOCKING DOWN the regulator on program activity. \
Negative = knockdown lowers the program = the regulator ACTIVATES/drives it. Positive = \
knockdown raises the program = the regulator REPRESSES/brakes it. Every mechanism you propose \
must be consistent with the signs of both ends, or you must say explicitly why it is not.

5. NO SIGN-RATIONALIZATION. Do not invent a mechanistic link just because two things move in \
compatible directions. A shared direction is a correlation, not a mechanism; report it as an \
observation.

6. CELL CONTEXT. Evidence obtained in {cell_system} or a closely related system is strong. \
Evidence from an unrelated cell type counts only if the biology is canonical and \
context-independent. Always record which system a cited result came from.

7. DO NOT FORCE-FIT. Chromosomal segments, technical artifacts and essentiality drag all \
produce programs, and some programs combine several unrelated processes. Say so honestly — \
through the `coherence` field and by naming the separate processes you see — rather than \
telling a confident single story the genes do not support.

8. Respond with ONLY the JSON object specified. No preamble, no markdown fences."""


# First line of section E2: how to read the motif evidence, worded for the Stage 2 test.
MOTIF_GUIDE_TESTS = {
    "ttest": "TF motifs over-represented in the promoters / linked enhancers of the top {n_top} \
genes versus expressed genes (FDR < 0.05, enrichment > 1)",
    "correlation": "TF motifs whose count in a gene's promoter / linked enhancers correlates \
positively with the gene's program loading across expressed genes (FDR < 0.05, r > 0)",
}
MOTIF_GUIDE = """{test}. Correlative: a motif nominates a TF \
family (TFs of one family share motifs) as a candidate co-regulation mechanism; it does not show \
that TF acts here. Motifs are grouped by family, strongest first; candidate TFs are the expressed \
TFs of the family's motifs, strongest support first: motif+regulator = its knockdown also moves the \
program; motif+expressed_in_program = it is among the program genes; motif+expressed = expressed only."""
DEFAULT_MOTIF_TEST = {"method": "ttest", "n_top": 300}   # Stage 2 run_motif_enrichment.py defaults


def format_motif_guide(motif_test: dict | None = None) -> str:
    """The first line of section E2 for the Stage 2 test (`method` ttest / correlation, `n_top`)."""
    motif_test = {**DEFAULT_MOTIF_TEST, **(motif_test or {})}
    return MOTIF_GUIDE.format(test=MOTIF_GUIDE_TESTS[motif_test["method"]].format(n_top=motif_test["n_top"]))


USER_TEMPLATE = """# PROGRAM {program_id} — {dataset_name}

## Experimental context
- System: {cell_system}
- Assay: {assay}
- Factorization: cNMF, k={k}, program {program_id} of {k}
- Regulator significance: {significance_label}

## A. Top-loading genes (primary evidence, top {n_top_loading} in detail)
{top_loading_block}

## A2. Full program gene list (all {n_program_genes} genes, in loading order)
The first {n_top_loading} carry the most weight. But a theme that runs through the whole list —
many genes of one pathway spread across ranks 30-300 — is still a theme, and may be the \
program's real signal. Read the whole list.
{full_gene_list_block}

## B. Distinctive genes (high loading here, low across the other {k} programs)
{unique_block}

## C. Significant regulators (knockdown effect on program activity)
{regulator_block}

## D. STRING enrichment over the top 300 genes (cross-check only, not primary evidence)
{enrichment_block}

## E. Explanation-class screens (deterministic, computed from the data — not opinions)
{screen_block}
{motif_section}
## F. REFERENCE POOL — the only citable sources
{reference_block}

## G. Gene summaries
{gene_summary_block}

---

# TASK

Step 1 — RULE OUT THE CONFOUNDERS. Before proposing any biology, clear the non-biological and
non-specific reasons these genes might co-vary. Section E is deterministic; trust its numbers
over your intuition. (The cis-target overlap in section E is for artifact detection only: a top
gene that is also a CRISPRi target in the library is a feature of the library design, not a
finding. Use it to assess cis_target_effects; do not narrate it anywhere else.) For each item below give a status and cite the number that decided it:
  positional · cell_cycle · technical_qc · essentiality_or_growth_arrest ·
  rna_processing_or_decay · cis_target_effects · ribosome_translation_housekeeping ·
  other_technical (anything else you can argue from the evidence — name it)
Status is one of: primary_explanation, contributing, ruled_out, cannot_assess.
If any is the primary explanation, say so plainly and do NOT then build a pathway story on it.

Step 2 — READ THE BIOLOGY IN LAYERS. Do not pick one category. Fill whichever of these three
slots the evidence supports and leave the others null:
  - upstream_trigger: what turns this program on in this system
  - coregulation_mechanism: why THESE genes move together — a named transcription factor or TF
    family, a miRNA family, a chromatin complex, a shared RNA-decay route, a shared genomic
    locus, a signaling cascade, or something else you can name. This slot is deliberately open:
    if the best answer is a mechanism with no standard name, describe it.
  - cellular_output: what the cells are doing as a result
A program can legitimately fill all three. That is one program described at three levels, not
three competing hypotheses.

Step 3 — LABEL, by reconciling the slots you filled. 2-4 words strongly preferred, 6 hard
maximum. Name the biology (or the confounder) plainly. Banned in the label: the word
"program"; "regulation of"; "process"; a bare gene dump; and any word that describes the
program's quality rather than its biology — "grab bag", "grab-bag", "incoherent", "low
coherence", "heterogeneous", "mixed", "unclear", "miscellaneous". Coherence belongs in the
`coherence` field, never in the label.

  PROGRAMS WITH SEVERAL UNRELATED PROCESSES. If the genes split into separate processes with no
  single theme, name the 2-3 processes you actually see as a comma-separated list — e.g.
  "Basement membrane, lysosomal, mitochondrial OXPHOS" (capitalise only the first word). Do not
  use slashes between them. A reader will understand from the list that no
  single theme was picked. Leave `label_family` empty in that case.

  POSITIONAL AND TECHNICAL PROGRAMS. If the primary explanation from Step 1 is positional or
  technical, label it as such (the locus or cluster, plus what the genes are) — those labels
  are useful.

  DISTINGUISHABILITY. This is one of {k} programs from the same experiment, and several of them
  are near-certainly variations on the same broad theme. A label is only useful if a reader can
  tell which program it refers to. So:
  - Give a `label_family`: the broad theme another program could plausibly share
    (e.g. "Angiogenesis", "Cell cycle", "UPR"). Use the plainest standard name for it, so that
    two programs in the same family end up with the SAME family string rather than two
    paraphrases of it.
  - Give a `label_distinguisher`: what separates THIS program from a sibling in that family.
    It must be a CELL-PROCESS, PATHWAY, COMPARTMENT, PHASE or STATE term — e.g.
    "Cell cycle - G2M", "Cell cycle - G1/S", "Angiogenesis - Tip cell". Use the distinctive genes in
    section B to find which process sets this program apart, then NAME THE PROCESS, not the gene.
  - Do NOT use a bare gene symbol as the distinguisher, and do not append a gene in
    parentheses to an otherwise complete label. "Cell cycle - G2M" is correct and complete;
    "Cell cycle - G2M (CDC20)" is not. Picking one gene out of a program looks arbitrary. The
    only exception is a gene that IS the accepted name of the process ("KLF2 flow response",
    "MEG3 imprinted locus").
  - PREFER THE ESTABLISHED NAME. If a well-known term for the process fits, use it rather than
    a family-plus-modifier of your own construction.
  - Then write `label` as either the plain family name (if this program is the clear
    archetype of it and you can defend that) or `Family - Distinguisher`.
  - If nothing in the evidence defensibly separates this program from a generic member of the
    family, say so: leave `label_distinguisher` empty and set `label` to the bare family name.
    A bare family name that later needs a number appended is an honest outcome; a specific
    distinguisher you cannot defend from the genes is not.

  CONSERVATISM OVERRIDES SPECIFICITY. Only claim what the marker genes support. A precise label
  that overstates the evidence is worse than a general one that does not. Do not reach for a
  mechanism, a cell state or a pathway name to make the label sound sharper than the gene list
  warrants.

Step 4 — Report exactly which genes and regulators drove that label, and why each one.

BRIEF SUMMARY (`brief_summary`, 1-3 sentences). Report what you see: name the gene sets and the
individual genes that drove the annotation, and the regulators if any. Specifically:
  - Do NOT comment on coherence, heterogeneity, or how confident or clean the program looks.
    That goes in the `coherence` field only.
  - Do NOT remark that top genes are themselves CRISPRi targets in the screen.
  - Do NOT draw any inference from a program having few or zero significant regulators. Say
    nothing about it.
  - The one exception: if the primary explanation is positional or technical, DO explain the
    artifact reasoning (which genes, which locus, why it points to an artifact).

Step 5 — COMPETING READINGS. At least two other readings of this same gene set that the
evidence does not exclude, each with what would distinguish it. Not drawn from any list.

Step 6 — Modules, regulator mechanisms, coherence, and what would falsify each uncertain call.

# OUTPUT — JSON only, exactly this shape

{output_schema}"""


OUTPUT_SCHEMA = """{
  "program_id": <int>,
  "label": "",
  "label_family": "",
  "label_distinguisher": "",
  "label_distinguisher_evidence": "",
  "brief_summary": "",
  "confounder_assessment": [
    {"confounder": "", "status": "primary_explanation|contributing|ruled_out|cannot_assess",
     "evidence": ""}
  ],
  "interpretation": {
    "upstream_trigger": {"claim": "", "support_genes": [], "support_regulators": [],
                         "support_terms": [], "confidence": "high|medium|low", "pmids": []},
    "coregulation_mechanism": {"claim": "", "mechanism_kind": "", "support_genes": [],
                               "support_regulators": [], "support_terms": [],
                               "confidence": "high|medium|low", "pmids": []},
    "cellular_output": {"claim": "", "support_genes": [], "support_regulators": [],
                        "support_terms": [], "confidence": "high|medium|low", "pmids": []}
  },
  "label_evidence": {
    "genes": [{"symbol": "", "loading_rank": <int>, "why": ""}],
    "regulators": [{"symbol": "", "log2fc": <float>, "adj_p": <float>, "why": ""}],
    "terms": [{"term": "", "fdr": ""}]
  },
  "competing_readings": [
    {"reading": "", "why_not_excluded": "", "what_would_distinguish_it": ""}
  ],
  "coherence": "strong|partial|weak|none",
  "overview": "",
  "modules": [
    {"name": "", "genes": [], "mechanism": "",
     "support": {"genes": [], "terms": [], "pmids": []},
     "strength": "supported|suggestive|speculative"}
  ],
  "distinctive_features": "",
  "regulators": [
    {"symbol": "", "role": "activator|repressor", "log2fc": <float>, "adj_p": <float>,
     "confidence": "high|medium|low", "hypothesis": "", "sign_coherent": <bool>,
     "sign_note": "", "pmids": []}
  ],
  "citations": [
    {"pmid": "", "supports": "", "evidence_system": "", "matched_cell_context": <bool>}
  ],
  "unplaced_genes": [],
  "open_questions": [{"claim": "", "what_would_test_it": ""}]
}"""


def format_gene_table(rows: List[dict], coordinates: Dict[str, dict]) -> str:
    lines = [f"{'rank':<5} {'gene':<16} {'loading':>10}  locus"]
    for row in rows:
        info = coordinates.get(row["Name"], {})
        locus = f"{info.get('chrom', '?')} ({info.get('gene_type', '?')})"
        lines.append(f"{row['rank']:<5} {row['Name']:<16} {row['Score']:>10.4f}  {locus}")
    return "\n".join(lines)


def format_regulators(
    regulators: pd.DataFrame, string_partners: Dict[str, List[str]]
) -> str:
    significant = regulators[regulators["significant"]]
    if significant.empty:
        return (
            "No regulator reached significance for this program. Do NOT read this as evidence "
            "that the program has no upstream regulators — it means none of the tested "
            "knockdowns moved it measurably. Any regulator you propose must be labeled as "
            "inference with log2FC=N/A."
        )

    activators = significant[significant["log2_fc"] < 0].sort_values("log2_fc")
    repressors = significant[significant["log2_fc"] > 0].sort_values("log2_fc", ascending=False)

    def render(frame: pd.DataFrame) -> List[str]:
        out = []
        for _, row in frame.iterrows():
            gene = row["target_gene"]
            partners = string_partners.get(gene, [])
            suffix = f" -> STRING partners among program genes: {', '.join(partners[:8])}" if partners else ""
            out.append(
                f"- {gene}: log2FC={row['log2_fc']:+.3f}, adj p={row['adj_pval']:.2e}{suffix}"
            )
        return out

    lines = [
        f"({len(significant)} of {len(regulators)} tested knockdowns are significant; "
        "ALL significant regulators listed, ranked by effect size within each sign, adjusted p shown)",
        "",
        "Activators — knockdown LOWERS the program:",
    ]
    lines += render(activators) or ["- none"]
    lines += ["", "Repressors — knockdown RAISES the program:"]
    lines += render(repressors) or ["- none"]
    return "\n".join(lines)


CONDITIONS_SYSTEM_CONTEXT = """This is a MULTI-CONDITION SCREEN, read out by Perturb-seq separately \
in each condition. Condition variable: {variable}. Conditions{in_order}:
{condition_lines}
Every regulator effect and every activity value in the evidence was measured within one \
condition. A program may be shared by all conditions or restricted to some; a regulator may act \
on it in one condition only, in several, in all, or with opposite signs.{ordered_note}

"""

ORDERED_NOTE = """ The conditions are ordered, so a program can also track a position along \
that order."""

CONDITIONS_RULE = """8. CONDITION IS PART OF THE EVIDENCE. Never pool conditions. A regulator \
that moves the program in one condition but not another is a different claim from one that moves \
it in every condition, so always name the condition(s). Conditions can differ in power (cell \
numbers, replicate spread), so missing significance in a condition is not proof of no effect \
there: use the cross-condition log2FC profile to tell "no effect" from "same direction, below \
threshold".

9. Respond with ONLY"""

COMPOSITION_CONFOUNDER = """If any is the primary explanation, say so plainly and do NOT then build a pathway story on it.
condition_composition (multi-condition only): does the program simply read out WHICH CELLS ARE
PRESENT in a condition, rather than a regulated process within them? Decide it from the
condition-composition screen in section E and the activity in section C2. It is primary only
when all three hold: activity is concentrated in one condition{block}; the top genes are that
condition's canonical identity markers rather than a specific pathway; and nothing beyond "being
that cell type or state" is visible in the genes. Condition-restricted activity on its own is
expected for real condition-specific biology and does NOT make composition primary."""

CONDITION_DEPENDENCE_SLOT = """  - cellular_output: what the cells are doing as a result
  - condition_dependence: WHERE. The condition(s) in which the program is most active (section
    C2); the condition(s) in which its regulators act on it (section C); and, for each key
    regulator, whether its effect is condition_specific (one condition{block}, or one level of a
    factor), constitutive (same sign in most conditions), or sign_switch. Then say whether that
    pattern fits the mechanism you propose.{ordering_check} Always fill this slot: every program
    has per-condition data.
A program can legitimately fill all four. That is one program described at four levels, not
four competing hypotheses."""

ORDERING_CHECK = """ The conditions are ordered, so also check the order: a
    regulator acting before or at the peak condition can be a trigger; one acting only after the
    peak cannot."""

COMPOSITION_LABEL = """are useful.

  COMPOSITION PROGRAMS. If condition_composition is the primary explanation, label the cell type
  or state the program reads out, in plain terms.
"""

CONDITION_DEPENDENCE_SCHEMA_SLOT = """                        "support_terms": [], "confidence": "high|medium|low", "pmids": []}},
    "condition_dependence": {{"claim": "", "peak_condition": "<condition label only, e.g. {example}>",
                             "regulator_pattern": [{{"symbol": "", "conditions": [],
                                                    "pattern": "condition_specific|constitutive|sign_switch"}}],
                             "consistent_with_mechanism": <bool>, "confidence": "high|medium|low"}}
  }},"""


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"template anchor not found exactly once: {old[:70]!r}")
    return text.replace(old, new)


def escape_braces(text: str) -> str:
    """Literal text placed into a template that is later str.format()-ed."""
    return text.replace("{", "{{").replace("}", "}}")


def adapt_templates_for_conditions(
    conditions: List[dict], design: str, variable: str, composition: bool
) -> Tuple[str, str, str]:
    """The v3 system prompt, user template and schema, adapted for a multi-condition screen.

    `design` is "ordered" or "unordered"; only "ordered" adds ordering language. `composition`
    adds the condition_composition confounder (the screen was built with --condition-markers).
    """
    ordered = design == "ordered"
    condition_lines = "\n".join(f"- {c['label']}: {c['description']}" for c in conditions)
    context = CONDITIONS_SYSTEM_CONTEXT.format(
        variable=variable, in_order=", in order" if ordered else "", condition_lines=condition_lines,
        ordered_note=ORDERED_NOTE if ordered else "",
    )
    system = replace_once(SYSTEM_PROMPT, "RULES — each of these", escape_braces(context) + "RULES — each of these")
    system = replace_once(system, "8. Respond with ONLY", CONDITIONS_RULE)

    separator = " -> " if ordered else "; "
    listing = separator.join(f"{c['label']} ({c['description']})" for c in conditions)
    user = replace_once(
        USER_TEMPLATE, "- Regulator significance: {significance_label}\n",
        "- Regulator significance: {significance_label}\n"
        + escape_braces(
            f"- Multi-condition screen — condition variable: {variable}; conditions"
            f"{', in order' if ordered else ''}: {listing}; Perturb-seq read out separately in each condition\n"
        ),
    )
    user = replace_once(
        user, "## C. Significant regulators (knockdown effect on program activity)\n{regulator_block}",
        "## C. Significant regulators by condition (knockdown effect on program activity, measured "
        "separately in each condition)\n{regulator_block}\n\n## C2. Program activity by condition\n{activity_block}",
    )
    block = " or one contiguous block of conditions" if ordered else ""
    if composition:
        user = replace_once(
            user, "  other_technical (anything else", "  condition_composition · other_technical (anything else"
        )
        user = replace_once(
            user,
            "If any is the primary explanation, say so plainly and do NOT then build a pathway story on it.",
            COMPOSITION_CONFOUNDER.format(block=block),
        )
        user = replace_once(
            user, "are useful.\n\n  DISTINGUISHABILITY", COMPOSITION_LABEL + "\n  DISTINGUISHABILITY"
        )
    user = replace_once(user, "Fill whichever of these three\nslots", "Fill whichever of these four\nslots")
    user = replace_once(
        user,
        "  - cellular_output: what the cells are doing as a result\n"
        "A program can legitimately fill all three. That is one program described at three levels, not\n"
        "three competing hypotheses.",
        CONDITION_DEPENDENCE_SLOT.format(block=block, ordering_check=ORDERING_CHECK if ordered else ""),
    )
    schema = replace_once(
        OUTPUT_SCHEMA,
        """                        "support_terms": [], "confidence": "high|medium|low", "pmids": []}
  },""",
        CONDITION_DEPENDENCE_SCHEMA_SLOT.format(example=conditions[-1]["label"]),
    )
    return system, user, schema


def format_regulators_by_condition(
    regulators: pd.DataFrame, conditions: List[dict], string_partners: Dict[str, List[str]]
) -> str:
    """One labelled block per condition in config order, then each regulator's cross-condition profile."""
    significant = regulators[regulators["significant"]]
    if significant.empty:
        return (
            "No regulator reached significance for this program in any condition. Do NOT read this as "
            "evidence that the program has no upstream regulators — it means none of the tested "
            "knockdowns moved it measurably. Any regulator you propose must be labeled as "
            "inference with log2FC=N/A."
        )

    def render(frame: pd.DataFrame) -> List[str]:
        return [
            f"- {row['target_gene']}: log2FC={row['log2_fc']:+.3f}, adj p={row['adj_pval']:.2e}"
            for _, row in frame.iterrows()
        ] or ["- none"]

    lines = [
        f"({len(significant)} significant (regulator, condition) results; "
        f"{significant['target_gene'].nunique()} distinct regulators. ALL significant results "
        "are listed, condition by condition, ranked by effect size within each sign, adjusted p shown.)",
    ]
    for condition in conditions:
        in_condition = regulators[regulators["condition"] == condition["label"]]
        in_condition_significant = in_condition[in_condition["significant"]]
        lines += [
            "",
            f"### {condition['label']} — {condition['description']}: {len(in_condition_significant)} of "
            f"{len(in_condition)} tested knockdowns significant",
            "Activators — knockdown LOWERS the program:",
        ]
        lines += render(in_condition_significant[in_condition_significant["log2_fc"] < 0].sort_values("log2_fc"))
        lines += ["Repressors — knockdown RAISES the program:"]
        lines += render(
            in_condition_significant[in_condition_significant["log2_fc"] > 0].sort_values("log2_fc", ascending=False)
        )

    labels = [c["label"] for c in conditions]
    profile = regulators[regulators["target_gene"].isin(significant["target_gene"])]
    n_conditions = significant.groupby("target_gene").size()
    best_p = significant.groupby("target_gene")["adj_pval"].min()
    order = sorted(n_conditions.index, key=lambda g: (-n_conditions[g], best_p[g]))
    lines += [
        "",
        "### Cross-condition profile — every regulator significant in at least one condition",
        "log2FC in each condition, * = significant in that condition. Non-significant values are shown so that "
        '"no effect" can be told apart from "same direction, below threshold".',
    ]
    for gene in order:
        rows = profile[profile["target_gene"] == gene].set_index("condition")
        cells = []
        for label in labels:
            if label not in rows.index:
                cells.append(f"{label} n/a")
                continue
            row = rows.loc[label]
            cells.append(f"{label} {row['log2_fc']:+.2f}{'*' if row['significant'] else ''}")
        partners = string_partners.get(gene, [])
        suffix = f" -> STRING partners among program genes: {', '.join(partners[:8])}" if partners else ""
        lines.append(f"- {gene}: {'  '.join(cells)}{suffix}")
    return "\n".join(lines)


def format_activity_by_condition(activity: pd.DataFrame, conditions: List[dict]) -> str:
    by_condition = activity.set_index("condition")["mean_score"]
    total = float(sum(by_condition.get(c["label"], 0.0) for c in conditions))
    width = max(len("description"), *(len(c["description"]) for c in conditions))
    column = max(len("condition"), *(len(c["label"]) for c in conditions))
    lines = [f"{'condition':<{column}} {'description':<{width}}  {'mean score':>10}  {'share':>5}  "
             "log2(condition / mean of other conditions)"]
    for condition in conditions:
        value = float(by_condition[condition["label"]])
        others = [float(by_condition[c["label"]]) for c in conditions if c is not condition]
        other_mean = sum(others) / len(others)
        ratio = np.log2(value / other_mean) if value > 0 and other_mean > 0 else float("nan")
        lines.append(
            f"{condition['label']:<{column}} {condition['description']:<{width}}  {value:>10.5f}  "
            f"{value / total if total else 0:>5.2f}  {ratio:+.2f}"
        )
    peak = max(conditions, key=lambda c: float(by_condition[c["label"]]))
    lines.append(f"Peak condition: {peak['label']} ({peak['description']})")
    return "\n".join(lines)


def read_composition_screen(screen: dict) -> dict | None:
    """The condition-composition screen, also read from screens built before the rename."""
    composition = screen.get("condition_composition") or screen.get("stage_composition")
    if composition is None:
        return None
    overlap = composition.get("marker_overlap") or composition.get("stage_marker_overlap") or {}
    for hit in overlap.values():
        if "description" not in hit:
            hit["description"] = hit.get("stage", "")
    return composition | {"marker_overlap": overlap}


def format_composition_screen(composition: dict) -> str:
    shares = ", ".join(f"{c}={composition['share_of_total'][c]}" for c in composition["condition_order"])
    lines = [
        "",
        "condition composition (multi-condition):",
        f"  - share of total activity by condition: {shares}; peak {composition['peak_condition']} holds "
        f"{composition['peak_share']:.0%}, {composition['peak_over_second']}x the second-highest condition "
        f"and {composition['peak_over_lowest']}x the lowest",
        "  - canonical marker genes of each condition among the top genes (hypergeometric, this run's gene universe):",
    ]
    for condition in composition["condition_order"]:
        hit = composition["marker_overlap"][condition]
        genes = ", ".join(hit["genes"]) or "none"
        p_value = f", p={hit['p_value']}" if hit["p_value"] else ""
        lines.append(
            f"    {condition} {hit['description']}: {hit['n_overlap']} observed vs {hit['expected']} "
            f"expected{p_value} — {genes}"
        )
    lines.append(
        "  - READ THIS CAREFULLY: condition-restricted activity is expected of real condition-specific "
        "biology. It points to composition only together with top genes that are the condition's "
        "identity markers and nothing more specific."
    )
    return "\n".join(lines)


def format_enrichment(enrichment: pd.DataFrame) -> str:
    if enrichment.empty:
        return (
            "STRING returned NO enriched terms for this program's top 300 genes. That is itself "
            "informative: STRING annotates protein-coding genes, so an empty result is expected "
            "for a program dominated by non-coding transcripts, and uninformative otherwise."
        )
    lines = []
    for category in ("Process", "KEGG"):
        rows = enrichment[enrichment["category"] == category].nsmallest(
            TOP_ENRICHMENT_PER_CATEGORY, "fdr"
        )
        for _, row in rows.iterrows():
            genes = str(row["inputGenes"]).split("|")[:GENES_PER_TERM]
            lines.append(
                f"- {category}: {row['description']} (FDR={row['fdr']:.2e}) — {', '.join(genes)}"
            )
    return "\n".join(lines) if lines else "No Process or KEGG terms passed filtering."


def format_screens(screen: dict) -> str:
    positional = screen["positional"]
    window = positional.get("densest_10mb_window", {})
    lines = [
        "positional concentration:",
        f"  - most-loaded chromosome: {positional.get('top_chromosome')} with "
        f"{positional.get('genes_on_top_chromosome')}/{positional.get('n_genes_located')} of the top genes "
        f"(expected {positional.get('expected_on_top_chromosome')}, "
        f"{positional.get('fold_enrichment')}x, binomial p={positional.get('binomial_p')})",
        f"  - densest 10 Mb window: {window.get('chrom')}:{window.get('start')} holding "
        f"{window.get('count')} top genes — {', '.join(window.get('genes', [])[:10]) or 'none'}",
        "  - READ THIS CAREFULLY: a ~2x excess on one chromosome is common and is NOT evidence of a "
        "copy-number segment on its own. What indicates a positional program is several top genes "
        "packed into one narrow window, especially a known cluster or imprinted locus.",
        "",
        "gene biotypes among the top genes: "
        + ", ".join(f"{k}={v}" for k, v in screen["gene_biotypes"].items()),
        "",
        "symbol families (observed vs expected by chance):",
    ]
    for family, hit in screen["symbol_families"].items():
        if hit["n_overlap"]:
            lines.append(
                f"  - {family}: {hit['n_overlap']} observed vs {hit['expected']} expected — "
                f"{', '.join(hit['genes'][:10])}"
            )
    if not any(h["n_overlap"] for h in screen["symbol_families"].values()):
        lines.append("  - none")

    lines += ["", "marker-set overlap (hypergeometric, within this run's own gene universe):"]
    fired = False
    for name, hit in screen["marker_sets"].items():
        if hit["n_overlap"]:
            fired = True
            lines.append(
                f"  - {name}: {hit['n_overlap']} observed vs {hit['expected']} expected "
                f"(p={hit['p_value']}) — {', '.join(hit['genes'][:10])}"
            )
    if not fired:
        lines.append("  - none of the cell-cycle, heat-shock, ISR, UPR or interferon sets overlap")

    cis = screen["cis_targets_in_top_genes"]
    lines += [
        "",
        f"cis-target overlap: {cis['n_overlap']} of the top genes are themselves CRISPRi targets "
        f"in this screen — {', '.join(cis['genes'][:10]) or 'none'}",
        f"regulators: {screen['regulators']['n_significant']} significant of "
        f"{screen['regulators']['n_tested']} tested",
    ]
    return "\n".join(lines)


def program_number(value) -> int | None:
    """Program id as an int: 12, "12" and "K10_12" all give 12; None if there is no number."""
    match = re.search(r"(\d+)$", str(value).strip())
    return int(match.group(1)) if match else None


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_motif_test(config: dict, motif_table_path: Path) -> dict:
    """How Stage 2 tested the motifs: {"method": "ttest" | "correlation", "n_top": int}. Config keys
    `motif_method` / `motif_n_top` win; else the run's `{K}_motif_enrichment_config.yml` next to the
    table (run_motif_enrichment.py writes it as JSON); else the Stage 2 defaults (t-test, top 300)."""
    motif_test = dict(DEFAULT_MOTIF_TEST)
    stage2_config = motif_table_path.with_name(motif_table_path.stem + "_config.yml")
    if stage2_config.is_file():
        try:
            arguments = json.loads(stage2_config.read_text()).get("arguments", {})
        except ValueError:
            arguments = {}
        motif_test.update({key: arguments[name] for key, name in (("method", "motif_method"), ("n_top", "n_top"))
                           if arguments.get(name) is not None})
    motif_test.update({key: config[name] for key, name in (("method", "motif_method"), ("n_top", "motif_n_top"))
                       if config.get(name) is not None})
    if motif_test["method"] not in MOTIF_GUIDE_TESTS:
        raise ValueError(f"motif_method must be one of {sorted(MOTIF_GUIDE_TESTS)}, got {motif_test['method']!r}")
    motif_test["n_top"] = int(motif_test["n_top"])
    return motif_test


def read_motif_tables(config: dict, data: Path) -> dict:
    """The optional Stage 2 TF-motif tables named by config keys `motif_enrichment` and
    `candidate_tfs` (TSV, relative to data_dir), keyed by int `program_id`, plus `motif_test`
    (read_motif_test). {} if not configured."""
    if not config.get("motif_enrichment"):
        return {}
    motifs = pd.read_csv(data / config["motif_enrichment"], sep="\t")
    motifs["program_id"] = motifs["program"].map(program_number)
    motifs["element_type"] = motifs["element_type"].astype(str).str.lower()
    if "significant" in motifs.columns:
        motifs["significant"] = as_bool(motifs["significant"])
    else:
        motifs["significant"] = (motifs["fdr"] < 0.05) & (motifs["enrichment"] > 1)
    tables = {"motif_enrichment": motifs, "candidate_tfs": None,
              "motif_test": read_motif_test(config, data / config["motif_enrichment"])}
    if config.get("candidate_tfs"):
        candidates = pd.read_csv(data / config["candidate_tfs"], sep="\t")
        candidates["program_id"] = candidates["program"].map(program_number)
        tables["candidate_tfs"] = candidates
    return tables


def fimo_source_label(motifs: pd.DataFrame) -> str:
    """"FIMO/MotifCompendium" when most FIMO motif names are MotifCompendium clusters (``KLF-SP_0``),
    else "FIMO/HOCOMOCO" (TF names such as ``KLF4``)."""
    fimo = motifs[motifs["motif_source"].astype(str) == "fimo"] if "motif_source" in motifs.columns else motifs
    names = fimo["tf"].astype(str).drop_duplicates()
    is_cluster = names.str.contains(MOTIFCOMPENDIUM_CLUSTER_RE, regex=True)
    return FIMO_DATABASE_LABELS["motifcompendium" if len(names) and is_cluster.mean() > 0.5 else "hocomoco"]


def list_motif_sections(motifs: pd.DataFrame) -> list:
    """(key, element_type, source) for each motif block: ('promoter', 'promoter', None), ('enhancer', ...)
    for one motif source; with several (`motif_source` column), one per source x element type, FIMO
    first, keyed '<element_type>_<source>'."""
    sources = [None]
    if "motif_source" in motifs.columns and motifs["motif_source"].nunique() > 1:
        present = list(motifs["motif_source"].astype(str).unique())
        sources = sorted(present, key=lambda s: (MOTIF_SOURCE_ORDER.index(s) if s in MOTIF_SOURCE_ORDER
                                                 else len(MOTIF_SOURCE_ORDER), s))
    return [(element_type if source is None else f"{element_type}_{source}", element_type, source)
            for source in sources for element_type in MOTIF_ELEMENT_TYPES]


def candidate_entry(row: pd.Series) -> dict:
    """One candidate TF as shown: symbol, tier, the motif it rests on, knockdown or loading-rank support."""
    symbol = row["tf_gene_symbol"] if pd.notna(row.get("tf_gene_symbol")) else row["tf"]
    entry = {"tf": str(symbol), "tier": row["evidence_tier"], "motif": str(row["tf"])}
    if row["evidence_tier"] == "motif+regulator":
        entry.update(log2fc=round(float(row["knockdown_log2fc"]), 3), adj_p=float(f"{row['knockdown_fdr']:.2g}"))
    elif pd.notna(row.get("tf_program_loading_rank")):
        entry["loading_rank"] = int(row["tf_program_loading_rank"])
    return entry


def select_program_motifs(program_id: int, motifs: pd.DataFrame, candidates: pd.DataFrame | None) -> dict:
    """What prompt section E2 and the viewer show for one program. Per element type (x motif source, see
    list_motif_sections): the number of motifs tested and significant, and the top
    MOTIF_FAMILIES_PER_ELEMENT_TYPE significant motif families (Stage 2 `motif_family`; the motif name
    when the table has none), ordered by their best motif (FDR, ties higher enrichment). Each family:
    `family`, `n_significant` motifs, its best MOTIFS_PER_FAMILY motifs [tf, enrichment, FDR] and up to
    CANDIDATES_PER_FAMILY candidate TFs of those significant motifs in CANDIDATE_TIERS_SHOWN (strongest tier,
    then FDR, then loading rank; one per TF gene).
    `section_sources` maps each block key to its raw Stage 2 motif_source (None if the table has none);
    with several sources, `sections` lists [key, element_type, source label]."""
    rows = motifs[motifs["program_id"] == program_id]
    sections = list_motif_sections(motifs)
    multi_source = sections[0][2] is not None
    labels = {**MOTIF_SOURCE_LABELS, "fimo": fimo_source_label(motifs)}
    program_candidates = None
    if candidates is not None:
        program_candidates = candidates[(candidates["program_id"] == program_id)
                                        & candidates["evidence_tier"].isin(CANDIDATE_TIERS_SHOWN)].copy()
        program_candidates["tier_rank"] = program_candidates["evidence_tier"].map(
            {t: i for i, t in enumerate(CANDIDATE_TIERS_SHOWN)})
        program_candidates["symbol"] = (program_candidates["tf_gene_symbol"].fillna(program_candidates["tf"])
                                        if "tf_gene_symbol" in program_candidates else program_candidates["tf"])
        tie_break = ["tf_program_loading_rank"] if "tf_program_loading_rank" in program_candidates else []
        program_candidates = program_candidates.sort_values(["tier_rank", "fdr"] + tie_break, kind="stable")
    single_source = (str(motifs["motif_source"].iloc[0]) if "motif_source" in motifs.columns and len(motifs)
                     else None)
    selection = {"n_tested": {}, "n_significant": {}, "n_families": {}, "families": {}, "section_sources": {}}
    for key, element_type, source in sections:
        typed = rows[rows["element_type"] == element_type]
        if source is not None:
            typed = typed[typed["motif_source"].astype(str) == source]
        hits = typed[typed["significant"]].sort_values(["fdr", "enrichment"], ascending=[True, False], kind="stable")
        hits = hits.assign(family=hits["motif_family"].fillna(hits["tf"]) if "motif_family" in hits else hits["tf"])
        selection["n_tested"][key] = int(typed["tf"].nunique())
        selection["n_significant"][key] = int(len(hits))
        selection["n_families"][key] = int(hits["family"].nunique())
        selection["section_sources"][key] = source if source is not None else single_source
        families = []
        for family, members in list(hits.groupby("family", sort=False))[:MOTIF_FAMILIES_PER_ELEMENT_TYPE]:
            entry = {"family": str(family), "n_significant": int(len(members)),
                     "motifs": [[str(tf), round(float(enrichment), 2), float(f"{fdr:.2g}")] for tf, enrichment, fdr
                                in zip(members["tf"], members["enrichment"], members["fdr"])][:MOTIFS_PER_FAMILY],
                     "candidates": []}
            if program_candidates is not None:
                chosen = program_candidates[(program_candidates["element_type"] == element_type)
                                            & program_candidates["tf"].isin(set(members["tf"]))]
                if source is not None and "motif_source" in chosen:
                    chosen = chosen[chosen["motif_source"].astype(str) == source]
                chosen = chosen.drop_duplicates("symbol").head(CANDIDATES_PER_FAMILY)
                entry["candidates"] = [candidate_entry(row) for _, row in chosen.iterrows()]
            families.append(entry)
        selection["families"][key] = families
    if multi_source:
        selection["sections"] = [[key, element_type, labels.get(source, source)] for key, element_type, source in sections]
    return selection


def format_candidate(candidate: dict) -> str:
    if candidate["tier"] == "motif+regulator":
        return (f"{candidate['tf']} (motif+regulator; knockdown log2FC={candidate['log2fc']:+.2f}, "
                f"adj p={candidate['adj_p']:.1e})")
    rank = f", loading rank {candidate['loading_rank']}" if candidate["tier"] == "motif+expressed_in_program" \
        and "loading_rank" in candidate else ""
    return f"{candidate['tf']} ({candidate['tier']}{rank})"


def format_motifs(selection: dict, method: str = "ttest") -> str:
    """Section E2 body: per element type (x motif source) a header line, then one line per motif family:
    the family's best motifs with enrichment (correlation r for `method` correlation) and FDR, then its
    candidate TFs with their evidence tier."""
    if not any(selection["n_tested"].values()):
        return "No motif enrichment results for this program."
    sections = selection.get("sections") or [[et, et, None] for et in MOTIF_ELEMENT_TYPES]
    lines = []
    if "sections" in selection:
        labels = list(dict.fromkeys(label for _, _, label in sections))
        lines.append("Motif sources, tested separately: "
                     + "; ".join(MOTIF_SOURCES_NOTES.get(label, label) for label in labels) + ".")
    effect = "{tf} r={value:.2f}" if method == "correlation" else "{tf} {value:.2f}x"
    for key, element_type, source_label in sections:
        families = selection["families"][key]
        label = element_type.capitalize() + (f" ({source_label})" if source_label else "")
        n_families = selection["n_families"][key]
        header = (f"{label} ({selection['n_significant'][key]} of {selection['n_tested'][key]} motifs significant"
                  + (f" in {n_families} {'family' if n_families == 1 else 'families'}" if n_families else "")
                  + (f"; top {len(families)} shown" if n_families > len(families) else "") + ")")
        if not families:
            lines.append(f"{header}: none")
            continue
        lines.append(f"{header}:")
        for family in families:
            motifs = ", ".join(effect.format(tf=tf, value=value) + f" FDR={fdr:.1e}"
                               for tf, value, fdr in family["motifs"])
            more = family["n_significant"] - len(family["motifs"])
            motifs += f" (+{more} more)" if more > 0 else ""
            names = [tf for tf, _, _ in family["motifs"]]
            prefix = "" if names == [family["family"]] else f"{family['family']}: "
            candidates = "; ".join(format_candidate(c) for c in family["candidates"])
            lines.append(f"- {prefix}{motifs}" + (f" | candidate TFs: {candidates}" if candidates else ""))
    if not any(family["candidates"] for families in selection["families"].values() for family in families):
        lines.append("Candidate TFs: none (no enriched motif lists an expressed TF)")
    return "\n".join(lines)


def format_reference_pool(context: dict, allowed_genes: set, excluded_pmids: frozenset = frozenset()) -> str:
    """One line per citable PMID, deduplicated across genes.

    The model may cite only these. A pooled, deduplicated list (rather than per-gene snippets)
    makes the constraint checkable: every PMID in the output must appear here. PMIDs in
    `excluded_pmids` (retracted or non-resolving, from flag_retracted_pool_pmids.py) are dropped.
    """
    pool: Dict[str, dict] = {}
    for gene, snippets in (context.get("evidence_snippets") or {}).items():
        if gene not in allowed_genes:
            continue
        for snippet in snippets:
            match = re.search(r"\(PMID:(\d+)\)", snippet)
            if not match:
                continue
            pmid = match.group(1)
            if pmid in excluded_pmids:
                continue
            sentence = re.sub(r"\s*\(PMID:\d+\)", "", snippet).strip()
            entry = pool.setdefault(pmid, {"genes": set(), "sentence": sentence})
            entry["genes"].add(gene)
            if len(sentence) > len(entry["sentence"]):
                entry["sentence"] = sentence

    if not pool:
        return (
            "No literature was retrieved for this program's genes. Cite nothing — an uncited "
            "hypothesis is the correct output here."
        )

    lines = [
        f"({len(pool)} papers retrieved for this program's top and distinctive genes. "
        "These PMIDs are the ONLY ones you may cite.)",
        "",
    ]
    for pmid, entry in sorted(pool.items(), key=lambda kv: -len(kv[1]["genes"]))[:MAX_REFERENCES]:
        genes = ", ".join(sorted(entry["genes"]))
        lines.append(f"- PMID:{pmid} [{genes}] {entry['sentence'][:320]}")
    return "\n".join(lines)


def format_gene_summaries(context: dict, genes_by_priority: List[str]) -> str:
    """Summaries for the highest-priority genes only; all 300 would not fit in one prompt."""
    summaries = context.get("gene_summaries") or {}
    source = "Harmonizome" if str(context.get("gene_summaries_source", "")).lower() == "harmonizome" else "Entrez (NCBI)"
    lines = [f"(source: {source}; the top-loading and most distinctive genes only)"]
    for gene in [g for g in genes_by_priority if g in summaries][:MAX_GENE_SUMMARIES]:
        lines.append(f"- {gene}: {str(summaries[gene])[:400]}")
    return "\n".join(lines) if len(lines) > 1 else "None available."


def format_full_gene_list(frame: pd.DataFrame) -> str:
    return ", ".join(f"{rank}.{name}" for rank, name in enumerate(frame["Name"], start=1))


def name_effect(text: str, effect_label: str) -> str:
    """The prompts call the regulator effect "log2FC"; name it `effect_label` instead (settings.effect_label).
    Applied only to text this script writes (templates, regulator and motif blocks), never to the
    literature evidence. The default leaves the text unchanged."""
    return text if effect_label == DEFAULT_EFFECT_LABEL else text.replace(DEFAULT_EFFECT_LABEL, effect_label)


def build_prompt(program_id: int, resources: dict, settings: dict) -> dict:
    loading = resources["loading"]
    frame = loading[loading["program_id"] == program_id].sort_values("Score", ascending=False)
    top_rows = frame.head(TOP_LOADING).assign(rank=range(1, min(TOP_LOADING, len(frame)) + 1))
    top_genes = top_rows["Name"].tolist()

    unique_ranked = frame.sort_values("UniquenessScore", ascending=False)
    unique_frame = unique_ranked[~unique_ranked["Name"].isin(top_genes)].head(TOP_UNIQUE)
    unique_rows = unique_frame.assign(rank=range(1, len(unique_frame) + 1))
    allowed_genes = set(top_genes) | set(unique_rows["Name"])
    # Interleave so the summary cap keeps both the strongest and the most distinctive genes.
    unique_genes = unique_rows["Name"].tolist()
    genes_by_priority = [g for pair in zip(top_genes, unique_genes) for g in pair]
    genes_by_priority += top_genes[len(unique_genes):] + unique_genes[len(top_genes):]

    context = resources["ncbi"].get(str(program_id), {})
    validation = context.get("regulator_validation") or {}
    string_partners: Dict[str, List[str]] = {}
    for bucket in ("positive_regulators", "negative_regulators"):
        for entry in validation.get(bucket, []):
            string_partners[entry.get("regulator", "")] = [
                f"{i.get('target', i.get('target_gene', '?'))}({i.get('score', 0)})"
                for i in entry.get("string_interactions", [])
                if i.get("score", 0) >= 400
            ]

    regulators = resources["regulators"]
    program_regulators = regulators[regulators["program_id"] == program_id]
    enrichment = resources["enrichment"]
    program_enrichment = enrichment[enrichment["program_id"] == program_id]

    motif_section = ""
    if resources.get("motif_enrichment") is not None:
        selection = select_program_motifs(program_id, resources["motif_enrichment"], resources.get("candidate_tfs"))
        motif_test = resources.get("motif_test") or DEFAULT_MOTIF_TEST
        motif_section = (f"\n## E2. TF motifs in program promoters/enhancers (correlative)\n"
                         f"{format_motif_guide(motif_test)}\n{format_motifs(selection, motif_test['method'])}\n")

    conditions = resources.get("conditions")
    condition_blocks = {}
    if conditions:
        screen = resources["screens"][str(program_id)]
        composition = read_composition_screen(screen)
        system_template, user_template, output_schema = adapt_templates_for_conditions(
            conditions, read_condition_design(settings), read_condition_variable(settings), composition is not None
        )
        activity = resources["activity"]
        condition_blocks["activity_block"] = format_activity_by_condition(
            activity[activity["program_id"] == program_id], conditions
        )
        regulator_block = format_regulators_by_condition(program_regulators, conditions, string_partners)
        screen_block = format_screens(screen).replace(" tested", " tested (significant in at least one condition)")
        if composition is not None:
            screen_block += format_composition_screen(composition)
    else:
        system_template, user_template, output_schema = SYSTEM_PROMPT, USER_TEMPLATE, OUTPUT_SCHEMA
        regulator_block = format_regulators(program_regulators, string_partners)
        screen_block = format_screens(resources["screens"][str(program_id)])

    effect_label = read_effect_label(settings)
    system_template = name_effect(system_template, escape_braces(effect_label))
    regulator_block = name_effect(regulator_block, effect_label)
    motif_section = name_effect(motif_section, effect_label)

    user = user_template.format(
        program_id=program_id,
        dataset_name=settings["dataset_name"],
        cell_system=settings["cell_system"],
        assay=settings["assay"],
        k=settings["k"],
        significance_label=settings["significance_label"],
        n_top_loading=len(top_rows),
        n_program_genes=len(frame),
        top_loading_block=format_gene_table(top_rows.to_dict("records"), resources["coordinates"]),
        full_gene_list_block=format_full_gene_list(frame),
        unique_block=format_gene_table(unique_rows.to_dict("records"), resources["coordinates"]),
        regulator_block=regulator_block,
        enrichment_block=format_enrichment(program_enrichment),
        screen_block=screen_block,
        motif_section=motif_section,
        reference_block=format_reference_pool(context, allowed_genes, resources["excluded_pmids"]),
        gene_summary_block=format_gene_summaries(context, genes_by_priority),
        output_schema=output_schema,
        **condition_blocks,
    )

    system = system_template.format(
        annotation_role=settings["annotation_role"], cell_system=settings["cell_system"]
    )

    return {
        "custom_id": f"topic_{program_id}_annotation",
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    data = Path(config["data_dir"])

    loading = pd.read_csv(data / config["gene_loading"])
    if "program_id" not in loading.columns:
        loading = loading.rename(columns={"RowID": "program_id"})

    # A multi-condition screen has one regulator table keyed by `condition`; a single condition has one flat table.
    regulators = pd.read_csv(data / config.get("regulators_by_condition", config.get("regulators")))
    regulators["significant"] = (
        regulators["significant"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )

    enrichment = pd.read_csv(data / config["enrichment"])

    coordinates = {}
    with (data / config["gene_coordinates"]).open() as handle:
        for line in handle:
            name, chrom, start, end, _strand, gene_type = line.rstrip("\n").split("\t")
            coordinates.setdefault(name, {"chrom": chrom, "gene_type": gene_type})

    resources = {
        "loading": loading,
        "regulators": regulators,
        "enrichment": enrichment,
        "coordinates": coordinates,
        "screens": json.loads((data / config["screens"]).read_text()),
        "ncbi": json.loads((data / config["ncbi_context"]).read_text()),
        "excluded_pmids": frozenset(),
    }
    if config.get("excluded_pmids"):
        excluded = json.loads((data / config["excluded_pmids"]).read_text())
        resources["excluded_pmids"] = frozenset(excluded["retracted"] + excluded["unresolved"])
    if config.get("conditions"):
        resources["conditions"] = normalise_conditions(config["conditions"])
        resources["activity"] = pd.read_csv(data / config["program_activity"])
    resources.update(read_motif_tables(config, data))

    requests = [build_prompt(pid, resources, config["settings"]) for pid in config["programs"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"requests": requests}, indent=2), encoding="utf-8")
    for request in requests:
        chars = len(request["params"]["messages"][0]["content"])
        print(f"{request['custom_id']}: {chars} chars")
    print(f"wrote {len(requests)} prompts -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
