"""Build one blinded annotation prompt per regulator group (the ProgramAnnotatorV3 recipe, for
groups of perturbed genes instead of gene programs).

A group is a set of CRISPRi targets whose knockdowns shift the cell's gene programs in the same
way (define_regulator_groups.py). The annotator is asked, in order:
  1. rule out the reasons unrelated genes co-cluster — a shared fitness / stress response,
     a shared delay of differentiation, promoter neighbours (the deterministic screen already
     excluded the clear cases; the flags are shown), noise at weak effects;
  2. the shared function: a complex, a pathway, a process the members have in common;
  3. why the group forms HERE — which programs it moves in this system, read through those
     programs' labels;
  4. a role for every member: core_explained (its known function is the shared function),
     consistent (compatible, not established), or unexplained — with a hypothesis for each
     unexplained member, because an unexpected member of a coherent group is the finding;
  5. a plain label, citations selected from the reference pool only, competing readings.
Members the promoter screen excluded are listed only as "do not interpret".

The answer uses the same keys as a v3 program answer where the meaning carries over (`label`,
`label_family`, `label_distinguisher`, `label_evidence.regulators`, `regulators[]` with a
confidence), so the shared citation pass (annotator_core, `--subject regulator_group`) runs on it
without group-specific code. Members are called regulators there: they are the perturbed genes.

Output: batch_request.json in the Anthropic batch format that
annotator_core/split_prompts_for_blinded_dispatch.py explodes into one directory per group.

Usage:
    python build_group_prompts.py --config my_group_config.json --output batch_request.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = """You are a {annotation_role} annotating groups of perturbed genes from a \
single-cell CRISPRi Perturb-seq screen in {cell_system}.

Each group is a set of genes whose knockdowns change the cell's gene programs in the same way: \
their effect profiles across the screen's cNMF programs correlate, stably under resampling of \
the programs. Co-clustering has several possible causes and only some of them are shared \
biology. So you work in two passes: first rule out the reasons unrelated genes end up \
together, then say what the members share, why that shows up in THIS system, and what each \
member's place in it is.

RULES — each of these exists because it was gotten wrong before.

1. EVIDENCE ONLY. Every gene, program, term, complex, statistic and PMID you mention must \
appear in the GROUP EVIDENCE below. Never add one from memory. Genes listed under EXCLUDED \
MEMBERS are not members: never interpret them, never give them a role.

2. NAME YOUR SUPPORT. Every claim names the members, complexes, terms or programs behind it. A \
claim resting on fewer than 2 items is a speculation and must be labeled as one.

3. CITATIONS ARE SELECTED, NOT RECALLED. Cite only PMIDs listed in the REFERENCE POOL, and only \
where the supplied sentence actually states what you claim. Never write a PMID from memory. No \
citation is strictly better than a guessed one.

4. PERTURBATION SIGN. log2FC is the effect of KNOCKING DOWN a member on a program's activity. \
Negative = knockdown lowers the program = the member is needed for it. Positive = knockdown \
raises the program = the member restrains it. A group moves programs together; read the \
shared direction, and say so when a member moves them the other way.

5. CORRELATION IS NOT MECHANISM. Two members moving the same programs is the reason they are \
grouped, not evidence that they interact. Interaction needs a complex, a STRING edge with \
physical evidence, or a cited paper.

6. CELL CONTEXT. Evidence from {cell_system} or a closely related system is strong. Evidence \
from an unrelated cell type counts only if the biology is canonical and context-independent.

7. DO NOT FORCE-FIT. Groups can be held together by a shared fitness defect or a shared \
differentiation delay rather than by shared biology, and some groups combine two unrelated \
sets. Say so honestly through `coherence` and `confounder_assessment`, rather than telling a \
single story the members do not support.

8. UNEXPECTED MEMBERS ARE THE POINT. A member whose known function is unrelated to the others \
is not an error to explain away: mark it `unexplained` and give a concrete, testable \
hypothesis for why its knockdown phenocopies the rest.

9. Respond with ONLY the JSON object specified. No preamble, no markdown fences."""


USER_TEMPLATE = """# REGULATOR GROUP {group_id} — {dataset_name}

## Experimental context
- System: {cell_system}
- Assay: {assay}
{conditions_block}- Programs: cNMF, k={k}{program_note}
- Program-effect significance: {significance_label}

## A. How this group was formed
{group_block}

## B. Members
Columns: role (core = stable under resampling; peripheral = less stable; rescued = shares a
curated complex with core members and correlates significantly with the group, but was not
clustered in), stability (bootstrap co-assignment with the rest of the group), r (correlation
of its effect profile with the group mean), reliability (share of its profile that is signal,
not noise), effect strength (tertile across the screen), connected to (other members it shares
a STRING edge, curated complex or enriched term with — none is a hint, not a verdict),
promoter caveat (from the deterministic promoter screen).
{member_block}

## C. EXCLUDED MEMBERS — do not interpret
These clustered with the group, but a neighbouring promoter could explain it: their CRISPRi
guides can silence the neighbour instead. They are not members for this annotation.
{excluded_block}

## D. What the group does to the cell's programs (effect signature)
The programs the members move most, by the members' mean log2FC{label_note}.
{signature_block}

## E. Curated complexes with >= 2 members (CORUM / ComplexPortal / SIGNOR)
{complex_block}

## F. STRING among members
{string_block}

## G. Functional enrichment of the members (STRING; background = the {n_targets} genes perturbed in this screen)
{enrichment_block}

## H. REFERENCE POOL — the only citable sources
{reference_block}

## I. Gene summaries
{summary_block}

---

# TASK

Step 1 — RULE OUT THE CONFOUNDERS. For each item give a status and the evidence that decided it:
  generic_fitness_or_stress (the members share a growth, viability or stress-response defect
    rather than a function: many unrelated essential genes moving proliferation / stress programs)
  differentiation_delay (the members all slow or block the {condition_word} progression, which
    moves every stage program at once)
  promoter_neighbour (section B/C caveats; excluded members are already gone)
  weak_effect_noise (members with low reliability or weak effects held together loosely)
  other (anything else you can argue — name it)
Status is one of: primary_explanation, contributing, ruled_out, cannot_assess. If one is the
primary explanation, say so and label the group by it — do not build a shared-function story on it.

Step 2 — SHARED FUNCTION. What do the members have in common? Name the kind: protein complex,
pathway, shared process, or none evident. Name the members, complexes, terms behind it.

Step 3 — WHY HERE. Why does knocking down these genes produce THIS effect signature in
{cell_system}? Read section D through the program labels: which programs, which direction,
at which {condition_word}, and what that says about the shared function's role in this system.

Step 4 — ROLE OF EVERY MEMBER. List every member of section B exactly once in `regulators`:
  core_explained  its known function IS the shared function (a subunit of the complex, a
                  component of the pathway)
  consistent      compatible with the shared function but not established by the evidence
  unexplained     no evidence here links it to the others — give a concrete `hypothesis` for why
                  its knockdown phenocopies them and `what_would_test_it`
`confidence` (high|medium|low) is your confidence in the stated role or hypothesis. Rescued
members are members; say whether the evidence supports their inclusion.

Step 5 — LABEL. 2-4 words preferred, 6 maximum; name the shared biology (or the confounder)
plainly: e.g. "SAGA acetyltransferase complex", "Hippo pathway kinases", "Microprocessor".
Banned in the label: "group", "cluster", "module", "regulators", "program", "regulation of",
a bare gene list, and any word about quality ("heterogeneous", "mixed", "unclear",
"miscellaneous", "grab bag"). Coherence goes in `coherence`, never in the label.
`label_family` is the broad theme another group could share (e.g. "Chromatin remodelling",
"Translation initiation"); `label_distinguisher` what separates this one — a process, complex,
pathway or state, never a bare gene symbol. If several unrelated sets are held together, name
the 2-3 as a comma-separated list and leave `label_family` empty.

Step 6 — `label_evidence.regulators`: the members the label rests on, and why each.

Step 7 — COMPETING READINGS: at least two other readings the evidence does not exclude, each
with what would distinguish it. Then open questions.

BRIEF SUMMARY (`brief_summary`, 1-3 sentences): what the members share and what their
knockdown does here; name the members and programs. No comment on coherence or confidence.

# OUTPUT — JSON only, exactly this shape

{output_schema}"""


OUTPUT_SCHEMA = """{
  "group_id": <int>,
  "label": "",
  "label_family": "",
  "label_distinguisher": "",
  "brief_summary": "",
  "confounder_assessment": [
    {"confounder": "", "status": "primary_explanation|contributing|ruled_out|cannot_assess", "evidence": ""}
  ],
  "shared_function": {"claim": "", "kind": "protein_complex|pathway|shared_process|none_evident",
                      "support_members": [], "support_complexes": [], "support_terms": [],
                      "confidence": "high|medium|low", "pmids": []},
  "why_here": {"claim": "",
               "programs": [{"program_id": <int>, "condition": "", "direction": "up|down", "reading": ""}],
               "confidence": "high|medium|low", "pmids": []},
  "regulators": [
    {"symbol": "", "role": "core_explained|consistent|unexplained", "confidence": "high|medium|low",
     "hypothesis": "", "what_would_test_it": "", "pmids": []}
  ],
  "label_evidence": {
    "regulators": [{"symbol": "", "why": ""}],
    "complexes": [""],
    "terms": [{"term": "", "fdr": ""}]
  },
  "competing_readings": [{"reading": "", "why_not_excluded": "", "what_would_distinguish_it": ""}],
  "coherence": "strong|partial|weak|none",
  "citations": [{"pmid": "", "supports": "", "evidence_system": "", "matched_cell_context": <bool>}],
  "open_questions": [{"claim": "", "what_would_test_it": ""}]
}"""


def format_members(members: list[dict]) -> str:
    lines = ["| member | role | stability | r | reliability | effect strength | connected to | promoter caveat |",
             "|---|---|---|---|---|---|---|---|"]
    for m in members:
        role = m["role"] + (f" (via {m['rescued_via']})" if m.get("rescued_via") else "")
        stability = "—" if m.get("stability") is None else f"{m['stability']:.2f}"
        caveat = "; ".join(m["promoter"]["reasons"]) if m["promoter"]["decision"] == "flag" else ""
        connected = ", ".join(m["connected_to"]) or "none"
        lines.append(f"| {m['gene']} | {role} | {stability} | {m['r_to_centroid']:.2f} | {m['reliability']:.2f} | "
                     f"{m['strength_tier']} ({m['n_significant']} significant program effects) | {connected} | {caveat} |")
    return "\n".join(lines)


def format_signature(signature: list[dict], multi_condition: bool) -> str:
    lines = []
    for s in signature:
        where = f"program {s['program_id']}" + (f" at {s['condition']}" if multi_condition else "")
        label = f" — \"{s['program_label']}\"" if s["program_label"] else ""
        lines.append(f"- {where}{label}: mean log2FC {s['mean_log2fc']:+.2f}; "
                     f"{s['members_significant_same_direction']} of {s['members']} members significant in this direction")
    return "\n".join(lines) or "- none"


def format_complexes(complexes: list[dict]) -> str:
    if not complexes:
        return "- none"
    return "\n".join(
        f"- {c['name']} ({c['id']}; {', '.join(s for s in c['sources'] if s in ('CORUM', 'ComplexPortal', 'SIGNOR'))}): "
        f"{len(c['members_in_group'])} members here ({', '.join(c['members_in_group'])}); "
        f"{len(c['members_perturbed'])} of its {c['size']} subunits were perturbed in the screen"
        for c in complexes)


def format_string(edges: list[dict], ppi: dict | None) -> str:
    lines = []
    if ppi:
        lines.append(f"PPI enrichment vs the screened genes: {ppi.get('number_of_edges')} edges observed, "
                     f"{ppi.get('expected_number_of_edges')} expected, p = {ppi.get('p_value')}")
    lines += [f"- {e['a']} – {e['b']}: combined {e['score']:.2f}"
              + (f", physical {e['physical_score']:.2f}" if e["physical_score"] else "") for e in edges[:40]]
    return "\n".join(lines) or "- no edges at combined score >= 0.4"


def format_enrichment(terms: list[dict]) -> str:
    return "\n".join(f"- [{t['category']}] {t['description']} ({t['term']}): FDR {t['fdr']:.1e}; "
                     f"{', '.join(t['genes'])} ({t['number_of_genes']} of {t['number_of_genes_in_background']} "
                     f"in background)" for t in terms) or "- no term at FDR < 0.05"


def format_pool(pool: list[dict]) -> str:
    if not pool:
        return ("No paper in the pool names two members together. Cite nothing — an uncited claim is "
                "the correct output here.")
    lines = [f"({len(pool)} papers. These PMIDs are the ONLY ones you may cite.)", ""]
    lines += [f"- PMID:{p['pmid']} ({p['year']}) [{', '.join(p['genes'])}] {p['sentence'][:320]}" for p in pool]
    return "\n".join(lines)


def build_prompt(evidence: dict, settings: dict, conditions: list[dict], n_targets: int, labelled: bool) -> dict:
    multi = len(conditions) > 1
    conditions_block = ("- Conditions, in order: " + "; ".join(f"{c['label']} = {c['stage']}" for c in conditions) + "\n") if multi else ""
    group_block = (f"{len(evidence['members'])} members after the promoter screen; stability {evidence['stability']:.2f} "
                   f"(mean bootstrap co-assignment of core members); mean correlation of effect profiles "
                   f"{evidence['mean_raw_r']:.2f} ({evidence['mean_corrected_r']:.2f} after correcting for noise); "
                   f"members by effect strength: {evidence['strength_tiers']}. Grouped by shared nearest "
                   f"neighbours of noise-corrected effect-profile correlation, so a group of weak regulators "
                   f"can be as coherent as a group of strong ones.")
    excluded_block = "\n".join(f"- {e['gene']}: {'; '.join(e['reasons'])}" for e in evidence["excluded"]) or "- none"
    summaries = "\n".join(f"- {m['gene']}: {m['summary'] or '(no summary)'}" for m in evidence["members"])
    user = USER_TEMPLATE.format(
        group_id=evidence["group_id"], dataset_name=settings.get("dataset_name", ""),
        cell_system=settings["cell_system"], assay=settings.get("assay", ""), conditions_block=conditions_block,
        k=settings.get("k", "?"),
        program_note="; a member's effect on a program is measured at each condition" if multi else "",
        significance_label=settings.get("significance_label", "adjusted p < 0.05"),
        group_block=group_block, member_block=format_members(evidence["members"]),
        excluded_block=excluded_block,
        label_note=", with each program's label from the program annotation" if labelled else "",
        signature_block=format_signature(evidence["signature"], multi),
        complex_block=format_complexes(evidence["complexes"]),
        string_block=format_string(evidence["string_edges"], evidence["ppi_enrichment"]),
        n_targets=n_targets, enrichment_block=format_enrichment(evidence["enrichment"]),
        reference_block=format_pool(evidence["reference_pool"]), summary_block=summaries,
        condition_word=settings.get("condition_word", "differentiation" if multi else "condition"),
        output_schema=OUTPUT_SCHEMA,
    )
    system = SYSTEM_PROMPT.format(annotation_role=settings.get("annotation_role", "cell biologist"),
                                  cell_system=settings["cell_system"])
    return {"custom_id": f"topic_{evidence['group_id']}_annotation",
            "params": {"model": settings.get("model", "claude-sonnet-5"), "max_tokens": 8000, "system": system,
                       "messages": [{"role": "user", "content": user}]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="see ../configs/example_config.json")
    parser.add_argument("--groups", help="comma list of group ids (default: every group with evidence)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    wanted = {int(g) for g in args.groups.split(",")} if args.groups else None

    config = json.loads(args.config.read_text())
    groups_dir = Path(config["groups_dir"])
    if not groups_dir.is_absolute():
        groups_dir = args.config.parent / groups_dir
    payload = json.loads((groups_dir / "group_evidence.json").read_text())
    n_targets = sum(1 for _ in open(Path(config["targets"]) if Path(config["targets"]).is_absolute()
                                    else args.config.parent / config["targets"])) - 1
    conditions = config.get("conditions") or [{"label": "all", "stage": config["settings"]["cell_system"]}]
    requests = []
    for gid, evidence in sorted(payload["groups"].items(), key=lambda kv: int(kv[0])):
        if wanted and int(gid) not in wanted:
            continue
        if evidence.get("skipped"):
            print(f"G{gid}: skipped — {evidence['skipped']}")
            continue
        request = build_prompt(evidence, config["settings"], conditions, n_targets, payload.get("programs_labelled", False))
        requests.append(request)
        print(f"G{gid}: {len(request['params']['messages'][0]['content'])} chars")
    args.output.write_text(json.dumps({"requests": requests}, indent=1))
    print(f"wrote {len(requests)} prompts -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
