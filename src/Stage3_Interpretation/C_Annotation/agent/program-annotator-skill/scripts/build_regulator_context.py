#!/usr/bin/env python3
"""Write Programs/P<k>/regulator_context.md: what each significant regulator is (C2), its known interactions with
the program's genes (C3), in how many other programs of the screen it is significant (C4) and the STRING links
among the regulators themselves (C5, from string_genes.json's regulator_network).

Needs regulators_sig.tsv, gene_info.jsonl, interactions.jsonl and string_genes.json of the program, and the other
programs' regulators_sig.tsv (and label.json when present). Reads one JSON object on stdin ({"program_dir": ...}) and writes
one JSON object on stdout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.program_files import (
    gene_card,
    read_json,
    read_jsonl,
    regulators,
    short,
    sibling_programs,
)  # noqa: E402
from _lib.skill_lib import SkillError, require_str, run  # noqa: E402


TYPE_TEXT = {
    "experimental": "experimental (an assay detected it)",
    "curated": "curated (a pathway / complex database lists it)",
    "predicted": "predicted (text mining, co-expression or genomic context only)",
}


def network_block(d, regs, max_links):
    """Section C5: STRING links among the significant regulators, each with its interaction type and channels."""
    lines = [
        "",
        "## C5. STRING links among the significant regulators (regulators linked to each other may act in one "
        "pathway or complex; a predicted link is not evidence that they interact)",
    ]
    rn = (read_json(d / "string_genes.json") or {}).get("regulator_network")
    if rn is None:
        return lines + [
            "Not computed (string_genes.json has no regulator_network: run 0.Prepare/refresh_string.py)."
        ]
    score = ((read_json(d / "string_genes.json") or {}).get("params") or {}).get(
        "required_score", 700
    )
    edges = [e for e in rn.get("edges") or [] if e["a"] in regs and e["b"] in regs]
    ppi = rn.get("ppi_enrichment") or {}
    head = (
        f"STRING (combined score >= {score / 1000:g}): {len(edges)} link(s) among the {rn.get('n_mapped')} "
        f"regulators found in STRING"
    )
    if ppi.get("p_value") is not None:
        head += (
            f"; {ppi.get('number_of_edges')} vs {ppi.get('expected_number_of_edges')} expected, PPI enrichment "
            f"p = {ppi['p_value']:.2g} (background: {rn.get('background')})"
        )
    lines.append(
        head
        + (
            "; by type: "
            + ", ".join(
                f"{t} {n}" for t, n in (rn.get("n_edges_by_type") or {}).items()
            )
            if edges
            else ""
        )
    )
    for e in edges[:max_links]:
        lines.append(
            f"- {e['a']} - {e['b']}: {TYPE_TEXT.get(e['type'], e['type'])}; score {e['score']:.3f}; "
            f"channels {', '.join(e['channels']) or 'none above 0.3'}"
        )
    if len(edges) > max_links:
        lines.append(
            f"- and {len(edges) - max_links} more link(s) in string_genes.json"
        )
    return lines


def handle(req):
    d = Path(require_str(req, "program_dir"))
    if not (d / "regulators_sig.tsv").exists():
        raise SkillError("not_found", f"{d} has no regulators_sig.tsv")
    regs = regulators(d)
    max_listed = int(req.get("max_listed", 6))
    lines = [f"# {d.name} — regulator context", ""]
    if not regs:
        lines.append("No regulator is significant for this program.")
        (d / "regulator_context.md").write_text("\n".join(lines) + "\n")
        return {
            "program_dir": str(d),
            "n_regulators": 0,
            "written": str(d / "regulator_context.md"),
        }

    info = {r["gene"]: r for r in read_jsonl(d / "gene_info.jsonl")}
    lines.append(
        "## C2. What each regulator is (gene databases; not evidence about this program)"
    )
    for g in regs:
        card = gene_card(info.get(g, {}))
        if not (card["name"] or card["function"]):
            lines.append(f"- {g}: no gene information retrieved")
            continue
        loc = (
            f" Location: {', '.join(card['location'][:3])}." if card["location"] else ""
        )
        lines.append(
            f"- {g} ({card['name'] or 'unnamed'}): {short(card['function'])}{loc}"
        )

    pairs = [
        p for p in read_jsonl(d / "interactions.jsonl") if p.get("regulator") in regs
    ]
    found = sorted(
        (p for p in pairs if p.get("found")),
        key=lambda p: -(p.get("curation_effort") or 0),
    )
    lines += [
        "",
        "## C3. Known interactions between the regulators and the program's genes (a curated direct "
        "interaction can support a mechanism; none found means the effect is likely indirect)",
    ]
    if not pairs:
        lines.append("Not searched (no interactions.jsonl rows).")
    else:
        lines.append(
            f"OmniPath: {len(found)} of {len(pairs)} tested regulator-gene pairs have a curated interaction; "
            "every pair not listed was tested and has none"
        )
        for p in found:
            pmids = [str(x) for x in (p.get("pmids") or [])[:3]]
            kinds = ", ".join(p.get("interaction_categories") or []) or "interaction"
            lines.append(
                f"- {p['regulator']} - {p['gene']}: {kinds}; direction "
                f"{', '.join(p.get('directions') or []) or 'unknown'}; sign "
                f"{', '.join(p.get('signs') or []) or 'unknown'}; {p.get('n_references', 0)} reference(s)"
                + (f" (PMIDs {', '.join(pmids)})" if pmids else "")
            )

    others = [o for o in sibling_programs(d) if o.name != d.name]
    lines += [
        "",
        "## C4. Regulator specificity across this screen's programs (a regulator significant in many "
        "programs is weak evidence for any one mechanism and may reflect growth or essentiality)",
    ]
    index = {}
    for o in others:
        name = (read_json(o / "label.json") or {}).get("program_name", "unlabeled")
        for g, r in regulators(o).items():
            index.setdefault(g, []).append((o.name, float(r["log2fc"]), name))
    for g in regs:
        hits = sorted(index.get(g, []), key=lambda x: -abs(x[1]))
        if not hits:
            lines.append(
                f"- {g}: significant in this program only (of {len(others) + 1})"
            )
            continue
        listed = "; ".join(
            f"{p} {name} ({fc:+.2f})" for p, fc, name in hits[:max_listed]
        )
        more = f"; and {len(hits) - max_listed} more" if len(hits) > max_listed else ""
        lines.append(
            f"- {g}: also significant in {len(hits)} of the other {len(others)} programs: {listed}{more}"
        )
    lines += network_block(d, regs, int(req.get("max_links", 30)))
    (d / "regulator_context.md").write_text("\n".join(lines) + "\n")
    return {
        "program_dir": str(d),
        "n_regulators": len(regs),
        "n_pairs_found": len(found),
        "written": str(d / "regulator_context.md"),
    }


if __name__ == "__main__":
    sys.exit(run(handle))
