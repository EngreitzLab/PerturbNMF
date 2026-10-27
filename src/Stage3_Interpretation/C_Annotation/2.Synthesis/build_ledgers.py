"""Join the per-program annotations into screen-wide ledgers, so programs can be read together instead of one at a
time. Deterministic: no model calls, no network; the same inputs give byte-identical outputs.

    python build_ledgers.py --run_dir <out_dir>/<run_name>/Annotation

Reads, per program folder (Programs/P<k>/), regulators_sig.tsv, genes_top300.tsv, label.json and interpretation.json,
and writes <run_dir>/Synthesis/:

    findings.jsonl            one row per interpretation claim (the three slots) and module, with a stable id
                              ("P10:coregulation_mechanism", "P10:module:2"), its support genes / regulators / PMIDs
                              and, for interpretations made with numbered evidence, its evidence ids ("P10:E07")
    evidence.jsonl            the numbered evidence lines of every program ("P10:E07" -> text), when present
    regulator_effects.tsv     regulator x program: condition, log2FC, adj p, activator / repressor, and the
                              finding ids that name the regulator as support
    regulator_reports.jsonl   one row per regulator over every program it significantly moves, with flags
    regulator_reports.md      the same, readable: the per-program claims citing a regulator side by side, to check
                              that one regulator is not given contradictory mechanisms in different programs
    program_neighborhoods.tsv program pairs sharing regulators (same / opposite sign) or top genes
    manifest.json             inputs (path, sha256), parameters, counts

Flags in regulator_reports: broad = significant in at least --broad_fraction of the programs (an essentiality or
growth regulator is weak evidence for any single mechanism); cited_in_several = named as support in findings of
2+ programs (review those claims together); never_cited = significant somewhere but named in no finding.
"""
import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0.Prepare"))
from program_dir import list_programs, read_tsv  # noqa: E402

SLOTS = ("upstream_trigger", "coregulation_mechanism", "cellular_output")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def program_findings(pid, interp):
    """Finding rows of one program's interpretation: its filled slots, then its modules."""
    rows = []
    for slot in SLOTS:
        s = (interp.get("interpretation") or {}).get(slot)
        if s:
            rows.append({"finding_id": f"P{pid}:{slot}", "program": f"P{pid}", "kind": slot, "claim": s["claim"],
                         "mechanism_kind": s.get("mechanism_kind", ""), "confidence": s["confidence"],
                         "support_genes": s.get("support_genes", []), "support_regulators": s.get("support_regulators", []),
                         "support_terms": s.get("support_terms", []), "pmids": s.get("pmids", []),
                         "evidence_ids": [f"P{pid}:{e}" for e in s.get("evidence_ids", [])]})
    for i, m in enumerate(interp.get("modules") or [], 1):
        rows.append({"finding_id": f"P{pid}:module:{i}", "program": f"P{pid}", "kind": "module", "claim": m["mechanism"],
                     "name": m["name"], "confidence": m["strength"], "support_genes": m.get("genes", []),
                     "support_regulators": [], "support_terms": [], "pmids": [],
                     "evidence_ids": [f"P{pid}:{e}" for e in m.get("evidence_ids", [])]})
    return rows


def main():
    p = argparse.ArgumentParser(description="Screen-wide ledgers over the per-program annotations (no model calls).")
    p.add_argument("--run_dir", required=True, help="Annotation run folder (Programs/P<k>/).")
    p.add_argument("--out_dir", default=None, help="Output folder. Default: <run_dir>/Synthesis.")
    p.add_argument("--top_gene", type=int, default=15, help="Top-loading genes per program compared in program_neighborhoods.")
    p.add_argument("--min_shared_regulators", type=int, default=2, help="Program pairs listed when they share at least this many regulators ...")
    p.add_argument("--min_gene_jaccard", type=float, default=0.1, help="... or their top genes overlap at least this much (Jaccard).")
    p.add_argument("--broad_fraction", type=float, default=0.2, help="A regulator significant in at least this fraction of programs is flagged broad.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir or run_dir / "Synthesis")
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = {int(d.name[1:]): d for d in list_programs(run_dir)}
    pids = sorted(folders)
    if not pids:
        sys.exit(f"no Programs/P<k>/ in {run_dir}")

    inputs, labels, findings, effects, top_genes, reg_sets, evidence = {}, {}, [], [], {}, {}, []
    for pid in pids:
        d = folders[pid]
        label = read(d / "label.json")
        interp = read(d / "interpretation.json")
        labels[pid] = label["program_name"] if label else ""
        for path in (d / "label.json", d / "interpretation.json"):
            if path.exists():
                inputs[str(path.relative_to(run_dir))] = sha256(path)
        current = interp if interp and interp.get("label") == labels[pid] else None
        rows = program_findings(pid, current) if current else []
        findings += rows
        cited_ev = {e for f in rows for e in f["evidence_ids"]}
        evidence += [{"evidence_id": f"P{pid}:{eid}", "program": f"P{pid}", "text": text,
                      "cited_by": sorted(f["finding_id"] for f in rows if f"P{pid}:{eid}" in f["evidence_ids"])}
                     for eid, text in ((current or {}).get("evidence") or {}).items()]
        missing = cited_ev - {e["evidence_id"] for e in evidence}
        if missing:   # program_interpretation.clean drops such ids, so this is a corrupted or hand-edited record
            sys.exit(f"P{pid}: findings cite evidence ids not in its evidence ledger: {sorted(missing)}")
        cited = {}
        for f in rows:
            for g in f["support_regulators"]:
                cited.setdefault(g, []).append(f["finding_id"])
        top_genes[pid] = {g["gene"] for g in read_tsv(d / "genes_top300.tsv")[:args.top_gene]}
        best = {}
        for r in read_tsv(d / "regulators_sig.tsv"):   # best adj p over conditions, as the earlier selection
            key = (float(r["adj_p"]), -abs(float(r["log2fc"])))
            if r["regulator"] not in best or key < best[r["regulator"]][0]:
                best[r["regulator"]] = (key, r)
        regs = [r for _, r in sorted(best.values(), key=lambda kv: kv[0])]
        reg_sets[pid] = {r["regulator"]: float(r["log2fc"]) for r in regs}
        for r in regs:
            lfc = round(float(r["log2fc"]), 2)
            effects.append({"regulator": r["regulator"], "program": f"P{pid}", "label": labels[pid],
                            "condition": r["condition"], "log2fc": lfc, "adj_pval": float(r["adj_p"]),
                            "role": "activator" if lfc < 0 else "repressor",
                            "cited_in": ",".join(cited.get(r["regulator"], []))})
    effects.sort(key=lambda e: (e["regulator"], int(e["program"][1:])))

    with (out_dir / "findings.jsonl").open("w") as fh:
        for f in findings:
            fh.write(json.dumps(f, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "evidence.jsonl").open("w") as fh:
        for e in evidence:
            fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "regulator_effects.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(effects[0]) if effects else ["regulator"], delimiter="\t")
        w.writeheader()
        w.writerows(effects)

    by_finding = {f["finding_id"]: f for f in findings}
    reports = []
    for reg in sorted({e["regulator"] for e in effects}):
        rows = [e for e in effects if e["regulator"] == reg]
        cited = [fid for e in rows for fid in e["cited_in"].split(",") if fid]
        cited_programs = sorted({fid.split(":")[0] for fid in cited}, key=lambda x: int(x[1:]))
        flags = []
        if len(rows) >= args.broad_fraction * len(pids):
            flags.append("broad")
        if len(cited_programs) >= 2:
            flags.append("cited_in_several")
        if not cited:
            flags.append("never_cited")
        reports.append({
            "regulator": reg, "n_programs": len(rows), "of_programs": len(pids), "flags": flags,
            "activator_of": [f"{e['program']} {e['label']} ({e['log2fc']:+.2f})" for e in rows if e["role"] == "activator"],
            "repressor_of": [f"{e['program']} {e['label']} ({e['log2fc']:+.2f})" for e in rows if e["role"] == "repressor"],
            "cited_in": [{"finding_id": fid, "kind": by_finding[fid]["kind"],
                          "mechanism_kind": by_finding[fid].get("mechanism_kind", ""), "claim": by_finding[fid]["claim"]}
                         for fid in cited]})
    with (out_dir / "regulator_reports.jsonl").open("w") as fh:
        for r in reports:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    md = [f"# Regulator reports\n\n{len(reports)} regulators over {len(pids)} programs. Flags: broad (significant in "
          f">= {args.broad_fraction:.0%} of programs), cited_in_several (named as support in 2+ programs: compare the "
          "claims below), never_cited.\n"]
    for r in sorted(reports, key=lambda r: (-len(r["cited_in"]), -r["n_programs"], r["regulator"])):
        md.append(f"## {r['regulator']} - {r['n_programs']} program(s)" + (f" [{', '.join(r['flags'])}]" if r["flags"] else ""))
        if r["activator_of"]:
            md.append("- activator of (knockdown lowers): " + "; ".join(r["activator_of"]))
        if r["repressor_of"]:
            md.append("- repressor of (knockdown raises): " + "; ".join(r["repressor_of"]))
        for c in r["cited_in"]:
            kind = f"{c['kind']}" + (f", {c['mechanism_kind']}" if c["mechanism_kind"] else "")
            md.append(f"  - **{c['finding_id']}** ({kind}): {c['claim']}")
        md.append("")
    (out_dir / "regulator_reports.md").write_text("\n".join(md))

    pairs = []
    for a, b in combinations(pids, 2):
        shared = sorted(set(reg_sets[a]) & set(reg_sets[b]))
        union = set(reg_sets[a]) | set(reg_sets[b])
        same = [g for g in shared if (reg_sets[a][g] < 0) == (reg_sets[b][g] < 0)]
        gene_union = top_genes[a] | top_genes[b]
        gene_jac = len(top_genes[a] & top_genes[b]) / len(gene_union) if gene_union else 0.0
        if len(shared) >= args.min_shared_regulators or gene_jac >= args.min_gene_jaccard:
            pairs.append({"program_a": f"P{a}", "label_a": labels[a], "program_b": f"P{b}", "label_b": labels[b],
                          "n_shared_regulators": len(shared),
                          "regulator_jaccard": round(len(shared) / len(union), 3) if union else 0.0,
                          "same_sign": ",".join(same), "opposite_sign": ",".join(g for g in shared if g not in same),
                          "top_gene_jaccard": round(gene_jac, 3),
                          "shared_top_genes": ",".join(sorted(top_genes[a] & top_genes[b]))})
    pairs.sort(key=lambda r: (-r["n_shared_regulators"], -r["top_gene_jaccard"], r["program_a"], r["program_b"]))
    with (out_dir / "program_neighborhoods.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]) if pairs else ["program_a"], delimiter="\t")
        w.writeheader()
        w.writerows(pairs)

    manifest = {"run_dir": str(run_dir), "params": {k: v for k, v in vars(args).items() if k not in ("run_dir", "out_dir")},
                "inputs": dict(sorted(inputs.items())),
                "counts": {"programs": len(pids), "programs_with_findings": len({f["program"] for f in findings}),
                           "findings": len(findings), "evidence_lines": len(evidence),
                           "findings_with_evidence_ids": sum(bool(f["evidence_ids"]) for f in findings),
                           "regulator_program_effects": len(effects),
                           "regulators": len(reports), "program_pairs": len(pairs),
                           "flags": {flag: sum(flag in r["flags"] for r in reports)
                                     for flag in ("broad", "cited_in_several", "never_cited")}}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
