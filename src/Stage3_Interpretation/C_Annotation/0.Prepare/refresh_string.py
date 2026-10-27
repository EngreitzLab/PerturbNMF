"""Re-run the STRING step of existing program folders: rewrite string_sig.tsv and string_genes.json (now with the
regulator network: the STRING links among the significant regulators, typed experimental / curated / predicted, and
the regulators' PPI enrichment) without touching the other prepare outputs, the plan or later steps.

    python refresh_string.py --run_dir <out_dir>/<run_name>/Interpretation/Annotation [--programs 1 2 3] \
        [--rebuild_context auto|all|none]

Uses the settings prepare_programs.py recorded in <run_dir>/Programs/meta.json (taxid, FDR, required score,
background) and each folder's genes_top300.tsv and regulators_sig.tsv, so the gene lists are the ones the program
was annotated with. The background is the spectra genes of the recorded MuData (string_background "spectra") or
STRING's genome.

--rebuild_context then rewrites regulator_context.md (adding section C5) and, when evidence.json exists,
evidence.md / evidence.json:
    auto  only where the interpretation step is not done (default; nothing already interpreted goes stale)
    all   every folder with regulator_context.md. Section C5 renumbers the evidence ids after it; a done
          interpretation whose cited ids now hold other text is reported as failing (its ids still exist, so
          status.py --recheck alone would not notice): re-run its interpretation step
    none  STRING files only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "agent" / "program-annotator-skill" / "scripts"
sys.path[:0] = [str(HERE), str(HERE.parent / "agent" / "string-skill" / "scripts"), str(HERE.parent / "1.Annotate")]
from program_dir import list_programs, read_plan, read_tsv  # noqa: E402
from string_enrichment import enrich_program, map_string_ids, write_tables  # noqa: E402


def spectra_genes(meta):
    """Gene names of the spectra of the MuData recorded in meta.json."""
    import mudata as mu
    from program_inputs import load_gene_spectra
    keys = meta["keys"]
    prog = mu.read_h5ad(meta["inputs"]["mdata_path"], mod=keys["prog_key"])
    return [str(g) for g in load_gene_spectra(prog, keys["loadings_key"], keys["gene_name_key"]).columns]


def run_skill(script, program_dir):
    """One program-annotator script, as the agent runs it (JSON on stdin; each skill has its own _lib)."""
    out = subprocess.run([sys.executable, str(SCRIPTS / script)], input=json.dumps({"program_dir": str(program_dir)}),
                         capture_output=True, text=True)
    reply = json.loads(out.stdout or "{}")
    if not reply.get("ok"):
        raise RuntimeError(f"{script} failed for {program_dir}: {reply.get('error') or out.stderr[-500:]}")
    return reply


def cited_ids(obj):
    """Every evidence id cited anywhere in interpretation.json."""
    if isinstance(obj, dict):
        return {x for k, v in obj.items() for x in (map(str, v) if k == "evidence_ids" and isinstance(v, list)
                                                    else cited_ids(v))}
    if isinstance(obj, list):
        return {x for v in obj for x in cited_ids(v)}
    return set()


def moved_ids(d, old_ledger):
    """Evidence ids cited by interpretation.json whose text differs from before the rebuild."""
    interp = d / "interpretation.json"
    if not interp.exists() or old_ledger is None:
        return []
    new = json.loads((d / "evidence.json").read_text())["evidence"]
    return sorted(i for i in cited_ids(json.loads(interp.read_text())) if old_ledger.get(i) != new.get(i))


def step_status(d):
    return {s.id: s.status for s in read_plan(d).steps} if (d / "plan.md").exists() else {}


def build_parser():
    p = argparse.ArgumentParser(description="Rewrite the STRING outputs (with the regulator network) of existing program folders.")
    p.add_argument("--run_dir", required=True, help="Annotation run folder (Programs/meta.json and Programs/P<k>/).")
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids. Default: every program folder.")
    p.add_argument("--rebuild_context", default="auto", choices=["auto", "all", "none"], help="Rewrite regulator_context.md (+ evidence) after STRING: auto = only where interpretation is not done, all = everywhere, none = STRING files only.")
    return p


def main():
    args = build_parser().parse_args()
    root = Path(args.run_dir) / "Programs"
    meta = json.loads((root / "meta.json").read_text())
    params = meta["params"]
    if params.get("skip_string"):
        sys.exit("this run was prepared with --skip_string; nothing to refresh")
    dirs = [d for d in list_programs(args.run_dir) if args.programs is None or int(d.name[1:]) in args.programs]
    if not dirs:
        sys.exit(f"no program folder under {root}")

    tables = {d: (read_tsv(d / "genes_top300.tsv"), read_tsv(d / "regulators_sig.tsv")) for d in dirs}
    names = [g["gene"] for genes, _ in tables.values() for g in genes]
    names += [r["regulator"] for _, regs in tables.values() for r in regs]
    background_genes = spectra_genes(meta) if params.get("string_background") == "spectra" else []
    ids = map_string_ids(names + background_genes, params["taxid"], root / "STRING_id_map.json")
    background = sorted({ids[g] for g in background_genes if g in ids}) or None
    print(f"[STRING] {len(ids):,} names mapped; background "
          + (f"{len(background):,} spectra genes" if background else "genome"))

    from validate_program import validate_step
    summary = []
    for d in dirs:
        genes, regs = tables[d]
        old_terms = (d / "string_sig.tsv").read_text() if (d / "string_sig.tsv").exists() else None
        candidates = {g["gene"]: "program gene" for g in genes}
        for r in regs:
            candidates.setdefault(r["regulator"], "regulator")
        result = enrich_program([g["gene"] for g in genes], ids, background, candidates,
                                {"taxid": params["taxid"], "max_fdr": params["fdr"],
                                 "required_score": params["string_required_score"]},
                                regulators=[r["regulator"] for r in regs])
        write_tables(result, d)
        prog = json.loads((d / "program.json").read_text())
        prog.setdefault("counts", {})["string_sig"] = len(result["terms"])
        (d / "program.json").write_text(json.dumps(prog, indent=2))
        rn = result["regulator_network"]
        row = {"program": d.name, "regulators": rn["n_regulators"], "links": len(rn["edges"]),
               "by_type": rn["n_edges_by_type"], "ppi_p": rn["ppi_enrichment"].get("p_value"),
               "terms_changed": old_terms is not None and old_terms != (d / "string_sig.tsv").read_text(),
               "context": "kept"}

        status = step_status(d)
        rebuild = (d / "regulator_context.md").exists() and (
            args.rebuild_context == "all"
            or (args.rebuild_context == "auto" and status.get("interpretation") != "done"))
        errors = []
        if rebuild:
            run_skill("build_regulator_context.py", d)
            row["context"] = "rebuilt"
            if (d / "evidence.json").exists() and (d / "label.json").exists():
                old_ledger = json.loads((d / "evidence.json").read_text()).get("evidence")
                run_skill("build_evidence.py", d)
                row["context"] = "rebuilt + evidence"
                moved = moved_ids(d, old_ledger)
                if moved:
                    errors.append(f"interpretation.json cites evidence ids whose text changed ({', '.join(moved[:10])}"
                                  f"{', ...' if len(moved) > 10 else ''}): re-run the interpretation step")
        errors += validate_step(d, "prepare")
        if status.get("interpretation") == "done" and row["context"] != "kept":
            errors += validate_step(d, "interpretation")
        row["errors"] = errors
        summary.append(row)
        print(f"  {d.name}: {row['regulators']} regulators, {row['links']} STRING links {row['by_type']}, "
              f"PPI p = {row['ppi_p']}; context {row['context']}"
              + ("; STRING terms CHANGED" if row["terms_changed"] else "")
              + (f"  [checks FAILED: {'; '.join(errors)[:200]}]" if errors else ""))

    (root / "string_refresh.json").write_text(json.dumps(summary, indent=2))
    n_fail = sum(bool(r["errors"]) for r in summary)
    print(f"[done] {len(summary)} program(s) refreshed; {n_fail} with failing checks -> {root / 'string_refresh.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
