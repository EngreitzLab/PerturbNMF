"""Measure whether a target's CRISPRi guides also knock down the genes next to its promoter.

For every grouped regulator (regulator_groups.json) and every gene with a TSS within --window bp
of where its guides act (guide coordinates from an IGVF guide table (--guide-table), else positions
parsed from hCRISPRi-v2-style guide names, else the target's TSSs), compare expression in cells carrying only that target's guides with
non-targeting-control cells, within each condition, then combine across conditions:
  log2fc   cell-weighted mean of per-condition log2(mean_target / mean_NTC)
  p_value  Stouffer-combined Welch t-test
The target itself is measured too (its own knockdown), so the confound screen can tell "the
neighbour went down and the target did not" from "both went down".

Per-condition comparison matters: a regulator that changes differentiation changes many genes
in trans, and pooled cells would read that as knockdown. A strong cis knockdown (the gene falls
as far as the target does) is still the thing to look for; modest drops can be trans.

Reads the h5mu (or h5ad) directly with h5py, streaming the expression matrix in row blocks, so it
needs neither scanpy nor the matrix in memory. Any normalised or raw count matrix works: only
ratios of means are used.

Output: knockdown.tsv (target, gene, relation self|neighbour, distance, log2fc, p_value, q_value,
n_target_cells, ntc_mean, conditions).

Usage:
    python measure_neighbour_knockdown.py --groups regulator_groups/regulator_groups.json \
        --gene-coordinates gene_coordinates.tsv --h5mu cNMF.h5mu --condition-key day \
        --output regulator_groups/knockdown.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm, ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from gene_coordinates import (  # noqa: E402
    guide_positions_on_this_assembly, load_gene_tss, load_guide_table_positions, parse_guide_position,
    promoter_neighbours,
)

ROW_BLOCK = 8000
MAX_NTC_CELLS_PER_CONDITION = 4000
PSEUDO_FRACTION = 0.01  # of the NTC mean: caps log2FC near -6.6 for a gene silenced to zero


def decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def read_strings(group: h5py.Group, key: str) -> list[str]:
    """An AnnData string column: a plain string array or a categorical (codes + categories)."""
    node = group[key]
    if isinstance(node, h5py.Dataset):
        return decode(node[:])
    categories = np.array(decode(node["categories"][:]), dtype=object)
    return list(categories[node["codes"][:]])


def read_csr(node: h5py.Group) -> csr_matrix:
    shape = tuple(node.attrs["shape"])
    return csr_matrix((node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape)


def index_column(frame: h5py.Group) -> list[str]:
    return read_strings(frame, frame.attrs["_index"])


def load_guides(handle: h5py.File, guide_prefix: str):
    names = decode(handle[f"{guide_prefix}/uns/guide_names"][:])
    targets = decode(handle[f"{guide_prefix}/uns/guide_targets"][:])
    assignment = read_csr(handle[f"{guide_prefix}/obsm/guide_assignment"])
    return names, targets, assignment


def cell_targets(assignment: csr_matrix, targets: list[str]) -> np.ndarray:
    """The single target of each cell ('' when it carries guides for zero or several targets)."""
    target_array = np.array(targets, dtype=object)
    out = np.full(assignment.shape[0], "", dtype=object)
    for cell in range(assignment.shape[0]):
        start, end = assignment.indptr[cell], assignment.indptr[cell + 1]
        hit = {target_array[j] for j, v in zip(assignment.indices[start:end], assignment.data[start:end]) if v > 0}
        if len(hit) == 1:
            out[cell] = hit.pop()
    return out


def stream_columns(node: h5py.Group, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Dense rows x columns block of a CSR matrix stored in HDF5, read in row blocks."""
    indptr = node["indptr"][:]
    column_position = {c: i for i, c in enumerate(columns)}
    wanted = np.zeros(int(node.attrs["shape"][1]), dtype=bool)
    wanted[columns] = True
    row_position = {r: i for i, r in enumerate(rows)}
    out = np.zeros((len(rows), len(columns)), dtype=np.float32)
    rows_sorted = np.sort(rows)
    for block_start in range(0, len(rows_sorted), ROW_BLOCK):
        block = rows_sorted[block_start:block_start + ROW_BLOCK]
        lo, hi = indptr[block[0]], indptr[block[-1] + 1]
        data = node["data"][lo:hi]
        indices = node["indices"][lo:hi]
        for r in block:
            a, b = indptr[r] - lo, indptr[r + 1] - lo
            idx, val = indices[a:b], data[a:b]
            keep = wanted[idx]
            for c, v in zip(idx[keep], val[keep]):
                out[row_position[r], column_position[c]] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", required=True, type=Path, help="define_regulator_groups.py regulator_groups.json")
    parser.add_argument("--gene-coordinates", required=True, type=Path)
    parser.add_argument("--h5mu", required=True, type=Path, help=".h5mu (or .h5ad with --expression-prefix '')")
    parser.add_argument("--expression-prefix", default="mod/rna", help="HDF5 path of the AnnData holding expression")
    parser.add_argument("--guide-prefix", default="mod/cNMF", help="HDF5 path of the AnnData holding guide_assignment")
    parser.add_argument("--gene-name-key", default=None, help="var column with gene symbols (default: the var index)")
    parser.add_argument("--condition-key", default=None, help="obs column to stratify by (e.g. day)")
    parser.add_argument("--ntc-pattern", default=r"(?i)^(?:non[-_]?targeting|NTC|safe[-_]?targeting)")
    parser.add_argument("--guide-table", type=Path,
                        help="IGVF 'guide RNA sequences' table with guide coordinates (overrides positions in guide names)")
    parser.add_argument("--window", type=int, default=3000, help="bp between a guide site / TSS and a neighbour TSS")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text())
    regulators = sorted({m["gene"] for g in groups["groups"] for m in g["members"]})
    tss = load_gene_tss(args.gene_coordinates)

    with h5py.File(args.h5mu, "r") as handle:
        guide_names, guide_targets, assignment = load_guides(handle, args.guide_prefix)
        guide_positions: dict[str, list[int]] = {}
        for name, target in zip(guide_names, guide_targets):
            parsed = parse_guide_position(name)
            if parsed:
                guide_positions.setdefault(target, []).append(parsed[1])
        guide_positions, off_assembly = guide_positions_on_this_assembly(guide_positions, tss)
        if args.guide_table:  # GRCh38 coordinates with chromosomes: checked by chromosome instead
            guide_positions, off_assembly = load_guide_table_positions(args.guide_table, tss), []
        print(f"guide positions used for {len(guide_positions)} targets", flush=True)
        if off_assembly:
            print(f"guide positions of {len(off_assembly)} targets are not at their TSS in "
                  f"{args.gene_coordinates.name} (another assembly?); using their TSSs instead")
        pairs = [(r, r, 0) for r in regulators]
        for r in regulators:
            pairs += [(r, n["gene"], n["distance"]) for n in promoter_neighbours(r, tss, guide_positions, args.window)]

        expression = handle[args.expression_prefix]
        var = expression["var"]
        symbols = read_strings(var, args.gene_name_key) if args.gene_name_key else index_column(var)
        column_of = {}
        for i, s in enumerate(symbols):
            column_of.setdefault(s, i)
        genes = sorted({g for _, g, _ in pairs if g in column_of})
        columns = np.array([column_of[g] for g in genes], dtype=int)

        per_cell_target = cell_targets(assignment, guide_targets)
        is_ntc = np.array([bool(re.search(args.ntc_pattern, t)) if t else False for t in per_cell_target])
        obs = expression["obs"]
        conditions = np.array(read_strings(obs, args.condition_key)) if args.condition_key else np.full(len(per_cell_target), "all")
        rng = np.random.default_rng(args.seed)
        ntc_cells = {}
        for condition in np.unique(conditions):
            cells = np.flatnonzero(is_ntc & (conditions == condition))
            if len(cells) > MAX_NTC_CELLS_PER_CONDITION:
                cells = rng.choice(cells, MAX_NTC_CELLS_PER_CONDITION, replace=False)
            ntc_cells[condition] = cells
        target_cells = {r: np.flatnonzero(per_cell_target == r) for r in regulators}
        rows = np.unique(np.concatenate([*ntc_cells.values(), *target_cells.values()]))
        print(f"{len(regulators)} regulators, {len(pairs) - len(regulators)} neighbour pairs, "
              f"{len(genes)} measured genes; reading {len(rows)} cells", flush=True)
        dense = stream_columns(expression["X"], rows, columns)

    row_of = {r: i for i, r in enumerate(rows)}
    gene_col = {g: i for i, g in enumerate(genes)}
    records = []
    for target, gene, distance in pairs:
        record = {"target": target, "gene": gene, "relation": "self" if gene == target else "neighbour",
                  "distance": distance, "log2fc": np.nan, "p_value": np.nan, "n_target_cells": 0,
                  "ntc_mean": np.nan, "conditions": 0, "measured": gene in gene_col}
        if gene in gene_col:
            weights, fcs, zs, ntc_means = [], [], [], []
            for condition, ntc in ntc_cells.items():
                cells = target_cells[target][conditions[target_cells[target]] == condition]
                if len(cells) < 5 or len(ntc) < 20:
                    continue
                a = dense[[row_of[c] for c in cells], gene_col[gene]]
                b = dense[[row_of[c] for c in ntc], gene_col[gene]]
                pseudo = max(PSEUDO_FRACTION * b.mean(), 1e-6)
                fcs.append(np.log2((a.mean() + pseudo) / (b.mean() + pseudo)))
                test = ttest_ind(a, b, equal_var=False)
                p = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
                zs.append(np.sign(fcs[-1]) * norm.isf(max(p, 1e-300) / 2))
                weights.append(len(cells))
                ntc_means.append(b.mean())
            if weights:
                w = np.array(weights, dtype=float)
                z = float(np.sum(np.sqrt(w) * np.array(zs)) / np.sqrt(np.sum(w)))
                record.update(log2fc=float(np.average(fcs, weights=w)), p_value=float(2 * norm.sf(abs(z))),
                              n_target_cells=int(w.sum()), ntc_mean=float(np.mean(ntc_means)), conditions=len(weights))
        records.append(record)
    table = pd.DataFrame(records)
    tested = table["p_value"].notna()
    table["q_value"] = np.nan
    if tested.any():
        p = table.loc[tested, "p_value"].to_numpy()
        order = np.argsort(p)
        ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
        q = np.minimum.accumulate(ranked[::-1])[::-1]
        q_out = np.empty_like(q)
        q_out[order] = np.minimum(q, 1.0)
        table.loc[tested, "q_value"] = q_out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False, float_format="%.4g")
    neighbours = table[table["relation"] == "neighbour"]
    print(f"wrote {len(table)} rows -> {args.output}; neighbours measured {int(neighbours['measured'].sum())} "
          f"of {len(neighbours)}; knocked down (log2FC <= -0.5, q < 0.05): "
          f"{int(((neighbours['log2fc'] <= -0.5) & (neighbours['q_value'] < 0.05)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
