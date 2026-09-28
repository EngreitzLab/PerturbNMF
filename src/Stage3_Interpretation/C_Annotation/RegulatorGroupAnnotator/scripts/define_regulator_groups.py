"""Group perturbed genes whose knockdowns move the gene programs the same way.

The obvious rule, one global cut on the correlation of effect profiles (GeneProgramExplorer:
1 - r average linkage at 0.5), favours strong regulators. Noise attenuates correlation, so two
subunits of a complex with weak effects correlate at r = 0.3 while two strong ones reach 0.8,
though both pairs are equally related. This script replaces the single cut with four steps:

  1. noise-corrected similarity   r* = r / sqrt(rel_i * rel_j), rel from
                                  build_regulator_effect_matrix.py (share of each profile's
                                  variance that is signal). Regulators below --reliability-floor
                                  are reported but not grouped.
  2. pair significance            each pair's raw r against a program-permutation null (the
                                  partner's profile with program labels shuffled, the same
                                  shuffle for every condition), per regulator, BH across pairs.
                                  Only positive, significant pairs can link regulators — a
                                  significance gate, not a magnitude cut.
  3. local consistency            shared-nearest-neighbour (SNN) similarity: each regulator's
                                  k nearest significant partners by r*; two regulators are similar
                                  when their neighbour sets overlap (Jaccard). Rank-based, so a
                                  weak regulator is judged against its own neighbours, not
                                  against the strongest pairs in the screen. Average linkage on
                                  1 - SNN, cut at --snn-cut.
  4. consensus                    bootstrap over programs (all conditions of a program together),
                                  re-running steps 1 and 3 each time, gives each pair's
                                  co-assignment frequency. The FINAL groups are average-linkage
                                  clusters of 1 - co-assignment, cut at --consensus-cut (default
                                  0.7: members co-assigned in >= 30% of bootstraps on average).
                                  A single clustering of the full data puts a gene that bridges two
                                  modules into whichever it happens to meet first (e.g. a gene
                                  co-assigned with module A in 41% of bootstraps and module B in
                                  26%). A member's
                                  stability is its mean co-assignment with the rest of its group;
                                  members at or above --min-member-stability are core, the rest
                                  peripheral. A group is kept when it has >= --min-group-size core
                                  members.

Recruitment (--recruit-correlated; build the effect matrix with --min-significant-features 0).
A regulator with no significant effect on any single program can still move many programs a
little, the same way as a module does. Such a regulator is recruited when it has significant
(step 2) correlation edges to >= --recruit-min-partners regulators that do have significant
effects; it is then grouped like any other and marked `recruited` downstream.
EXPERIMENTAL — not trustworthy until the pair null models correlated sampling noise: on one
test screen the program-permutation null called a sizeable share of pairs significant and
recruited nearly every target (sampling noise moves e.g. the cell-cycle programs together, which
shuffling programs destroys). Needs an NTC fake-perturbation null.

Curated complexes (CORUM / ComplexPortal / SIGNOR via OmniPath, annotator_core/complexes.py)
are used twice, and both uses are reported, not hidden:
  calibration (--calibrate)  recovery of co-complex regulator pairs over a grid of k x snn-cut,
                             split by the pair's effect-strength tier, next to the global-cut
                             baseline — written to grouping_calibration.tsv. Choosing the
                             parameters from it means complex enrichment of the final groups is
                             partly circular; the parameters used are recorded in the output.
  rescue                     a regulator in the same complex as >= 2 core members of a group, not
                             in any group (including regulators below the reliability floor),
                             whose profile correlates positively with the group centroid beyond
                             the permutation null (BH across the group's candidates, q <
                             --rescue-fdr), is added with role "rescued". The complex is the
                             prior, the correlation test the evidence; how its r compares with
                             the core members' is recorded. Rescued members are labelled as such
                             everywhere downstream.

Outputs (in --output-dir): regulator_groups.json, group_membership.tsv and, with --calibrate,
grouping_calibration.tsv.

Usage:
    python define_regulator_groups.py --matrix-dir regulator_groups \
        --complexes resources/omnipath_complexes.tsv [--calibrate] --output-dir regulator_groups
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from complexes import load_curated_complexes  # noqa: E402

CALIBRATION_NEIGHBORS = (5, 10, 15)
CALIBRATION_SNN_CUTS = (0.6, 0.7, 0.8, 0.9)
BASELINE_DISTANCE_CUTS = (0.3, 0.4, 0.5, 0.6, 0.7)
CALIBRATION_CONSENSUS_CUTS = (0.6, 0.7, 0.8)
CALIBRATION_BOOTSTRAPS = 50
CALIBRATION_MAX_COMPLEX_SIZE = 40  # bulk "complexes" of 100+ proteins say little about a pair


# ---- similarity ---------------------------------------------------------------------------
def row_standardize(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    return centred / np.where(norm == 0, 1.0, norm)


def correlation(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    a = row_standardize(x)
    b = a if y is None else row_standardize(y)
    return np.clip(a @ b.T, -1.0, 1.0)


def corrected_similarity(r: np.ndarray, reliability: np.ndarray, floor: float) -> np.ndarray:
    rel = np.maximum(reliability, floor)
    return np.clip(r / np.sqrt(np.outer(rel, rel)), -1.0, 1.0)


def program_blocks(features: list[str]) -> list[list[int]]:
    """Column indices per program ("P<program>|<condition>"), conditions in column order."""
    blocks: dict[str, list[int]] = {}
    for i, name in enumerate(features):
        blocks.setdefault(name.split("|")[0], []).append(i)
    return list(blocks.values())


def permute_programs(blocks: list[list[int]], rng: np.random.Generator) -> np.ndarray:
    """Column order with whole programs shuffled; every condition of a program moves together."""
    order = rng.permutation(len(blocks))
    columns = np.empty(sum(len(b) for b in blocks), dtype=int)
    for target, source in zip(blocks, order):
        columns[target] = blocks[source]
    return columns


def permutation_null(x: np.ndarray, blocks: list[list[int]], n_permutations: int, seed: int) -> np.ndarray:
    """Per regulator, sorted correlations with every other regulator's program-shuffled profile."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    null = np.empty((n, (n - 1) * n_permutations), dtype=np.float32)
    off_diagonal = ~np.eye(n, dtype=bool)
    for b in range(n_permutations):
        shuffled = x[:, permute_programs(blocks, rng)]
        values = correlation(x, shuffled)[off_diagonal].reshape(n, n - 1)
        null[:, b * (n - 1):(b + 1) * (n - 1)] = values
    null.sort(axis=1)
    return null


def upper_tail_p(null_row: np.ndarray, values: np.ndarray) -> np.ndarray:
    exceed = null_row.size - np.searchsorted(null_row, values, side="left")
    return (1.0 + exceed) / (1.0 + null_row.size)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def significant_edges(r: np.ndarray, null: np.ndarray, fdr: float) -> np.ndarray:
    n = r.shape[0]
    p = np.ones((n, n))
    for i in range(n):
        p[i] = upper_tail_p(null[i], r[i])
    p = np.maximum(p, p.T)  # significant against BOTH partners' nulls
    iu = np.triu_indices(n, 1)
    q = np.ones((n, n))
    q[iu] = benjamini_hochberg(p[iu])
    q = np.minimum(q, q.T)
    edges = (q < fdr) & (r > 0)
    np.fill_diagonal(edges, False)
    return edges


# ---- SNN clustering -----------------------------------------------------------------------
def snn_similarity(similarity: np.ndarray, edges: np.ndarray, k: int) -> np.ndarray:
    n = similarity.shape[0]
    neighbours = []
    for i in range(n):
        partners = np.flatnonzero(edges[i])
        top = partners[np.argsort(-similarity[i, partners])][:k]
        neighbours.append(set(top.tolist()) | {i})
    snn = np.eye(n)
    for i, j in zip(*np.nonzero(np.triu(edges, 1))):
        shared = len(neighbours[i] & neighbours[j])
        snn[i, j] = snn[j, i] = shared / len(neighbours[i] | neighbours[j])
    return snn


def split_oversize_block(idx: list[int], distance: np.ndarray, max_size: int) -> list[list[int]]:
    """Bisect a block at the root of its own average-linkage tree until it fits (ported from
    GeneProgramExplorer clustering.split_oversize_block); an inseparable block is kept whole."""
    if len(idx) <= max_size:
        return [idx]
    block = distance[np.ix_(idx, idx)]
    root = to_tree(linkage(squareform(block, checks=False), method="average"))
    if root.dist <= max(root.left.dist, root.right.dist):
        return [idx]
    left = [idx[i] for i in root.left.pre_order()]
    right = [idx[i] for i in root.right.pre_order()]
    return split_oversize_block(left, distance, max_size) + split_oversize_block(right, distance, max_size)


def cluster(distance: np.ndarray, cut: float, max_size: int) -> np.ndarray:
    """Cluster label per row (-1 = singleton) from average linkage on a distance matrix."""
    n = distance.shape[0]
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"), t=cut, criterion="distance")
    out = np.full(n, -1)
    next_label = 0
    for label in np.unique(labels):
        for piece in split_oversize_block(np.flatnonzero(labels == label).tolist(), distance, max_size):
            if len(piece) >= 2:
                out[piece] = next_label
                next_label += 1
    return out


def snn_clusters(x, reliability, edges, k, cut, floor, max_size) -> np.ndarray:
    similarity = corrected_similarity(correlation(x), reliability, floor)
    return cluster(1.0 - snn_similarity(similarity, edges, k), cut, max_size)


def consensus_clusters(consensus: np.ndarray, cut: float, max_size: int) -> np.ndarray:
    return cluster(1.0 - consensus, cut, max_size)


def co_assignment(labels: np.ndarray) -> np.ndarray:
    return (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)


def bootstrap_consensus(x, reliability_of, edges, blocks, k, cut, floor, max_size, n_boot, seed) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    together = np.zeros((x.shape[0], x.shape[0]))
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), len(blocks))
        columns = np.concatenate([blocks[c] for c in chosen])
        sample = x[:, columns]
        together += co_assignment(snn_clusters(sample, reliability_of(sample), edges, k, cut, floor, max_size))
    return together / n_boot


# ---- complexes ----------------------------------------------------------------------------
def co_complex_pairs(genes: list[str], complexes) -> set:
    present = set(genes)
    pairs = set()
    for entry in complexes:
        if len(entry.members) > CALIBRATION_MAX_COMPLEX_SIZE:
            continue
        members = sorted(present & set(entry.members))
        pairs.update(itertools.combinations(members, 2))
    return pairs


def recovery(labels: np.ndarray, genes: list[str], pairs: set, tier: dict, in_any_complex: set) -> dict:
    index = {g: i for i, g in enumerate(genes)}
    grouped_pairs = {tuple(sorted((genes[i], genes[j]))) for i, j in zip(*np.nonzero(np.triu(co_assignment(labels), 1)))}
    rank = {"weak": 0, "medium": 1, "strong": 2}
    row = {"n_groups": int(labels.max() + 1), "n_grouped": int((labels >= 0).sum())}
    for name in ("all", "weak", "medium", "strong"):
        subset = [p for p in pairs if name == "all" or min(tier[p[0]], tier[p[1]], key=rank.get) == name]
        hit = sum(labels[index[a]] >= 0 and labels[index[a]] == labels[index[b]] for a, b in subset)
        row[f"recall_{name}"] = round(hit / len(subset), 3) if subset else None
        row[f"pairs_{name}"] = len(subset)
    annotated = [p for p in grouped_pairs if p[0] in in_any_complex and p[1] in in_any_complex]
    row["precision"] = round(sum(p in pairs for p in annotated) / len(annotated), 3) if annotated else None
    return row


def calibrate(x, genes, reliability, reliability_of, edges, blocks, floor, max_size, complexes, tier) -> pd.DataFrame:
    pairs = co_complex_pairs(genes, complexes)
    in_any = {g for p in pairs for g in p}
    rows = []
    for k in CALIBRATION_NEIGHBORS:
        for cut in CALIBRATION_SNN_CUTS:
            labels = snn_clusters(x, reliability, edges, k, cut, floor, max_size)
            rows.append({"method": "snn", "neighbors": k, "cut": cut, **recovery(labels, genes, pairs, tier, in_any)})
            consensus = bootstrap_consensus(x, reliability_of, edges, blocks, k, cut, floor, max_size,
                                            CALIBRATION_BOOTSTRAPS, 0)
            for consensus_cut in CALIBRATION_CONSENSUS_CUTS:
                labels = consensus_clusters(consensus, consensus_cut, max_size)
                rows.append({"method": f"snn_consensus_{consensus_cut}", "neighbors": k, "cut": cut,
                             **recovery(labels, genes, pairs, tier, in_any)})
    r = correlation(x)
    for cut in BASELINE_DISTANCE_CUTS:
        # GeneProgramExplorer's rule: 1 - r (positive r only), average linkage, one global cut.
        labels = cluster(1.0 - r, cut, max_size)
        rows.append({"method": "global_1-r", "neighbors": None, "cut": cut, **recovery(labels, genes, pairs, tier, in_any)})
    return pd.DataFrame(rows)


def rescue_candidates(group, labels, genes, x, null, reliability, complexes, alpha) -> list[dict]:
    """Complex partners of >= 2 members that no group claimed, tested against the group centroid.

    `genes`, `x`, `null`, `labels` cover EVERY regulator in the effect matrix, including those
    below the reliability floor: a subunit that just missed the bar is the case rescue is for."""
    members = [genes.index(g) for g in group["core"]]
    centroid = row_standardize(x[members]).mean(axis=0, keepdims=True)
    member_r = []
    for m in members:  # leave-one-out, so a member is not compared with itself
        others = [o for o in members if o != m]
        loo = row_standardize(x[others]).mean(axis=0, keepdims=True)
        member_r.append(float(correlation(x[[m]], loo)[0, 0]))
    member_range = [round(min(member_r), 3), round(max(member_r), 3)]
    tests = []
    for entry in complexes:
        inside = sorted(set(entry.members) & set(group["core"]))
        if len(inside) < 2:
            continue
        for gene in entry.members:
            if gene not in genes or gene in group["member_genes"]:
                continue
            i = genes.index(gene)
            r = float(correlation(x[[i]], centroid)[0, 0])
            tests.append({"gene": gene, "complex": entry.name, "complex_id": entry.complex_id,
                          "complex_members_in_group": inside, "r_to_centroid": round(r, 3),
                          "p": float(upper_tail_p(null[i], np.array([r]))[0]),
                          "assigned_elsewhere": bool(labels[i] >= 0), "i": i})
    if not tests:
        return []
    best = {}
    for t in sorted(tests, key=lambda t: t["p"]):  # one test per gene: its strongest complex link
        best.setdefault(t["gene"], t)
    tests = list(best.values())
    q = benjamini_hochberg(np.array([t["p"] for t in tests]))
    rescued = []
    for t, q_value in zip(tests, q):
        t["q"] = round(float(q_value), 4)
        t["core_member_r_range"] = member_range  # leave-one-out r of core members to the centroid
        t["accepted"] = bool(q_value < alpha and t["r_to_centroid"] > 0 and not t["assigned_elsewhere"])
        t["reliability"] = round(float(reliability[t.pop("i")]), 3)
        rescued.append(t)
    return rescued


# ---- main ---------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix-dir", required=True, type=Path, help="build_regulator_effect_matrix.py output")
    parser.add_argument("--complexes", type=Path, help="OmniPath complexes TSV (downloaded if missing); enables rescue and --calibrate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--neighbors", type=int, default=5, help="k for the shared-nearest-neighbour graph")
    parser.add_argument("--snn-cut", type=float, default=0.7, help="average-linkage cut on 1 - SNN similarity")
    parser.add_argument("--edge-fdr", type=float, default=0.05, help="BH FDR for a pair to count as similar")
    parser.add_argument("--reliability-floor", type=float, default=0.2, help="regulators below this are not grouped")
    parser.add_argument("--permutations", type=int, default=20, help="program permutations for the pair null")
    parser.add_argument("--bootstraps", type=int, default=100, help="program bootstraps for stability")
    parser.add_argument("--consensus-cut", type=float, default=0.7,
                        help="average-linkage cut on 1 - bootstrap co-assignment (the final groups)")
    parser.add_argument("--recruit-correlated", action="store_true",
                        help="let regulators with no significant program effect join through correlation edges")
    parser.add_argument("--recruit-min-partners", type=int, default=2,
                        help="significant edges to significant regulators a recruit needs")
    parser.add_argument("--min-group-size", type=int, default=2, help="core members a group needs")
    parser.add_argument("--max-group-size", type=int, default=20)
    parser.add_argument("--min-member-stability", type=float, default=0.3,
                        help="bootstrap co-assignment a member needs to count as core")
    parser.add_argument("--rescue-fdr", type=float, default=0.05)
    parser.add_argument("--calibrate", action="store_true", help="write the complex-recovery grid (needs --complexes)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    effects = pd.read_csv(args.matrix_dir / "effect_matrix.tsv", sep="\t", index_col=0)
    summary = pd.read_csv(args.matrix_dir / "regulator_summary.tsv", sep="\t", index_col=0).loc[effects.index]
    reliable = summary["reliability"] >= args.reliability_floor
    low_reliability = summary.index[~reliable].tolist()
    features = effects.columns.tolist()
    blocks = program_blocks(features)
    all_genes = summary.index.tolist()
    x_all = effects.to_numpy()
    null_all = permutation_null(x_all, blocks, args.permutations, args.seed)
    anchored = (summary["n_significant"] >= 1).to_numpy()
    candidates = np.flatnonzero(reliable.to_numpy() & (anchored | args.recruit_correlated))
    recruited = []
    if args.recruit_correlated:
        candidate_edges = significant_edges(correlation(x_all[candidates]), null_all[candidates], args.edge_fdr)
        partners = candidate_edges[:, anchored[candidates]].sum(axis=1)
        keep = anchored[candidates] | (partners >= args.recruit_min_partners)
        recruited = [all_genes[i] for i in candidates[keep & ~anchored[candidates]]]
        candidates = candidates[keep]
    eligible = candidates
    genes = [all_genes[i] for i in eligible]
    x = x_all[eligible]
    standard_error = summary.loc[genes, "standard_error"].to_numpy()
    reliability = summary.loc[genes, "reliability"].to_numpy()
    tier = summary.loc[genes, "strength_tier"].astype(str).to_dict()

    def reliability_of(sample: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - standard_error ** 2 / np.maximum(sample.var(axis=1), 1e-12), 0.0, 1.0)

    r = correlation(x)
    edges = significant_edges(r, null_all[eligible], args.edge_fdr)
    similarity = corrected_similarity(r, reliability, args.reliability_floor)
    consensus = bootstrap_consensus(x, reliability_of, edges, blocks, args.neighbors, args.snn_cut,
                                    args.reliability_floor, args.max_group_size, args.bootstraps, args.seed)
    labels = consensus_clusters(consensus, args.consensus_cut, args.max_group_size)
    complexes = load_curated_complexes(args.complexes) if args.complexes else []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        if not complexes:
            raise SystemExit("--calibrate needs --complexes")
        table = calibrate(x, genes, reliability, reliability_of, edges, blocks, args.reliability_floor,
                          args.max_group_size, complexes, tier)
        table.to_csv(args.output_dir / "grouping_calibration.tsv", sep="\t", index=False)
        print(table.to_string(index=False))

    groups = []
    for label in range(labels.max() + 1):
        idx = np.flatnonzero(labels == label)
        block = consensus[np.ix_(idx, idx)]
        member_stability = (block.sum(axis=1) - 1.0) / (len(idx) - 1)
        core = member_stability >= args.min_member_stability
        if core.sum() < args.min_group_size:
            continue
        core_block = block[np.ix_(core, core)]
        stability = float(core_block[~np.eye(int(core.sum()), dtype=bool)].mean())  # among core members
        sub = similarity[np.ix_(idx, idx)]
        groups.append({
            "idx": idx.tolist(), "stability": round(stability, 3),
            "mean_corrected_r": round(float(sub[~np.eye(len(idx), dtype=bool)].mean()), 3),
            "mean_raw_r": round(float(r[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)].mean()), 3),
            "member_stability": member_stability.round(3).tolist(),
        })
    groups.sort(key=lambda g: (-len(g["idx"]), -g["stability"]))
    labels_all = np.full(len(all_genes), -1)  # final group per regulator in the full matrix
    for group_id, group in enumerate(groups):
        labels_all[eligible[group["idx"]]] = group_id

    out_groups = []
    for group_id, group in enumerate(groups):
        idx = group["idx"]
        centroid = row_standardize(x[idx]).mean(axis=0, keepdims=True)
        members = []
        for i, stable in zip(idx, group["member_stability"]):
            members.append({
                "gene": genes[i], "role": "core" if stable >= args.min_member_stability else "peripheral",
                "stability": stable, "r_to_centroid": round(float(correlation(x[[i]], centroid)[0, 0]), 3),
                "reliability": round(float(reliability[i]), 3), "strength_tier": tier[genes[i]],
                "recruited": genes[i] in recruited,
                "n_significant": int(summary.loc[genes[i], "n_significant"]),
                "signal_rms": round(float(summary.loc[genes[i], "signal_rms"]), 3),
            })
        entry = {
            "group_id": group_id, "size": len(members), "stability": group["stability"],
            "mean_corrected_r": group["mean_corrected_r"], "mean_raw_r": group["mean_raw_r"],
            "strength_tiers": {t: sum(m["strength_tier"] == t for m in members) for t in ("weak", "medium", "strong")},
            "members": members,
        }
        entry["core"] = [m["gene"] for m in members if m["role"] == "core"]
        entry["member_genes"] = [m["gene"] for m in members]
        tests = rescue_candidates(entry, labels_all, all_genes, x_all, null_all,
                                  summary["reliability"].to_numpy(), complexes, args.rescue_fdr) if complexes else []
        entry["rescue_tests"] = tests
        del entry["member_genes"]
        out_groups.append(entry)

    # A gene can pass rescue for two groups (TRRAP is in SAGA and TFIID); it joins only the one
    # with the strongest evidence (lowest q, then highest r) and the other test records why not.
    accepted = [(t["q"], -t["r_to_centroid"], g["group_id"], t) for g in out_groups for t in g["rescue_tests"] if t["accepted"]]
    placed = {}
    for _, _, group_id, t in sorted(accepted, key=lambda a: a[:3]):
        if t["gene"] in placed:
            t["accepted"] = False
            t["rescued_into_group"] = placed[t["gene"]]
            continue
        placed[t["gene"]] = group_id
        out_groups[group_id]["members"].append({
            "gene": t["gene"], "role": "rescued", "stability": None,
            "r_to_centroid": t["r_to_centroid"], "reliability": t["reliability"],
            "strength_tier": str(summary.loc[t["gene"], "strength_tier"]),
            "rescued_via": t["complex"],
            "n_significant": int(summary.loc[t["gene"], "n_significant"]),
            "signal_rms": round(float(summary.loc[t["gene"], "signal_rms"]), 3)})
    for entry in out_groups:
        entry["size"] = len(entry["members"])

    grouped = {m["gene"] for g in out_groups for m in g["members"]}
    payload = {
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_regulators": len(summary), "n_eligible": len(genes), "n_edges": int(np.triu(edges, 1).sum()),
        "n_recruited": len(recruited), "recruited": recruited,
        "calibration_used_for_parameters": bool(args.calibrate),
        "groups": out_groups,
        "ungrouped": sorted(set(genes) - grouped),
        "low_reliability": sorted(set(low_reliability) - grouped),
        "features": features,
    }
    (args.output_dir / "regulator_groups.json").write_text(json.dumps(payload, indent=1))
    rows = [{"group_id": g["group_id"], **{k: m[k] for k in ("gene", "role", "stability", "r_to_centroid", "reliability", "strength_tier")}}
            for g in out_groups for m in g["members"]]
    pd.DataFrame(rows).to_csv(args.output_dir / "group_membership.tsv", sep="\t", index=False)
    n_rescued = sum(m["role"] == "rescued" for g in out_groups for m in g["members"])
    print(f"{len(out_groups)} groups covering {len(grouped)} of {len(genes)} eligible regulators "
          f"({n_rescued} rescued via complexes); {len(low_reliability)} below reliability "
          f"{args.reliability_floor}; {payload['n_edges']} significant pairs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
