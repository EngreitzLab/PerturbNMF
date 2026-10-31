"""
Core functions for U-test perturbation calibration.

These functions are extracted from
`Slurm_version/U-test_perturbation_calibration/U-test_perturbation_calibration.py`
so they can be shared between the SLURM driver and the interactive notebook.

They are fully parameterized (no dependence on a global argparse `args`); the
callers pass the run configuration explicitly.

Steps:
- `compute_real_perturbation_tests`  : U-test on real targeting guides
- `compute_fake_perturbation_tests`  : calibration null via randomly relabeled guides
- `load_real_perturbation_tests`     : reload pre-computed real results
- `load_fake_perturbation_tests`     : reload pre-computed fake results
- `plot_calibration_comparison`      : violin plot of -ln(p) real vs null
- `plot_qq_comparison`               : QQ plot real vs null (all programs pooled;
                                       per_program=True adds one panel per program)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import muon as mu

import seaborn as sns
from matplotlib import pyplot as plt

# Make the repo `src` root importable so we can reach the Stage2 metrics package
# regardless of the caller's own sys.path setup.
_SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _SRC_ROOT not in sys.path:
    sys.path.append(_SRC_ROOT)

from Stage2_Evaluation.A_Metrics.src import (
    compute_perturbation_association,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def resolve_reference_targets(patterns, available_targets):
    """Resolve control-target patterns against the dataset's actual target labels.

    Each entry in `patterns` (from `guide_annotation_key`) is matched against
    `available_targets` (the set in `mdata.uns[guide_targets_key]`) as either
    (a) an exact string, or (b) a regex substring via `re.search`. Substring
    semantics mean a plain key like ``'non-targeting'`` also captures suffixed
    variants such as ``non-targeting_1`` / ``non-targeting_2``, and ``'_WT'`` captures
    ``sampleA_WT111`` — no explicit ``.*`` wildcards needed.

    Note: matching is greedy/substring, so a key that is a substring of an unrelated
    target will also match (e.g. ``'TRIB1'`` would also match ``'TRIB10'`` if present).
    Use a more specific key or anchors (``'^non-targeting'``) to tighten it.

    Returns the sorted list of matched labels. If nothing matches, returns the
    original literal `patterns` unchanged so the caller's overlap guard still fires
    a clear error instead of masking the miss.
    """
    available = set(available_targets)
    matched = set()
    for pat in patterns:
        if pat in available:  # fast path / exact label
            matched.add(pat)
            continue
        try:
            rx = re.compile(pat)
        except re.error:  # not a valid regex and not an exact label
            continue
        matched.update(t for t in available if rx.search(t))
    return sorted(matched) if matched else list(patterns)


# ─────────────────────────────────────────────────────────────────────────────
# Compute real perturbation tests
# ─────────────────────────────────────────────────────────────────────────────
def compute_real_perturbation_tests(
    out_dir,
    run_name,
    K,
    sel_threshs,
    prog_key="cNMF",
    data_key="rna",
    categorical_key="sample",
    guide_targets_key="guide_targets",
    guide_names_key="guide_names",
    guide_annotation_path=None,
    guide_annotation_key=("non-targeting",),
    FDR_method="StoreyQ",
    skip_existing=False,
    reference_mode="complement",
    balanced=False,
    balanced_seed=0,
):
    """Compute perturbation association tests on real data.

    Resolves `reference_targets` (path-based annotation overrides key when
    provided), validates it overlaps `mdata.uns[guide_targets_key]`, then runs
    `compute_perturbation_association` per (K, sel_thresh, sample) and writes
    `{K}_perturbation_association_results_{sample}.txt` into each Evaluation
    folder.

    reference_mode ('ntc_only' or 'complement') picks the reference cells;
    each target's own cells are always removed from its reference (see
    `compute_perturbation_association`).
    balanced subsamples each target's reference to its cell count
    (balanced_seed seeds the draw).
    """

    # Load guide annotations and extract non-targeting guides or use the name of non-targeting
    # NOTE: guide_annotation_path extracts guide NAMES (e.g. "non-targeting_00014")
    # from the annotation table index, but compute_perturbation_association expects
    # target GROUP names (e.g. "non-targeting") from mdata.uns["guide_targets"].
    # Prefer using guide_annotation_key (default: ["non-targeting"]) for key-based access.
    if guide_annotation_path is not None:
        df_target = pd.read_csv(guide_annotation_path, sep="\t", index_col=0)
        df_target_non = df_target[df_target["targeting"] == False]
        reference_targets = df_target_non.index.values.tolist()
    else:
        df_target = None
        reference_targets = list(guide_annotation_key)

    test_stats_real_df = []

    # Validate once using the first K and sel_thresh
    first_k = K[0]
    first_thresh = sel_threshs[0]
    thresh_str = str(first_thresh).replace(".", "_")
    mdata_check = mu.read(
        f"{out_dir}/{run_name}/Inference/adata/cNMF_{first_k}_{thresh_str}.h5mu"
    )

    mdata_targets = set(mdata_check[prog_key].uns[guide_targets_key])

    # Key-based access: treat each guide_annotation_key entry as an exact label OR a
    # regex, expanding it to every matching group name in mdata guide_targets. This
    # handles controls stored as suffixed variants (e.g. non-targeting_1, *_WT111)
    # rather than a single "non-targeting" label. Path-based reference_targets (guide
    # names) are left untouched.
    if guide_annotation_path is None:
        reference_targets = resolve_reference_targets(
            guide_annotation_key, mdata_targets
        )

    matched_ref = mdata_targets.intersection(reference_targets)
    print(f"reference_targets overlap with mdata guide_targets: {matched_ref}")
    if len(matched_ref) == 0:
        raise ValueError(
            f"No reference_targets found in mdata guide_targets. "
            f"reference_targets contains guide names (e.g. {list(reference_targets)[:3]}), "
            f"but guide_targets contains group names (e.g. {list(mdata_targets)[:3]}). "
            f"Use guide_annotation_key instead of guide_annotation_path."
        )

    if guide_annotation_path is not None:
        n_file_guides = len(df_target)
        n_mdata_guides = len(mdata_check[prog_key].uns[guide_names_key])
        if n_file_guides != n_mdata_guides:
            raise ValueError(
                f"Guide count mismatch: annotation file has {n_file_guides} guides, "
                f"but mdata has {n_mdata_guides} guides. "
                f"Ensure the annotation file matches the mdata guide set."
            )

    del mdata_check

    for sel_thresh in sel_threshs:
        for k in K:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            output_folder = f"{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace('.', '_')}"
            os.makedirs(output_folder, exist_ok=True)

            # Load mdata
            mdata = mu.read(
                f"{out_dir}/{run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace('.', '_')}.h5mu"
            )

            # Run perturbation association for each sample
            for samp in mdata[data_key].obs[categorical_key].unique():
                out_path = (
                    f"{output_folder}/{k}_perturbation_association_results_{samp}.txt"
                )

                if skip_existing and os.path.exists(out_path):
                    print(
                        f"  Skipping K={k}, sel_thresh={sel_thresh}, samp={samp}: output exists"
                    )
                    test_stats_df = pd.read_csv(out_path, sep="\t")
                else:
                    mdata_ = mdata[mdata[data_key].obs[categorical_key] == samp]

                    test_stats_df = compute_perturbation_association(
                        mdata_,
                        prog_key=prog_key,
                        collapse_targets=True,
                        pseudobulk=False,
                        reference_targets=reference_targets,
                        reference_mode=reference_mode,
                        balanced=balanced,
                        balanced_seed=balanced_seed,
                        FDR_method=FDR_method,
                        n_jobs=-1,
                        inplace=False,
                    )

                    # Save results
                    test_stats_df.to_csv(out_path, sep="\t", index=False)

                # Add metadata
                test_stats_df["sample"] = samp
                test_stats_df["K"] = k
                test_stats_df["sel_thresh"] = sel_thresh
                test_stats_real_df.append(test_stats_df)

    # Concatenate all results
    test_stats_real_df = pd.concat(test_stats_real_df, ignore_index=True)
    test_stats_real_df["real"] = True

    return test_stats_real_df


# ─────────────────────────────────────────────────────────────────────────────
# Compute fake perturbation tests (calibration)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fake_perturbation_tests(
    out_dir,
    run_name,
    K,
    sel_threshs,
    prog_key="cNMF",
    data_key="rna",
    categorical_key="sample",
    guide_names_key="guide_names",
    guide_targets_key="guide_targets",
    guide_assignment_key="guide_assignment",
    guide_annotation_path=None,
    guide_annotation_key=("non-targeting",),
    number_run=300,
    number_guide=6,
    FDR_method="StoreyQ",
    skip_existing=False,
    reference_mode="complement",
    balanced=False,
    balanced_seed=0,
):
    """Calibrate tests using fake target guides.

    For each (K, sel_thresh) and `number_run` iterations, randomly relabels
    `number_guide` non-targeting guides as "targeting" and reruns the U-test,
    writing `{K}_fake_perturbation_association_results_{sample}.txt` per sample.

    The fake test keeps the full guide matrix: the chosen guides are relabeled
    'targeting', the other control guides 'non-targeting', real-target guides keep
    their targets, and only 'targeting' is tested (test_targets). The reference
    therefore follows the same reference_mode rule as the real test:
    'complement' (all cells without the fake guides) or 'ntc_only' (cells whose
    guides are all control guides), with the fake target's cells removed.
    """

    # read guide annotation file to find non-targeting guide names and targets
    if guide_annotation_path is not None:
        guide_target = pd.read_csv(guide_annotation_path, sep="\t")
        non_targeting_idx = guide_target.index[
            guide_target.targeting == False
        ]  # targeting is col with True / False
        guide_target = guide_target.loc[
            non_targeting_idx
        ]  # subset only non-targeting/safe-targeting guides
        guide_target["type"] = (
            "non-targeting"  # set both safe-targeting and non-targeting to be non-targeting
        )
    else:
        guide_target = None
        non_targeting_idx = None

    test_stats_fake_dfs = []

    # Extract guide info from mdata if no annotation file provided
    if guide_target is None:
        first_k = K[0]
        thresh_str = str(sel_threshs[0]).replace(".", "_")
        mdata_check = mu.read(
            f"{out_dir}/{run_name}/Inference/adata/cNMF_{first_k}_{thresh_str}.h5mu"
        )

        guide_targets_arr = mdata_check[prog_key].uns[guide_targets_key]
        guide_names_arr = mdata_check[prog_key].uns[guide_names_key]

        # Expand guide_annotation_key (exact label OR regex) against the dataset's
        # actual target labels so suffixed control variants are all captured.
        reference_targets = resolve_reference_targets(
            guide_annotation_key, set(guide_targets_arr)
        )
        non_targeting_idx = np.where(
            np.isin(guide_targets_arr, list(reference_targets))
        )[0]

        if len(non_targeting_idx) == 0:
            raise ValueError(
                f"No control guides found. guide_annotation_key={list(guide_annotation_key)} "
                f"(resolved to {list(reference_targets)}) did not match any entries in "
                f"mdata.uns['{guide_targets_key}']. "
                f"Example mdata guide_targets values: {list(set(guide_targets_arr))[:5]}"
            )

        print(f"  Non-targeting guides: {len(non_targeting_idx)}")

        # make a new annotation df when we don't have annotation file
        guide_target = pd.DataFrame(
            {
                guide_names_key: guide_names_arr[non_targeting_idx],
                "type": "non-targeting",
            }
        )
        del mdata_check

    # Validate guide count between annotation file and mdata
    if guide_annotation_path is not None:
        first_k = K[0]
        thresh_str = str(sel_threshs[0]).replace(".", "_")
        mdata_check = mu.read(
            f"{out_dir}/{run_name}/Inference/adata/cNMF_{first_k}_{thresh_str}.h5mu"
        )

        n_file_guides = len(pd.read_csv(guide_annotation_path, sep="\t"))
        n_mdata_guides = len(mdata_check[prog_key].uns[guide_names_key])

        if n_file_guides != n_mdata_guides:
            raise ValueError(
                f"Guide count mismatch: annotation file has {n_file_guides} guides, "
                f"but mdata has {n_mdata_guides} guides. "
                f"Ensure the annotation file matches the mdata guide set."
            )
        del mdata_check

    # Validate that there are enough non-targeting guides to sample number_guide without replacement.
    # np.random.choice(..., replace=False) would otherwise raise an opaque
    # "Cannot take a larger sample than population" error.
    n_non_targeting = len(guide_target)
    if number_guide >= n_non_targeting:
        raise ValueError(
            f"number_guide ({number_guide}) must be less than the number of available "
            f"non-targeting guides ({n_non_targeting}). At least one non-targeting guide must "
            f"remain as a reference after sampling. Reduce number_guide to at most {n_non_targeting - 1}."
        )

    for sel_thresh in sel_threshs:
        for k in K:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            # Load mdata
            output_folder = f"{out_dir}/{run_name}/Evaluation/{k}_{str(sel_thresh).replace('.', '_')}"
            os.makedirs(output_folder, exist_ok=True)

            mdata = mu.read(
                f"{out_dir}/{run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace('.', '_')}.h5mu"
            )

            # Skip this (K, sel_thresh) if all fake outputs already exist; load them into the accumulator.
            if skip_existing:
                samples_for_iter = list(mdata[data_key].obs[categorical_key].unique())
                expected = [
                    f"{output_folder}/{k}_fake_perturbation_association_results_{samp}.txt"
                    for samp in samples_for_iter
                ]
                if all(os.path.exists(p) for p in expected):
                    print(
                        f"  Skipping K={k}, sel_thresh={sel_thresh}: all fake outputs exist; loading from disk"
                    )
                    for samp, p in zip(samples_for_iter, expected):
                        samp_df = pd.read_csv(p, sep="\t")
                        test_stats_fake_dfs.append(samp_df)
                    continue

            # full guide labels for the fake runs: control guides -> 'non-targeting', real guides keep their target
            all_guide_names = np.asarray(mdata[prog_key].uns[guide_names_key])
            base_targets = (
                np.asarray(mdata[prog_key].uns[guide_targets_key]).astype(object).copy()
            )
            is_ntc = np.zeros(base_targets.size, dtype=bool)
            is_ntc[non_targeting_idx] = True
            clash = set(base_targets[~is_ntc]) & {"targeting", "non-targeting"}
            if clash:
                raise ValueError(
                    f"Real targets named {sorted(clash)} clash with the fake-test labels "
                    f"'targeting' / 'non-targeting'; add them to guide_annotation_key."
                )
            base_targets[is_ntc] = "non-targeting"

            # compute_perturbation_association only reads mdata[prog_key], so keep just that modality
            # and the per-cell condition labels, then drop the full mdata (incl. the large rna matrix)
            prog_mdata = mu.MuData({prog_key: mdata[prog_key].copy()})
            cell_conditions = (
                mdata[data_key]
                .obs[categorical_key]
                .loc[prog_mdata[prog_key].obs_names]
                .values
            )
            conditions_k = mdata[data_key].obs[categorical_key].unique()
            del mdata

            test_stats_fake_dfs_temp = []
            for i in range(number_run):
                print(f"  Running iteration {i + 1}/{number_run}")

                # Randomly pick number_guide non-targeting guides to act as the fake "targeting" target
                guide_target_ = guide_target.copy()
                selected_guides = np.random.choice(
                    guide_target_[guide_names_key], number_guide, replace=False
                )

                # Filter to cells with non-targeting guides that exist in both datasets
                valid_guide_mask = np.isin(
                    all_guide_names, guide_target_[guide_names_key].values
                )
                valid_indices = np.where(valid_guide_mask)[0]

                print(
                    f"  Found {len(valid_indices)} valid non-targeting guides out of "
                    f"{len(all_guide_names)} total"
                )

                if len(valid_indices) == 0:
                    raise ValueError("No valid guides found")

                # keep every guide column; only the chosen control guides become the fake 'targeting' target
                fake_targets = base_targets.copy()
                fake_targets[is_ntc & np.isin(all_guide_names, selected_guides)] = (
                    "targeting"
                )
                # overwritten every iteration; compute_perturbation_association(inplace=False) never mutates it
                prog_mdata[prog_key].uns[guide_targets_key] = fake_targets

                # Run perturbation association for each sample
                for samp in conditions_k:
                    mdata_samp = prog_mdata[cell_conditions == samp]

                    test_stats_df = compute_perturbation_association(
                        mdata_samp,
                        prog_key=prog_key,
                        collapse_targets=True,
                        pseudobulk=False,
                        reference_targets=["non-targeting"],
                        reference_mode=reference_mode,
                        balanced=balanced,
                        balanced_seed=balanced_seed,
                        test_targets=["targeting"],
                        FDR_method=FDR_method,
                        n_jobs=-1,
                        inplace=False,
                    )

                    test_stats_df[categorical_key] = samp
                    test_stats_df["K"] = k
                    test_stats_df["run"] = i
                    test_stats_df["sel_thresh"] = sel_thresh
                    test_stats_fake_dfs.append(test_stats_df)  # combine all
                    test_stats_fake_dfs_temp.append(
                        test_stats_df
                    )  # combine for each k and sel_thresh

            # Save results per condition
            test_stats_fake_dfs_temp = pd.concat(
                test_stats_fake_dfs_temp, ignore_index=True
            )
            for samp, samp_df in test_stats_fake_dfs_temp.groupby(categorical_key):
                samp_df.to_csv(
                    f"{output_folder}/{k}_fake_perturbation_association_results_{samp}.txt",
                    sep="\t",
                    index=False,
                )

    # Concatenate all results
    test_stats_fake_dfs = pd.concat(test_stats_fake_dfs, ignore_index=True)

    test_stats_fake_dfs["real"] = False

    return test_stats_fake_dfs


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed results
# ─────────────────────────────────────────────────────────────────────────────
def load_real_perturbation_tests(out_dir, run_name, K, sel_threshs):
    """Load pre-computed real perturbation test results (auto-discovers samples)."""

    # Discover sample names from the first K directory
    first_k = K[0]
    thresh_str = str(sel_threshs[0]).replace(".", "_")
    first_dir = f"{out_dir}/{run_name}/Evaluation/{first_k}_{thresh_str}"
    sample_files = [
        f
        for f in os.listdir(first_dir)
        if f.startswith(f"{first_k}_perturbation_association_results_")
        and f.endswith(".txt")
    ]
    samples = [
        f.replace(f"{first_k}_perturbation_association_results_", "").replace(
            ".txt", ""
        )
        for f in sample_files
    ]
    print(f"Discovered samples: {samples}")

    test_stats_real_df = []

    for sel_thresh in sel_threshs:
        for k in K:
            for samp in samples:
                thresh_str = str(sel_thresh).replace(".", "_")
                test_stats_df_ = pd.read_csv(
                    f"{out_dir}/{run_name}/Evaluation/{k}_{thresh_str}/{k}_perturbation_association_results_{samp}.txt",
                    sep="\t",
                )
                test_stats_df_["sample"] = samp
                test_stats_df_["K"] = k
                test_stats_df_["sel_thresh"] = sel_thresh
                test_stats_real_df.append(test_stats_df_)

    test_stats_real_df = pd.concat(test_stats_real_df, ignore_index=True)
    test_stats_real_df["real"] = True

    return test_stats_real_df


def load_fake_perturbation_tests(out_dir, run_name, K, sel_threshs):
    """Load pre-computed fake perturbation test results (per condition)."""

    # Discover sample names from the first K directory
    first_k = K[0]
    thresh_str = str(sel_threshs[0]).replace(".", "_")
    first_dir = f"{out_dir}/{run_name}/Evaluation/{first_k}_{thresh_str}"
    sample_files = [
        f
        for f in os.listdir(first_dir)
        if f.startswith(f"{first_k}_fake_perturbation_association_results_")
        and f.endswith(".txt")
    ]
    samples = [
        f.replace(f"{first_k}_fake_perturbation_association_results_", "").replace(
            ".txt", ""
        )
        for f in sample_files
    ]
    print(f"Discovered samples for fake tests: {samples}")

    test_stats_fake_df = []

    for sel_thresh in sel_threshs:
        for k in K:
            for samp in samples:
                thresh_str = str(sel_thresh).replace(".", "_")
                test_stats_df_ = pd.read_csv(
                    f"{out_dir}/{run_name}/Evaluation/{k}_{thresh_str}/{k}_fake_perturbation_association_results_{samp}.txt",
                    sep="\t",
                )
                test_stats_fake_df.append(test_stats_df_)

    test_stats_fake_df = pd.concat(test_stats_fake_df, ignore_index=True)
    test_stats_fake_df["real"] = False

    return test_stats_fake_df


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_calibration_comparison(test_stats_dfs, out_dir, run_name, K, sel_threshs):
    """Save a violin plot comparing real vs fake perturbation tests for each (K, sel_thresh) into its own folder."""

    for sel_thresh in sel_threshs:
        for k in K:
            thresh_str = str(sel_thresh).replace(".", "_")
            output_folder = f"{out_dir}/{run_name}/Evaluation/{k}_{thresh_str}"
            os.makedirs(output_folder, exist_ok=True)

            test_stats_k = test_stats_dfs[
                (test_stats_dfs.K == k) & (test_stats_dfs.sel_thresh == sel_thresh)
            ].copy()

            if len(test_stats_k) == 0:
                print(f"  Skipping K={k}, sel_thresh={sel_thresh}: no data")
                continue

            test_stats_k["neg_log_pval"] = test_stats_k["pval"].apply(
                lambda x: -np.log(x)
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            sns.violinplot(
                x="sample", y="neg_log_pval", hue="real", data=test_stats_k, ax=ax
            )

            ax.set_title(f"K={k}, sel_thresh={sel_thresh}")
            ax.set_xlabel("Sample ID")
            ax.set_ylabel("-ln(p-value)")
            ax.set_ylim(0, 50)
            ax.axhline(8, color="grey", linestyle="dashed")

            plt.tight_layout()
            plt.savefig(
                f"{output_folder}/U_test_perturbation_association_calibration.png",
                dpi=100,
            )
            plt.close(fig)


def _qq_points(ax, pvals, label, color):
    """Scatter -log10 sorted p against -log10 Uniform quantiles (i - 0.5) / n."""
    p = np.sort(np.asarray(pvals, dtype=np.float64))
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return
    expected = (np.arange(1, p.size + 1) - 0.5) / p.size
    ax.scatter(
        -np.log10(expected),
        -np.log10(p),
        s=4,
        color=color,
        label=label,
        rasterized=True,
        lw=0,
    )


def _qq_axes(ax, n_max):
    """y = x line; x spans the expected range only, so a few huge real -log10 p do not squash the null."""
    x_max = 1.05 * -np.log10(0.5 / max(n_max, 1))
    ax.plot([0, x_max], [0, x_max], color="k", lw=0.8, ls="--")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(ax.get_ylim()[1], x_max))


def _null_frac(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=np.float64)
    return float(np.mean(pvals < alpha)) if pvals.size else float("nan")


def plot_qq_comparison(
    test_stats_dfs, out_dir, run_name, K, sel_threshs, per_program=False, n_cols=6
):
    """Save QQ plots comparing real and null U-test p-values for each (K, sel_thresh) into its own folder.

    Real tests are red, null (fake targeting) tests are grey, against Uniform(0, 1);
    the title gives the null's fraction of p < 0.05 (expected 0.05 when calibrated).

    Writes, in {out_dir}/{run_name}/Evaluation/{K}_{thresh}/:
    - U_test_perturbation_association_qqplot.png: all programs pooled.
    - U_test_perturbation_association_qqplot_per_program.png (per_program=True):
      one panel per program (``program_name``), n_cols panels per row.
    """

    for sel_thresh in sel_threshs:
        for k in K:
            thresh_str = str(sel_thresh).replace(".", "_")
            output_folder = f"{out_dir}/{run_name}/Evaluation/{k}_{thresh_str}"
            os.makedirs(output_folder, exist_ok=True)

            real_df = test_stats_dfs.loc[
                (test_stats_dfs.K == k)
                & (test_stats_dfs.sel_thresh == sel_thresh)
                & (test_stats_dfs.real == True)
            ]

            null_df = test_stats_dfs.loc[
                (test_stats_dfs.K == k)
                & (test_stats_dfs.sel_thresh == sel_thresh)
                & (test_stats_dfs.real == False)
                & (test_stats_dfs.target_name == "targeting")
            ]

            if len(real_df) == 0 and len(null_df) == 0:
                print(f"  Skipping K={k}, sel_thresh={sel_thresh}: no data")
                continue

            # pooled over programs
            fig, ax = plt.subplots(figsize=(5, 5))
            _qq_points(ax, real_df["pval"], f"Real ({len(real_df):,} tests)", "tab:red")
            _qq_points(ax, null_df["pval"], f"Null ({len(null_df):,} tests)", "0.4")
            _qq_axes(ax, max(len(real_df), len(null_df)))
            ax.set_xlabel("expected -log10 p (Uniform)")
            ax.set_ylabel("observed -log10 p")
            ax.set_title(
                f"U-test QQ: K={k}, sel_thresh={sel_thresh}\n"
                f"null frac p<0.05 = {_null_frac(null_df['pval']):.3f}",
                fontsize=9,
            )
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            fig.tight_layout()
            fig.savefig(
                f"{output_folder}/U_test_perturbation_association_qqplot.png", dpi=150
            )
            plt.close(fig)

            if not per_program:
                continue

            # one panel per program
            programs = list(
                pd.unique(pd.concat([real_df["program_name"], null_df["program_name"]]))
            )
            try:  # natural order (e.g. 1, 2, ..., 10) when program names are numeric
                programs = sorted(programs, key=lambda x: float(x))
            except (TypeError, ValueError):
                programs = sorted(programs, key=str)
            real_by = dict(tuple(real_df.groupby("program_name")["pval"]))
            null_by = dict(tuple(null_df.groupby("program_name")["pval"]))
            n_max = max(
                [len(v) for v in list(real_by.values()) + list(null_by.values())] or [1]
            )

            n_rows = int(np.ceil(len(programs) / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), squeeze=False
            )
            for i, ax in enumerate(axes.flat):
                if i >= len(programs):
                    ax.axis("off")
                    continue
                prog = programs[i]
                p_null = null_by.get(prog, pd.Series(dtype=float))
                _qq_points(
                    ax, real_by.get(prog, pd.Series(dtype=float)), "Real", "tab:red"
                )
                _qq_points(ax, p_null, "Null", "0.4")
                _qq_axes(ax, n_max)
                ax.set_title(
                    f"{prog}  null p<0.05: {_null_frac(p_null):.3f}", fontsize=7
                )
                ax.tick_params(labelsize=6)
            fig.suptitle(
                f"U-test QQ per program: K={k}, sel_thresh={sel_thresh}  "
                f"(red = real, grey = null; x = expected, y = observed -log10 p)",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            fig.savefig(
                f"{output_folder}/U_test_perturbation_association_qqplot_per_program.png",
                dpi=150,
            )
            plt.close(fig)
