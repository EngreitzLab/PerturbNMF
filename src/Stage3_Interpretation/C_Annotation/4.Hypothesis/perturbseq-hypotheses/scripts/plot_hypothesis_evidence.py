#!/usr/bin/env python3
"""Per-hypothesis evidence plots — reference implementation for the Analyst.

Copy into the analysis `src/`, adapt the loaders to this run's layout, commit the adapted copy.
Every plot function takes tidy pandas tables, so only the loaders change between runs:

    effects      target, program, log2fc, qvalue          (one row per target x program)
    loadings     program, gene, loading                    (cNMF spectra, long format)
    correlation  square DataFrame, index = columns = target (effect-vector correlation)
    guides       guide, target, program, log2fc            (per-guide effects, if the run has them)

Each function writes <out>.pdf, <out>.png and <out>.csv (the plotted values) and returns the
DataFrame it plotted. Masked / non-significant cells are hatched grey, never a palette colour.

    python plot_hypothesis_evidence.py --self-test OUT_DIR    # synthetic data, all four plots
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_house_style_rcparams() -> dict | None:
    """Load the lab house-style rcParams from EngreitzLabAgents/style/house_style.py.

    That module exposes `load()` (reads house_style.yaml) and `to_rcparams(spec)` --
    there is no `apply_house_style()`. Point `HOUSE_STYLE_DIR` at the directory holding
    `house_style.py` (a sterile run vendors a copy at `<run_dir>/style/`); a few
    conventional locations are tried first. Returns None if the module can't be found or
    can't be imported (e.g. this interpreter lacks PyYAML) -- either way we fall back.
    """
    candidates = []
    if os.environ.get("HOUSE_STYLE_DIR"):
        candidates.append(Path(os.environ["HOUSE_STYLE_DIR"]))
    here = Path(__file__).resolve()
    candidates += [
        here.parents[3] / "style" if len(here.parents) > 3 else None,  # <run_dir>/style
        Path.home() / "Claude/projects/EngreitzLabAgents/style",
    ]
    for candidate in candidates:
        if candidate is None or not (candidate / "house_style.py").exists():
            continue
        sys.path.insert(0, str(candidate))
        try:
            from house_style import load, to_rcparams  # type: ignore

            return to_rcparams(load()["spec"])
        except Exception as error:  # missing dependency (e.g. PyYAML), bad yaml, etc.
            warnings.warn(f"found house_style.py at {candidate} but could not load it: {error}")
            return None
    return None


ARIAL_WARNED = False


def warn_once_if_arial_missing() -> None:
    global ARIAL_WARNED
    if ARIAL_WARNED:
        return
    ARIAL_WARNED = True
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Arial" not in available:
        warnings.warn(
            "Arial not found; falling back to Helvetica/DejaVu Sans. Figures may not match "
            "house style exactly -- install Arial or set HOUSE_STYLE_DIR to the lab style module."
        )


house_style_rcparams = load_house_style_rcparams()
if house_style_rcparams is not None:
    plt.rcParams.update(house_style_rcparams)
else:  # fall back to Nature-ish defaults
    warn_once_if_arial_missing()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
    })

DIVERGING = "RdBu_r"
MASK_FACE = "#d9d9d9"


def save_figure(fig: plt.Figure, plotted: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plotted.to_csv(out.with_suffix(".csv"), index=False)
    plt.close(fig)


def plot_effects_across_programs(effects: pd.DataFrame, targets: list[str], out: Path,
                                 programs: list[str] | None = None, q_threshold: float = 0.05,
                                 program_titles: dict[str, str] | None = None,
                                 max_programs: int = 20) -> pd.DataFrame:
    """Heatmap: hypothesis members (+ complex partners, + a control row) x the programs they move.

    Cells with q >= q_threshold are hatched grey, so white means a measured effect near zero,
    never "not significant".
    """
    subset = effects[effects["target"].isin(targets)]
    if programs is None:
        significant = subset[subset["qvalue"] < q_threshold]
        order = significant.groupby("program")["log2fc"].apply(lambda s: s.abs().max())
        programs = order.sort_values(ascending=False).index[:max_programs].tolist()
    if not programs:
        raise ValueError(f"no program is significant for {targets} at q < {q_threshold}")
    values = subset.pivot_table(index="target", columns="program", values="log2fc").reindex(index=targets, columns=programs)
    qvalues = subset.pivot_table(index="target", columns="program", values="qvalue").reindex(index=targets, columns=programs)
    masked = np.ma.masked_where(~(qvalues < q_threshold).to_numpy(), values.to_numpy())
    abs_values = np.abs(values.to_numpy())
    limit = float(np.nanmax(abs_values)) if np.isfinite(abs_values).any() else 1.0
    limit = limit or 1.0

    fig, ax = plt.subplots(figsize=(0.28 * len(programs) + 1.4, 0.24 * len(targets) + 1.0))
    ax.set_facecolor(MASK_FACE)
    image = ax.imshow(masked, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for i, j in zip(*np.where(masked.mask)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", lw=0, edgecolor="white"))
    titles = [program_titles.get(p, p) if program_titles else p for p in programs]
    ax.set_xticks(range(len(programs)), titles, rotation=60, ha="right")
    ax.set_yticks(range(len(targets)), targets)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    colorbar.set_label(f"log2FC (hatched: q ≥ {q_threshold})")
    plotted = values.reset_index().melt(id_vars="target", var_name="program", value_name="log2fc")
    plotted = plotted.merge(qvalues.reset_index().melt(id_vars="target", var_name="program", value_name="qvalue"))
    save_figure(fig, plotted, out)
    return plotted


JUNK_GENE_PREFIXES = ("MT-", "RPL", "RPS", "ENSG", "MALAT1", "mt-", "Rpl", "Rps", "ENSMUSG", "Malat1")


def plot_program_top_genes(loadings: pd.DataFrame, program: str, out: Path, n_genes: int = 20,
                           exclude_prefixes: tuple[str, ...] = JUNK_GENE_PREFIXES,
                           program_titles: dict[str, str] | None = None) -> pd.DataFrame:
    """Horizontal bars: the program's top genes by loading, junk transcripts filtered."""
    genes = loadings[(loadings["program"] == program) & ~loadings["gene"].str.startswith(exclude_prefixes)]
    top = genes.nlargest(n_genes, "loading").iloc[::-1]
    fig, ax = plt.subplots(figsize=(2.2, 0.13 * len(top) + 0.6))
    ax.barh(top["gene"], top["loading"], color="#525252", height=0.7)
    ax.set_xlabel("loading")
    ax.set_title(program_titles.get(program, program) if program_titles else program)
    ax.spines[["top", "right"]].set_visible(False)
    plotted = top.iloc[::-1][["program", "gene", "loading"]]
    save_figure(fig, plotted, out)
    return plotted


def plot_perturbation_neighbours(correlation: pd.DataFrame, target: str, out: Path, k: int = 8,
                                 highlight: list[str] | None = None) -> pd.DataFrame:
    """Bars: the k most positively and k most negatively correlated perturbations."""
    row = correlation.loc[target].drop(target).dropna().sort_values()
    picked = pd.concat([row.head(k), row.tail(k)])
    fig, ax = plt.subplots(figsize=(2.4, 0.13 * len(picked) + 0.6))
    # Correlation sign is not a driver/brake call, so neutral greys that differ in luminance.
    colors = ["#252525" if value > 0 else "#969696" for value in picked]
    bars = ax.barh(picked.index, picked.to_numpy(), color=colors, height=0.7)
    for bar, name in zip(bars, picked.index):
        if highlight and name in highlight:
            bar.set_hatch("....")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"correlation with {target} effect vector")
    ax.spines[["top", "right"]].set_visible(False)
    plotted = picked.rename("correlation").rename_axis("neighbour").reset_index().assign(target=target)
    save_figure(fig, plotted, out)
    return plotted


def plot_guide_concordance(guides: pd.DataFrame, target: str, programs: list[str], out: Path,
                           control_guides: pd.DataFrame | None = None,
                           program_titles: dict[str, str] | None = None) -> pd.DataFrame:
    """Dot plot: each guide's effect on the claimed programs, with the control-guide spread.

    A claim resting on one guide shows as one dot far from the others. Shapes differ per guide so
    the plot survives grayscale. The saved CSV includes the plotted per-guide rows plus, when
    `control_guides` is given, one row per program recording the control band's 5th/95th
    percentile (`guide` == "control_band_low"/"control_band_high").
    """
    subset = guides[(guides["target"] == target) & guides["program"].isin(programs)]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    fig, ax = plt.subplots(figsize=(0.55 * len(programs) + 1.2, 1.8))
    band_rows = []
    if control_guides is not None:
        spread = control_guides[control_guides["program"].isin(programs)].groupby("program")["log2fc"].quantile([0.05, 0.95]).unstack()
        for x, program in enumerate(programs):
            if program in spread.index:
                low, high = spread.loc[program, 0.05], spread.loc[program, 0.95]
                ax.fill_between([x - 0.35, x + 0.35], low, high, color=MASK_FACE, lw=0,
                                label="control guides, 5–95%" if x == 0 else None)
                band_rows.append({"guide": "control_band_low", "target": target, "program": program, "log2fc": low})
                band_rows.append({"guide": "control_band_high", "target": target, "program": program, "log2fc": high})
    for index, (guide, rows) in enumerate(subset.groupby("guide")):
        xs = [programs.index(p) + (index - 1) * 0.12 for p in rows["program"]]
        ax.scatter(xs, rows["log2fc"], marker=markers[index % len(markers)], s=14, color="black", label=guide, zorder=3)
    ax.axhline(0, color="black", lw=0.5)
    titles = [program_titles.get(p, p) if program_titles else p for p in programs]
    ax.set_xticks(range(len(programs)), titles, rotation=45, ha="right")
    ax.set_ylabel(f"{target} log2FC per guide")
    ax.legend(frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    plotted = pd.concat([subset, pd.DataFrame(band_rows)], ignore_index=True) if band_rows else subset
    save_figure(fig, plotted, out)
    return plotted


def plot_overview_map(effects: pd.DataFrame, ranked_targets: list[str], out: Path, **kwargs) -> pd.DataFrame:
    """The report's overview figure: the top-N regulators across the programs they move."""
    return plot_effects_across_programs(effects, ranked_targets, out, **kwargs)


def synthetic_screen(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    targets = [f"GENE{i}" for i in range(30)] + ["safe-targeting"]
    programs = [f"Program {i}" for i in range(12)]
    effects = pd.DataFrame([(t, p) for t in targets for p in programs], columns=["target", "program"])
    effects["log2fc"] = rng.normal(0, 0.2, len(effects))
    effects.loc[effects["target"].isin(["GENE1", "GENE2", "GENE3"]) & effects["program"].isin(programs[:3]), "log2fc"] += 1.2
    effects["qvalue"] = np.where(effects["log2fc"].abs() > 0.5, 0.01, 0.5)
    loadings = pd.DataFrame([(p, f"G{j}", rng.gamma(1.0)) for p in programs for j in range(60)], columns=["program", "gene", "loading"])
    wide = effects.pivot(index="target", columns="program", values="log2fc")
    correlation = wide.T.corr()
    guides = pd.DataFrame([(f"{t}_g{g}", t, p) for t in targets for g in range(3) for p in programs], columns=["guide", "target", "program"])
    guides = guides.merge(effects[["target", "program", "log2fc"]])
    guides["log2fc"] += rng.normal(0, 0.25, len(guides))
    return {"effects": effects, "loadings": loadings, "correlation": correlation, "guides": guides}


def run_self_test(out_dir: Path) -> None:
    data = synthetic_screen()
    members = ["GENE1", "GENE2", "GENE3", "safe-targeting"]
    plot_effects_across_programs(data["effects"], members, out_dir / "effects_across_programs")
    plot_program_top_genes(data["loadings"], "Program 0", out_dir / "program_top_genes")
    plot_perturbation_neighbours(data["correlation"], "GENE1", out_dir / "perturbation_neighbours", highlight=["GENE2", "GENE3"])
    controls = data["guides"][data["guides"]["target"] == "safe-targeting"]
    plot_guide_concordance(data["guides"], "GENE1", ["Program 0", "Program 1", "Program 2"], out_dir / "guide_concordance", controls)
    plot_overview_map(data["effects"], [f"GENE{i}" for i in range(1, 11)], out_dir / "overview_map")
    written = sorted(p.name for p in out_dir.iterdir())
    expected = sorted(
        f"{stem}.{suffix}"
        for stem in ("effects_across_programs", "program_top_genes", "perturbation_neighbours",
                     "guide_concordance", "overview_map")
        for suffix in ("pdf", "png", "csv")
    )
    assert written == expected, (written, expected)
    print(f"self-test OK: {len(written)} files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", type=Path, metavar="OUT_DIR")
    arguments = parser.parse_args()
    if arguments.self_test:
        run_self_test(arguments.self_test)
    else:
        parser.print_help()
