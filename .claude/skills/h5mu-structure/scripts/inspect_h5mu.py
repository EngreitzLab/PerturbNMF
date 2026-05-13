#!/usr/bin/env python3
"""
Inspect an .h5mu (MuData) file and write a structured txt summary.

Usage:
    python inspect_h5mu.py <path_to_file.h5mu>

Output:
    Writes <filename>_structure.txt in the same directory as the input file.
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd


def _fmt_value(x):
    """Format a single value for compact display."""
    if isinstance(x, float):
        if np.isnan(x):
            return "NaN"
        if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.3g}"
        return f"{x:.4g}"
    if isinstance(x, (bytes,)):
        try:
            return repr(x.decode())
        except Exception:
            return repr(x)
    if isinstance(x, str):
        return repr(x)
    return str(x)


def get_unique_examples(data, n=3):
    """Return a string showing up to n unique values from a column/array."""
    try:
        if isinstance(data, pd.Series):
            uniques = data.dropna().unique()[:n]
            return ", ".join(_fmt_value(x) for x in uniques)
        elif isinstance(data, pd.Categorical):
            uniques = list(data.categories[:n])
            return ", ".join(_fmt_value(x) for x in uniques)
        elif sp.issparse(data):
            flat = data.toarray().flatten()
            uniques = np.unique(flat[~np.isnan(flat)] if np.issubdtype(flat.dtype, np.floating) else flat)[:n]
            return ", ".join(_fmt_value(x) for x in uniques)
        elif isinstance(data, np.ndarray):
            flat = data.flatten()
            if np.issubdtype(flat.dtype, np.floating):
                uniques = np.unique(flat[~np.isnan(flat)])[:n]
            else:
                uniques = np.unique(flat)[:n]
            return ", ".join(_fmt_value(x) for x in uniques)
        elif isinstance(data, (list, tuple)):
            seen = []
            for x in data:
                if x not in seen:
                    seen.append(x)
                if len(seen) >= n:
                    break
            return ", ".join(_fmt_value(x) for x in seen)
        elif isinstance(data, dict):
            keys = list(data.keys())[:n]
            return ", ".join(_fmt_value(k) for k in keys)
        else:
            return repr(data)[:80]
    except Exception:
        return "..."


def get_head_examples(series, n=3):
    """Return a string showing the first n values (positional, NOT unique) from a Series."""
    try:
        values = list(series.iloc[:n]) if hasattr(series, "iloc") else list(series[:n])
        return ", ".join(_fmt_value(x) for x in values)
    except Exception:
        return "..."


def _safe_nunique(series):
    """Return number of unique values (dropna=True), or None if it can't be computed."""
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return None


def describe_column(series, n=3):
    """Return a compact description of a pandas Series: dtype + unique-count + n example values.

    Example:
        "categorical[3 unique]: 'A', 'B', 'C' | e.g. 'A', 'B', 'A'"
        "int64 [842 unique]: e.g. 12, 7, 33"
        "float32 [91866 unique]: e.g. 0.12, 0.45, 0.78"
    """
    try:
        n_unique = _safe_nunique(series)
        unique_tag = f"{n_unique} unique" if n_unique is not None else "? unique"

        # Categorical: show first n categories + first n head values
        if isinstance(series.dtype, pd.CategoricalDtype):
            n_cat = len(series.cat.categories)
            cats = list(series.cat.categories[:n])
            cat_str = ", ".join(_fmt_value(c) for c in cats)
            head_str = get_head_examples(series, n=n)
            return f"categorical[{n_cat} unique] cats: {cat_str} | e.g. {head_str}"

        dtype_name = str(series.dtype)
        head_str = get_head_examples(series, n=n)
        return f"{dtype_name} [{unique_tag}]: e.g. {head_str}"
    except Exception:
        return "..."


def describe_matrix(mat):
    """Return a human-readable description of a matrix (sparse or dense)."""
    if mat is None:
        return "None"
    if sp.issparse(mat):
        return f"sparse {type(mat).__name__}, shape={mat.shape}, dtype={mat.dtype}"
    if isinstance(mat, np.ndarray):
        return f"dense ndarray, shape={mat.shape}, dtype={mat.dtype}"
    return f"{type(mat).__name__}, shape={getattr(mat, 'shape', '?')}"


def describe_uns_value(v):
    """Return a human-readable description of a value stored in .uns."""
    if isinstance(v, np.ndarray):
        examples = get_unique_examples(v)
        return f"ndarray, shape={v.shape}, dtype={v.dtype}, unique e.g. {examples}"
    if isinstance(v, (list, tuple)):
        examples = get_unique_examples(v)
        return f"{type(v).__name__}, len={len(v)}, unique e.g. {examples}"
    if isinstance(v, dict):
        examples = get_unique_examples(v)
        return f"dict with {len(v)} keys, e.g. {examples}"
    if hasattr(v, "shape"):
        return f"{type(v).__name__}, shape={v.shape}"
    return f"{type(v).__name__}: {repr(v)[:80]}"


def describe_obsm_value(v):
    """Return a human-readable description of an obsm/varm entry."""
    examples = get_unique_examples(v)
    if sp.issparse(v):
        return f"sparse {type(v).__name__}, shape={v.shape}, unique e.g. {examples}"
    if isinstance(v, np.ndarray):
        return f"ndarray, shape={v.shape}, dtype={v.dtype}, unique e.g. {examples}"
    if hasattr(v, "shape"):
        return f"{type(v).__name__}, shape={v.shape}, unique e.g. {examples}"
    return f"{type(v).__name__}"


def tree_line(is_last, text, indent=""):
    """Return a tree-formatted line."""
    connector = "└── " if is_last else "├── "
    return f"{indent}{connector}{text}"


def tree_child_indent(is_last, indent=""):
    """Return the indent prefix for children of this node."""
    return indent + ("    " if is_last else "│   ")


def format_mapping_section(name, mapping, indent="", is_last_section=False):
    """Format a section like uns/, obsm/, layers/, etc."""
    lines = []
    items = list(mapping.keys()) if hasattr(mapping, "keys") else []
    if not items:
        lines.append(tree_line(is_last_section, f"{name}/ (empty)", indent))
    else:
        lines.append(tree_line(is_last_section, f"{name}/", indent))
        child_indent = tree_child_indent(is_last_section, indent)
        for i, key in enumerate(items):
            is_last_item = (i == len(items) - 1)
            val = mapping[key]
            if name in ("uns",):
                desc = describe_uns_value(val)
            else:
                desc = describe_obsm_value(val)
            lines.append(tree_line(is_last_item, f"{key} ({desc})", child_indent))
    return lines


def format_adata(adata, mod_name=None):
    """Format an AnnData object's structure as tree lines."""
    lines = []
    n_obs, n_var = adata.shape

    # X
    lines.append(tree_line(False, f"X ({describe_matrix(adata.X)})"))

    # obs
    obs_cols = list(adata.obs.columns)
    lines.append(tree_line(False, f"obs/ ({n_obs} observations)"))
    obs_indent = tree_child_indent(False)
    obs_names_examples = ", ".join(str(x) for x in adata.obs_names[:3])
    all_obs_items = ["obs_names"] + obs_cols
    for i, col in enumerate(all_obs_items):
        is_last = (i == len(all_obs_items) - 1)
        if col == "obs_names":
            lines.append(tree_line(is_last, f"obs_names: e.g. {obs_names_examples}", obs_indent))
        else:
            desc = describe_column(adata.obs[col])
            lines.append(tree_line(is_last, f"{col} — {desc}", obs_indent))

    # var
    var_cols = list(adata.var.columns)
    lines.append(tree_line(False, f"var/ ({n_var} variables)"))
    var_indent = tree_child_indent(False)
    var_names_examples = ", ".join(str(x) for x in adata.var_names[:3])
    all_var_items = ["var_names"] + var_cols
    for i, col in enumerate(all_var_items):
        is_last = (i == len(all_var_items) - 1)
        if col == "var_names":
            lines.append(tree_line(is_last, f"var_names: e.g. {var_names_examples}", var_indent))
        else:
            desc = describe_column(adata.var[col])
            lines.append(tree_line(is_last, f"{col} — {desc}", var_indent))

    # Remaining sections in order: uns, obsm, layers, obsp, varm, varp
    sections = [
        ("uns", adata.uns),
        ("obsm", adata.obsm),
        ("layers", adata.layers),
        ("obsp", adata.obsp),
        ("varm", adata.varm),
        ("varp", adata.varp),
    ]
    for i, (sec_name, sec_data) in enumerate(sections):
        is_last = (i == len(sections) - 1)
        lines.extend(format_mapping_section(sec_name, sec_data, indent="", is_last_section=is_last))

    return lines


def format_mudata(mdata, filename):
    """Format the full MuData structure."""
    lines = []
    lines.append(f"{filename} Structure")
    lines.append("=" * len(lines[0]))
    lines.append(f"Type: MuData")
    lines.append(f"Modalities: {list(mdata.mod.keys())}")
    lines.append(f"Total cells (obs): {mdata.n_obs}")
    lines.append(f"Total variables (var): {mdata.n_vars}")
    lines.append("")

    # Per-modality sections
    for mod_name, adata in mdata.mod.items():
        lines.append("=" * 80)
        lines.append(f"Modality: {mod_name}")
        lines.append("=" * 80)
        lines.append(f"Shape: ({adata.n_obs} cells, {adata.n_vars} variables)")
        lines.append("")
        lines.extend(format_adata(adata, mod_name))
        lines.append("")

    # MuData-level
    lines.append("=" * 80)
    lines.append("MuData-level (merged across modalities)")
    lines.append("=" * 80)
    lines.append("")

    # MuData obs columns
    obs_cols = list(mdata.obs.columns)
    if obs_cols:
        lines.append(tree_line(False, f"obs columns ({len(obs_cols)} total):"))
        obs_indent = tree_child_indent(False)
        for i, col in enumerate(obs_cols):
            desc = describe_column(mdata.obs[col])
            lines.append(tree_line(i == len(obs_cols) - 1, f"{col} — {desc}", obs_indent))
    else:
        lines.append(tree_line(False, "obs columns: (none beyond obs_names)"))

    # MuData var columns
    var_cols = list(mdata.var.columns)
    if var_cols:
        lines.append(tree_line(False, f"var columns ({len(var_cols)} total):"))
        var_indent = tree_child_indent(False)
        for i, col in enumerate(var_cols):
            desc = describe_column(mdata.var[col])
            lines.append(tree_line(i == len(var_cols) - 1, f"{col} — {desc}", var_indent))
    else:
        lines.append(tree_line(False, "var columns: (none beyond var_names)"))

    # MuData-level mappings
    sections = [
        ("obsm", mdata.obsm),
        ("uns", mdata.uns),
        ("obsp", mdata.obsp),
        ("varm", mdata.varm),
        ("varp", mdata.varp),
    ]
    for i, (sec_name, sec_data) in enumerate(sections):
        is_last = (i == len(sections) - 1)
        lines.extend(format_mapping_section(sec_name, sec_data, indent="", is_last_section=is_last))

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5mu.py <path_to_file.h5mu>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    import muon as mu

    print(f"Reading {filepath} ...", file=sys.stderr)
    mdata = mu.read(filepath)

    filename = os.path.basename(filepath)
    output_name = filename.replace(".h5mu", "") + "_structure.txt"
    output_path = os.path.join(os.path.dirname(filepath), output_name)

    content = format_mudata(mdata, filename)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Structure written to: {output_path}")
    # Also print to stdout for convenience
    print(content)


if __name__ == "__main__":
    main()
