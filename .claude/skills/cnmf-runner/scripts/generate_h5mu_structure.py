#!/usr/bin/env python3
"""
Generate a structured text summary of an .h5mu (MuData) file.

Produces a tree-formatted .txt file showing all modalities, their X matrices,
obs/var columns, uns, obsm, layers, obsp, varm, varp, and MuData-level attributes.

Output file is saved alongside the input with suffix '_structure.txt'.
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp


def format_shape_dtype(arr):
    """Return a description string for an array-like object."""
    if sp.issparse(arr):
        fmt = type(arr).__name__  # e.g. csr_matrix
        return f"sparse {fmt}, shape={arr.shape}, dtype={arr.dtype}"
    elif isinstance(arr, np.ndarray):
        return f"dense ndarray, shape={arr.shape}, dtype={arr.dtype}"
    elif hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
        return f"{type(arr).__name__}, shape={arr.shape}, dtype={arr.dtype}"
    else:
        return f"{type(arr).__name__}"


def format_examples(values, n=3):
    """Return first n example values as a comma-separated string."""
    vals = [str(v) for v in values[:n]]
    return ", ".join(vals)


def write_slot(lines, name, mapping, prefix, is_last):
    """Write a tree entry for a slot (uns, obsm, layers, obsp, varm, varp)."""
    connector = "\u2514" if is_last else "\u251c"
    continuation = " " if is_last else "\u2502"

    items = list(mapping.keys()) if hasattr(mapping, 'keys') else []
    if not items:
        lines.append(f"{prefix}{connector}\u2500\u2500 {name}/ (empty)")
    else:
        lines.append(f"{prefix}{connector}\u2500\u2500 {name}/")
        for i, key in enumerate(items):
            is_last_item = (i == len(items) - 1)
            item_connector = "\u2514" if is_last_item else "\u251c"
            val = mapping[key]
            desc = format_shape_dtype(val)
            # Add example values for 1-D object arrays
            example_str = ""
            if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 1:
                examples = format_examples(val, n=5)
                example_str = f", e.g. {examples}"
            lines.append(f"{prefix}{continuation}   {item_connector}\u2500\u2500 {key} ({desc}{example_str})")


def write_modality(lines, mod_key, adata):
    """Write the tree structure for a single modality (AnnData)."""
    lines.append("=" * 80)
    lines.append(f"Modality: {mod_key}")
    lines.append("=" * 80)
    lines.append(f"Shape: ({adata.n_obs} cells, {adata.n_vars} variables)")
    lines.append("")

    # X matrix
    if adata.X is not None:
        desc = format_shape_dtype(adata.X)
        lines.append(f"\u251c\u2500\u2500 X ({desc})")
    else:
        lines.append(f"\u251c\u2500\u2500 X (None)")

    # obs
    obs_cols = list(adata.obs.columns)
    lines.append(f"\u251c\u2500\u2500 obs/ ({adata.n_obs} observations)")
    obs_names_examples = format_examples(adata.obs_names, n=3)
    lines.append(f"\u2502   \u251c\u2500\u2500 obs_names: e.g. {obs_names_examples}")
    for i, col in enumerate(obs_cols):
        is_last = (i == len(obs_cols) - 1)
        connector = "\u2514" if is_last else "\u251c"
        lines.append(f"\u2502   {connector}\u2500\u2500 {col}")

    # var
    var_cols = list(adata.var.columns)
    lines.append(f"\u251c\u2500\u2500 var/ ({adata.n_vars} variables)")
    var_names_examples = format_examples(adata.var_names, n=3)
    if not var_cols:
        lines.append(f"\u2502   \u2514\u2500\u2500 var_names: e.g. {var_names_examples}")
    else:
        lines.append(f"\u2502   \u251c\u2500\u2500 var_names: e.g. {var_names_examples}")
        for i, col in enumerate(var_cols):
            is_last = (i == len(var_cols) - 1)
            connector = "\u2514" if is_last else "\u251c"
            lines.append(f"\u2502   {connector}\u2500\u2500 {col}")

    # uns, obsm, layers, obsp, varm, varp
    write_slot(lines, "uns", adata.uns, "", False)
    write_slot(lines, "obsm", adata.obsm, "", False)
    write_slot(lines, "layers", adata.layers, "", False)
    write_slot(lines, "obsp", adata.obsp, "", False)
    write_slot(lines, "varm", adata.varm, "", False)
    write_slot(lines, "varp", adata.varp, "", True)


def write_mudata_level(lines, mdata):
    """Write the MuData-level merged attributes."""
    lines.append("=" * 80)
    lines.append("MuData-level (merged across modalities)")
    lines.append("=" * 80)
    lines.append("")

    # obs columns
    obs_cols = list(mdata.obs.columns)
    lines.append(f"\u251c\u2500\u2500 obs columns ({len(obs_cols)} total):")
    for i, col in enumerate(obs_cols):
        is_last = (i == len(obs_cols) - 1)
        connector = "\u2514" if is_last else "\u251c"
        lines.append(f"\u2502   {connector}\u2500\u2500 {col}")

    # var columns
    var_cols = list(mdata.var.columns)
    lines.append(f"\u251c\u2500\u2500 var columns ({len(var_cols)} total):")
    for i, col in enumerate(var_cols):
        is_last = (i == len(var_cols) - 1)
        connector = "\u2514" if is_last else "\u251c"
        lines.append(f"\u2502   {connector}\u2500\u2500 {col}")

    # obsm
    write_slot(lines, "obsm", mdata.obsm, "", False)
    write_slot(lines, "uns", mdata.uns, "", False)
    write_slot(lines, "obsp", mdata.obsp, "", False)
    write_slot(lines, "varm", mdata.varm, "", False)
    write_slot(lines, "varp", mdata.varp, "", True)


def generate_structure(h5mu_path, output_path=None):
    """Generate the structure summary for an .h5mu file."""
    import muon as mu

    if output_path is None:
        base = os.path.splitext(h5mu_path)[0]
        output_path = f"{base}_structure.txt"

    print(f"Loading: {h5mu_path}")
    mdata = mu.read(h5mu_path)

    basename = os.path.splitext(os.path.basename(h5mu_path))[0]
    mod_keys = list(mdata.mod.keys())

    lines = []
    lines.append("")
    lines.append(f"{basename}.h5mu Structure")
    lines.append("=" * len(f"{basename}.h5mu Structure"))
    lines.append(f"Type: MuData")
    lines.append(f"Modalities: {mod_keys}")
    lines.append(f"Total cells (obs): {mdata.n_obs}")
    lines.append(f"Total variables (var): {mdata.n_vars}")
    lines.append("")

    for mod_key in mod_keys:
        write_modality(lines, mod_key, mdata.mod[mod_key])
        lines.append("")

    write_mudata_level(lines, mdata)
    lines.append("")

    text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(text)

    print(f"Structure written to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured text summary of an .h5mu file"
    )
    parser.add_argument(
        'h5mu_path', type=str,
        help='Path to the .h5mu file'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for the structure file (default: <input>_structure.txt)'
    )
    parser.add_argument(
        '--adata_dir', type=str, default=None,
        help='Directory containing .h5mu files. If provided, generates structure '
             'for all .h5mu files in this directory (h5mu_path is ignored).'
    )

    args = parser.parse_args()

    if args.adata_dir:
        # Batch mode: process all .h5mu files in the directory
        h5mu_files = sorted([
            os.path.join(args.adata_dir, f)
            for f in os.listdir(args.adata_dir)
            if f.endswith('.h5mu')
        ])
        if not h5mu_files:
            print(f"No .h5mu files found in {args.adata_dir}")
            sys.exit(1)
        print(f"Found {len(h5mu_files)} .h5mu file(s)")
        for path in h5mu_files:
            try:
                generate_structure(path)
            except Exception as e:
                print(f"ERROR processing {path}: {e}")
    else:
        if not os.path.exists(args.h5mu_path):
            print(f"ERROR: File not found: {args.h5mu_path}")
            sys.exit(1)
        generate_structure(args.h5mu_path, args.output)


if __name__ == '__main__':
    main()
