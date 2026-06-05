#!/usr/bin/env python3
"""
Data format validation for cNMF pipeline.

Checks input .h5ad or .h5mu files for required structure, reports dataset
statistics (cell count, gene count, sparsity, file size), and validates
guide information when present.

Exit code 0 = valid, 1 = issues found.
"""

import argparse
import os
import sys

# Add pipeline root to path
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PIPELINE_ROOT)


def get_file_size_str(path):
    """Return human-readable file size."""
    size_bytes = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}", size_bytes
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB", size_bytes


def compute_sparsity(X):
    """Compute fraction of zeros in the matrix."""
    import scipy.sparse as sp
    import numpy as np
    if sp.issparse(X):
        nnz = X.nnz
        total = X.shape[0] * X.shape[1]
    else:
        nnz = np.count_nonzero(X)
        total = X.size
    return 1.0 - (nnz / total) if total > 0 else 0.0


def check_data_format(adata, guide_names_key="guide_names", guide_targets_key="guide_targets",
                      categorical_key="batch", guide_assignment_key="guide_assignment"):
    """Validate that an AnnData has the keys cNMF inference relies on."""
    is_valid = True
    if categorical_key not in adata.obs:
        print(f"WARNING: Not found in adata.obs['{categorical_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.obs['{categorical_key}']\n")

    if guide_names_key not in adata.uns:
        print(f"WARNING: Not found in adata.uns['{guide_names_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.uns['{guide_names_key}']\n")

    if guide_targets_key not in adata.uns:
        print(f"WARNING: Not found in adata.uns['{guide_targets_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.uns['{guide_targets_key}']\n")

    if "X_pca" not in adata.obsm:
        print("WARNING: Not found adata.obsm['X_pca']\n")
        is_valid = False
    else:
        print("Found adata.obsm['X_pca']\n")

    if "X_umap" not in adata.obsm:
        print("WARNING: Not found adata.obsm['X_umap']\n")
        is_valid = False
    else:
        print("Found adata.obsm['X_umap']\n")

    if guide_assignment_key not in adata.obsm:
        print(f"WARNING: Not found adata.obsm['{guide_assignment_key}']\n")
        is_valid = False
    else:
        guide_assignment = adata.obsm[guide_assignment_key]
        print(f"Found adata.obsm['{guide_assignment_key}']\n")
        try:
            import scipy.sparse as sp
            if sp.issparse(guide_assignment):
                print(f"WARNING: '{guide_assignment_key}' is sparse. Converting to dense array...")
                dense_array = guide_assignment.toarray()
                adata.obsm[guide_assignment_key] = dense_array
                print(f"'{guide_assignment_key}' converted to dense array (shape: {dense_array.shape})\n")
            else:
                print(f"'{guide_assignment_key}' is already dense (shape: {guide_assignment.shape})\n")
        except Exception as e:
            print(f"WARNING: Error checking '{guide_assignment_key}' sparsity: {e}\n")
            is_valid = False
    return is_valid


def _validate_against_reference_gtf(rna_gene_names, gtf_path):
    """Compare RNA var_names against gene_name attributes extracted from a GTF."""
    import pandas as pd
    validation = {"is_valid": True, "gtf_genes": set(), "not_in_gtf": []}
    try:
        gtf_df = pd.read_csv(
            gtf_path, sep="\t", comment="#", header=None,
            dtype={0: str, 1: str, 2: str, 3: int, 4: int, 5: str, 6: str, 7: str, 8: str},
        )
        extracted = gtf_df[8].str.extract(r'gene_name "([^"]+)"')[0]
        if extracted.notna().sum() == 0:
            print("gene_name not found in any rows\n")
        else:
            print(f"Successfully extracted gene_name from {extracted.notna().sum()} rows")
            extracted = set(extracted)
            validation["gtf_genes"] = extracted

        rna_not_in_gtf = rna_gene_names - extracted
        print(f"Found {len(rna_gene_names)} gene names in adata.var_name")
        if rna_not_in_gtf:
            print(f"WARNING: {len(rna_not_in_gtf)}/{len(rna_gene_names)} adata.var RNA NOT found in reference GTF."
                  "\n This might be caused by the adata.var RNA being Ensembl ID instead of gene symbols.")
            print(f"Examples: {list(rna_not_in_gtf)[:5]}\n")
            validation["not_in_gtf"] = sorted(list(rna_not_in_gtf))
        else:
            print("All RNA genes found in reference GTF\n")
    except FileNotFoundError:
        print(f"ERROR: Reference GTF file not found at {gtf_path}\n")
        validation["is_valid"] = False
    except Exception as e:
        print(f"ERROR: Failed to read reference GTF: {e}\n")
        validation["is_valid"] = False
    return validation


def check_guide_names(adata, guide_names_key="guide_names", guide_targets_key="guide_targets",
                     categorical_key="batch", reference_gtf_path=None, guide_annotation_path=None,
                     guide_assignment_key="guide_assignment", check_var_names_align_with_guide_targets=True):
    """Validate guide/RNA name consistency, optionally against a GTF and/or guide annotation TSV."""
    import pandas as pd

    results = {"is_valid": True, "missing_from_rna": [], "gtf_validation": None}
    results["is_valid"] = check_data_format(
        adata, guide_names_key=guide_names_key, guide_targets_key=guide_targets_key,
        categorical_key=categorical_key, guide_assignment_key=guide_assignment_key,
    )

    rna_gene_names = set(adata.var_names)
    guide_targets = adata.uns[guide_targets_key]
    guide_names = adata.uns[guide_names_key]

    if isinstance(guide_targets, (list, tuple)):
        guide_gene_targets = set(guide_targets)
        guide_gene_names = set(guide_names)
    elif isinstance(guide_targets, (pd.Index, pd.Series)):
        guide_gene_targets = set(guide_targets.values)
        guide_gene_names = set(guide_names.values)
    else:
        guide_gene_targets = set(guide_targets)
        guide_gene_names = set(guide_names)

    if len(guide_targets) != len(guide_names) and len(guide_gene_targets) != adata.obsm[guide_assignment_key].shape[1]:
        print("WARNING: guide_targets and guide_names and guide_assignment col should be equal length but not in here\n")
        results["is_valid"] = False

    if check_var_names_align_with_guide_targets:
        guide_not_in_rna = guide_gene_targets - rna_gene_names
        print(f"Found {len(guide_gene_targets)} genes in gene_targets")
        if guide_not_in_rna:
            print(f"WARNING: {len(guide_not_in_rna)}/{len(guide_gene_targets)} gene names in guide_targets but NOT in adata.var.\n"
                  "                This might be caused by mismatch between gene symbol vs Ensembl ID or some perturbed genes are not expressed in the dataset."
                  " Might cause issue in evalutation or plotting if naming conventions are different.")
            print(f"Examples: {list(guide_not_in_rna)[:5]}\n")
            results["missing_from_rna"] = sorted(list(guide_not_in_rna))
        else:
            print("All guide_targets found in RNA gene names\n")

    if reference_gtf_path is None:
        print("No reference GTF provided (optional check skipped)")
        results["gtf_validation"] = None
    else:
        gtf_validation = _validate_against_reference_gtf(rna_gene_names, reference_gtf_path)
        results["gtf_validation"] = gtf_validation
        if not gtf_validation["is_valid"]:
            results["is_valid"] = False

    if guide_annotation_path is None:
        print("\nNo reference guide annotation provided (optional check skipped)")
        results["guide_annotation"] = None
    else:
        guide_annotation_df = pd.read_csv(guide_annotation_path, sep="\t", index_col=0)
        if guide_names_key not in guide_annotation_df.columns:
            print(f"{guide_names_key} not in guide annotation file")
            results["is_valid"] = False
            return results
        annotation_guide_names = set(guide_annotation_df[guide_names_key].values)

        if guide_targets_key not in guide_annotation_df.columns:
            results["is_valid"] = False
            print(f"{guide_targets_key} not in guide annotation file")
            return results
        annotation_guide_targets = set(guide_annotation_df[guide_targets_key].values)

        if "targeting" not in guide_annotation_df.columns:
            results["is_valid"] = False
            print("'targeting' not in guide annotation file")

        guide_targets_not_in_annotation = guide_gene_targets - annotation_guide_targets
        guide_names_not_in_annotation = guide_gene_names - annotation_guide_names
        if guide_targets_not_in_annotation or guide_names_not_in_annotation:
            if guide_targets_not_in_annotation:
                print(f"\n WARNING: {len(guide_targets_not_in_annotation)}/{len(guide_gene_targets)} guide targets in data but NOT in annotation.")
                print(f"Examples: {list(guide_targets_not_in_annotation)[:5]}\n")
            if guide_names_not_in_annotation:
                print(f"\n WARNING: {len(guide_names_not_in_annotation)}/{len(guide_gene_names)} guide names in data but NOT in annotation.")
                print(f"Examples: {list(guide_names_not_in_annotation)[:5]}\n")
            results["is_valid"] = False
        else:
            print("All guide_targets and guide_names match guide annotation\n")

    return results


def validate_adata(adata, args):
    """Validate an AnnData object and print report."""
    import scipy.sparse as sp

    issues = []
    print("\n=== Dataset Summary ===")
    print(f"  Cells:     {adata.n_obs:,}")
    print(f"  Genes:     {adata.n_vars:,}")

    # Matrix info
    is_sparse = sp.issparse(adata.X)
    print(f"  Sparse:    {is_sparse}")
    if adata.X is not None:
        sparsity = compute_sparsity(adata.X)
        print(f"  Sparsity:  {sparsity:.2%}")

    # obs columns
    print(f"\n=== obs columns ({len(adata.obs.columns)}) ===")
    for col in adata.obs.columns[:20]:
        print(f"  - {col}")
    if len(adata.obs.columns) > 20:
        print(f"  ... and {len(adata.obs.columns) - 20} more")

    # Check categorical key
    if args.categorical_key not in adata.obs:
        issues.append(f"Missing obs['{args.categorical_key}'] (categorical/sample key)")
    else:
        n_cats = adata.obs[args.categorical_key].nunique()
        print(f"\n  obs['{args.categorical_key}']: {n_cats} unique values")

    # Check obsm
    print(f"\n=== obsm keys ===")
    for key in adata.obsm:
        shape = adata.obsm[key].shape if hasattr(adata.obsm[key], 'shape') else 'N/A'
        print(f"  - {key}: {shape}")

    has_pca = 'X_pca' in adata.obsm
    has_umap = 'X_umap' in adata.obsm
    if not has_pca:
        issues.append("Missing obsm['X_pca']")
    if not has_umap:
        issues.append("Missing obsm['X_umap']")

    # Check guide info
    print(f"\n=== Guide Information ===")
    has_guide_names = args.guide_names_key in adata.uns
    has_guide_targets = args.guide_targets_key in adata.uns
    has_guide_assignment = args.guide_assignment_key in adata.obsm

    if has_guide_names:
        n_guides = len(adata.uns[args.guide_names_key])
        print(f"  uns['{args.guide_names_key}']: {n_guides} guides")
    else:
        print(f"  uns['{args.guide_names_key}']: NOT FOUND")
        issues.append(f"Missing uns['{args.guide_names_key}']")

    if has_guide_targets:
        targets = adata.uns[args.guide_targets_key]
        if hasattr(targets, '__len__'):
            n_targets = len(set(targets))
            print(f"  uns['{args.guide_targets_key}']: {n_targets} unique targets")
        else:
            print(f"  uns['{args.guide_targets_key}']: present")
    else:
        print(f"  uns['{args.guide_targets_key}']: NOT FOUND")
        issues.append(f"Missing uns['{args.guide_targets_key}']")

    if has_guide_assignment:
        ga = adata.obsm[args.guide_assignment_key]
        ga_shape = ga.shape if hasattr(ga, 'shape') else 'N/A'
        ga_sparse = sp.issparse(ga)
        print(f"  obsm['{args.guide_assignment_key}']: {ga_shape} (sparse={ga_sparse})")
    else:
        print(f"  obsm['{args.guide_assignment_key}']: NOT FOUND")
        issues.append(f"Missing obsm['{args.guide_assignment_key}']")

    # Consistency check
    if has_guide_names and has_guide_assignment:
        n_guides = len(adata.uns[args.guide_names_key])
        n_cols = adata.obsm[args.guide_assignment_key].shape[1]
        if n_guides != n_cols:
            issues.append(
                f"guide_names length ({n_guides}) != guide_assignment columns ({n_cols})"
            )

    return issues


def validate_mdata(mdata, args):
    """Validate a MuData object."""
    issues = []
    print(f"\n=== MuData Modalities ===")
    for mod_key in mdata.mod:
        mod = mdata.mod[mod_key]
        print(f"  [{mod_key}]: {mod.n_obs:,} cells x {mod.n_vars:,} features")

    # Validate RNA modality
    if args.data_key in mdata.mod:
        print(f"\n--- Validating '{args.data_key}' modality ---")
        issues.extend(validate_adata(mdata[args.data_key], args))
    else:
        issues.append(f"Missing modality '{args.data_key}'")

    # Check cNMF modality if present (post-inference)
    if args.prog_key in mdata.mod:
        print(f"\n--- Found '{args.prog_key}' modality ---")
        cnmf_adata = mdata[args.prog_key]
        print(f"  Programs: {cnmf_adata.n_vars}")
        if 'loadings' in cnmf_adata.varm:
            print(f"  varm['loadings']: {cnmf_adata.varm['loadings'].shape}")
        else:
            print(f"  varm['loadings']: NOT FOUND (expected post-inference)")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate input data for cNMF pipeline"
    )
    parser.add_argument(
        '--counts_fn', type=str, required=True,
        help='Path to input data file (.h5ad or .h5mu)'
    )
    parser.add_argument(
        '--categorical_key', type=str, default='sample',
        help='Key in .obs for categorical/sample variable (default: sample)'
    )
    parser.add_argument(
        '--guide_names_key', type=str, default='guide_names',
        help='Key in .uns for guide names (default: guide_names)'
    )
    parser.add_argument(
        '--guide_targets_key', type=str, default='guide_targets',
        help='Key in .uns for guide targets (default: guide_targets)'
    )
    parser.add_argument(
        '--guide_assignment_key', type=str, default='guide_assignment',
        help='Key in .obsm for guide assignment matrix (default: guide_assignment)'
    )
    parser.add_argument(
        '--data_key', type=str, default='rna',
        help='Key for RNA modality in MuData (default: rna)'
    )
    parser.add_argument(
        '--prog_key', type=str, default='cNMF',
        help='Key for cNMF modality in MuData (default: cNMF)'
    )
    parser.add_argument(
        '--reference_gtf_path', type=str, default=None,
        help='Path to reference GTF for gene name validation (optional)'
    )
    parser.add_argument(
        '--guide_annotation_path', type=str, default=None,
        help='Path to guide annotation TSV file (optional)'
    )

    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.counts_fn):
        print(f"ERROR: File not found: {args.counts_fn}")
        sys.exit(1)

    # File size
    size_str, size_bytes = get_file_size_str(args.counts_fn)
    print(f"File: {args.counts_fn}")
    print(f"Size: {size_str}")

    ext = os.path.splitext(args.counts_fn)[1].lower()

    # Load and validate
    issues = []
    if ext == '.h5mu':
        import muon as mu
        print("Loading MuData...")
        mdata = mu.read(args.counts_fn)
        issues = validate_mdata(mdata, args)
    elif ext in ('.h5ad', '.h5'):
        import scanpy as sc
        print("Loading AnnData...")
        adata = sc.read_h5ad(args.counts_fn)
        issues = validate_adata(adata, args)
    else:
        print(f"ERROR: Unsupported file extension '{ext}'. Expected .h5ad or .h5mu")
        sys.exit(1)

    # Optional: run full format checking from pipeline
    if args.reference_gtf_path or args.guide_annotation_path:
        try:
            if ext == '.h5mu':
                adata_check = mdata[args.data_key]
            else:
                adata_check = adata
            result = check_guide_names(
                adata_check,
                guide_names_key=args.guide_names_key,
                guide_targets_key=args.guide_targets_key,
                categorical_key=args.categorical_key,
                reference_gtf_path=args.reference_gtf_path,
                guide_annotation_path=args.guide_annotation_path,
                guide_assignment_key=args.guide_assignment_key
            )
            if not result.get('is_valid', True) if isinstance(result, dict) else not result:
                issues.append("Format check reported validation failures (see output above)")
        except Exception as e:
            print(f"\nWARNING: Could not run extended format checks: {e}")

    # Resource recommendations
    if ext == '.h5mu':
        n_cells = mdata[args.data_key].n_obs if args.data_key in mdata.mod else 0
        n_genes = mdata[args.data_key].n_vars if args.data_key in mdata.mod else 0
    else:
        n_cells = adata.n_obs
        n_genes = adata.n_vars

    print(f"\n=== Resource Recommendations ===")
    if n_cells < 50000:
        mem_rec, time_rec_sk, time_rec_torch = "64G", "4-6h", "2-4h"
    elif n_cells < 200000:
        mem_rec, time_rec_sk, time_rec_torch = "128-256G", "8-14h", "6-10h"
    elif n_cells < 500000:
        mem_rec, time_rec_sk, time_rec_torch = "256-512G", "14-24h", "10-15h"
    else:
        mem_rec, time_rec_sk, time_rec_torch = "512-786G", "24h+", "15h+"

    print(f"  Cells: {n_cells:,} | Genes: {n_genes:,}")
    print(f"  sk-cNMF:    mem={mem_rec}, time={time_rec_sk} (numiter=10, 8 K values)")
    print(f"  torch-cNMF: mem={'64G' if n_cells < 100000 else '128G' if n_cells < 300000 else '256G'}, time={time_rec_torch}")
    partition = "engreitz,owners" if n_cells < 200000 else "engreitz,owners,bigmem"
    print(f"  Partition:  {partition}")

    # Report
    print("\n" + "=" * 60)
    if issues:
        print(f"VALIDATION: {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    else:
        print("VALIDATION: All checks passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
