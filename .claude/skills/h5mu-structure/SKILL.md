---
name: h5mu-structure
description: >
  Inspect an .h5mu (MuData) file and generate a structured txt summary of its contents
  in a tree format. Lists all modalities, their X matrices, obs/var columns, uns, obsm,
  layers, obsp, varm, varp, and MuData-level attributes. Use this skill whenever the user
  asks to inspect, describe, summarize, or explore the structure of a .h5mu file, or when
  they want to know what's inside a MuData object saved on disk. Also trigger when the user
  says things like "what's in this h5mu", "show me the structure", "list the columns",
  "describe this mudata", or references .h5mu files and wants to understand their contents.
---

# h5mu Structure Inspector

When the user wants to inspect or summarize the structure of an `.h5mu` file, run the
bundled Python script to generate a detailed tree-format summary.

## How to use

1. Identify the `.h5mu` file path the user wants to inspect.
2. Run the inspection script:

```bash
python3 SKILL_DIR/scripts/inspect_h5mu.py <path_to_file.h5mu>
```

Replace `SKILL_DIR` with the actual path to this skill's directory (where this SKILL.md lives).

3. The script writes `<filename>_structure.txt` in the same directory as the input file
   and also prints the output to stdout.

4. Share the key findings with the user — modalities found, shapes, notable obs/var columns,
   what's stored in uns/obsm, etc.

## What the output contains

For each modality in the MuData:
- **X**: matrix type (sparse/dense), shape, dtype
- **obs/**: all column names, with example obs_names (cell barcodes); each column annotated with its dtype/cardinality and **3 example values** from the first 3 cells
- **var/**: all column names, with example var_names (gene IDs); each column annotated with its dtype/cardinality and **3 example values** from the first 3 genes
- **uns/**: each key with its type, shape, and example values
- **obsm/**: each key with type and shape (e.g., guide_assignment, X_PCA, X_umap)
- **layers/**, **obsp/**, **varm/**, **varp/**: listed if non-empty

Plus a MuData-level section showing merged obs/var columns (with 3 example values per column) and top-level obsm/uns/etc.

### Column annotation format

Each obs/var column line looks like:
```
├── <col_name> — <dtype>: e.g. <v1>, <v2>, <v3>
```

For categorical columns, the cardinality and first 3 categories are also shown:
```
├── batch — categorical[26] cats: 'IGVFDS6990PWDJ', 'IGVFDS7696ONVJ', 'IGVFDS3...' | e.g. 'IGVFDS6990PWDJ', 'IGVFDS7696ONVJ', 'IGVFDS6990PWDJ'
```

## Requirements

- Python 3 with `muon`, `anndata`, `numpy`, `scipy` installed
- The environment on Stanford Sherlock HPC already has these available

## Example output

```
cNMF_5_0_2.h5mu Structure
=========================
Type: MuData
Modalities: ['rna', 'cNMF']
Total cells (obs): 91866

================================================================================
Modality: rna
================================================================================
Shape: (91866 cells, 9397 variables)

├── X (sparse csr_matrix, shape=(91866, 9397), dtype=float32)
├── obs/ (91866 observations)
│   ├── obs_names: e.g. CGTTCTGCAGCGATCC_0, ATAACGCCATGCCTTC_0, GACACGCTCGTCCGTT_0
│   ├── batch — categorical[26] cats: 'IGVFDS6990PWDJ', 'IGVFDS7696ONVJ', 'IGVFDS3...' | e.g. 'IGVFDS6990PWDJ', 'IGVFDS7696ONVJ', 'IGVFDS6990PWDJ'
│   ├── n_counts — int32: e.g. 2713, 1773, 5421
│   └── ...
├── var/ (9397 variables)
│   ├── var_names: e.g. ENSG00000230021, ENSG00000228794, SAMD11
│   ├── symbol — object: e.g. 'LINC01128', 'SAMD11', 'NOC2L'
│   └── ...
├── uns/
│   ├── guide_names (ndarray, shape=(416,), ...)
│   └── guide_targets (ndarray, shape=(416,), ...)
├── obsm/
│   └── guide_assignment (sparse csr_matrix, shape=(91866, 416))
...
```
