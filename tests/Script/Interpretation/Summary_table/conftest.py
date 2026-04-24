"""Fixtures for Summary_table unit tests.

Provides synthetic MuData, enrichment DataFrames, and perturbation files
so tests run without real data.
"""
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy import sparse

TESTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = TESTS_ROOT / "output" / "Interpretation" / "Summary_table"


@pytest.fixture(scope="session")
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Synthetic MuData
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_mdata():
    """Build a minimal MuData with 50 cells, 20 genes, 3 programs, 3 guides."""
    import muon as mu
    import anndata as ad

    n_cells, n_genes, n_programs, n_guides = 50, 20, 3, 3
    rng = np.random.default_rng(42)

    # RNA modality
    X_rna = sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=42).astype(np.float32)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    cell_ids = [f"Cell{i}" for i in range(n_cells)]
    samples = rng.choice(["S1", "S2", "S3"], size=n_cells)
    obs_rna = pd.DataFrame({"sample": pd.Categorical(samples)}, index=cell_ids)
    var_rna = pd.DataFrame(index=gene_names)
    adata_rna = ad.AnnData(X=X_rna, obs=obs_rna, var=var_rna)

    # cNMF modality
    program_names = [str(i) for i in range(n_programs)]
    X_cnmf = rng.random((n_cells, n_programs)).astype(np.float32)
    obs_cnmf = pd.DataFrame(index=cell_ids)
    var_cnmf = pd.DataFrame(index=program_names)
    adata_cnmf = ad.AnnData(X=X_cnmf, obs=obs_cnmf, var=var_cnmf)

    # loadings: programs x genes
    loadings = rng.random((n_programs, n_genes)).astype(np.float32)
    adata_cnmf.varm["loadings"] = loadings
    adata_cnmf.uns["var_names"] = gene_names

    # guide assignment: cells x guides (sparse binary)
    guide_targets = gene_names[:n_guides]
    guide_assignment = np.zeros((n_cells, n_guides), dtype=np.float32)
    for i in range(n_cells):
        guide_assignment[i, i % n_guides] = 1.0
    adata_cnmf.obsm["guide_assignment"] = sparse.csr_matrix(guide_assignment)
    adata_cnmf.uns["guide_targets"] = guide_targets

    mdata = mu.MuData({"rna": adata_rna, "cNMF": adata_cnmf})
    return mdata


@pytest.fixture
def mdata_copy(synthetic_mdata):
    return synthetic_mdata.copy()


# ---------------------------------------------------------------------------
# Synthetic enrichment DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_go_df():
    """Minimal GO enrichment DataFrame matching Compile_GO_sheet output."""
    rows = []
    for prog in range(3):
        for t in range(5):
            rows.append({
                "program_name": str(prog),
                "Term": f"GO:00{prog}{t}",
                "Adjusted P-value": 0.001 * (t + 1),
                "Genes": ";".join([f"Gene{j}" for j in range(7)]),
                "Combined Score": 100 - t * 10,
            })
    df = pd.DataFrame(rows).set_index("Term")
    return df


@pytest.fixture(scope="session")
def synthetic_geneset_df():
    rows = []
    for prog in range(3):
        for t in range(3):
            rows.append({
                "program_name": str(prog),
                "Term": f"GENESET_{prog}_{t}",
                "Adjusted P-value": 0.01 * (t + 1),
                "Genes": ";".join([f"Gene{j}" for j in range(4)]),
            })
    df = pd.DataFrame(rows).set_index("Term")
    return df


@pytest.fixture(scope="session")
def synthetic_explained_variance_df():
    return pd.DataFrame({
        "program_name": [0, 1, 2],
        "variance_explained": [0.15, 0.10, 0.08],
    }).set_index("program_name")


# ---------------------------------------------------------------------------
# Synthetic perturbation files on disk
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def perturbation_dir(output_dir):
    """Write synthetic perturbation TSV files and return base path (without _Sample.txt)."""
    perturb_dir = output_dir / "perturbation_data"
    perturb_dir.mkdir(parents=True, exist_ok=True)
    base = perturb_dir / "3_perturbation_association_results"

    rng = np.random.default_rng(99)
    targets = [f"Gene{i}" for i in range(5)]
    programs = [str(p) for p in range(3)]

    for samp in ["S1", "S2", "S3"]:
        rows = []
        for tgt in targets:
            for prog in programs:
                rows.append({
                    "target_name": tgt,
                    "program_name": prog,
                    "log2FC": rng.normal(0, 1),
                    "adj_pval": rng.uniform(0, 0.2),
                })
        df = pd.DataFrame(rows)
        df.to_csv(f"{base}_{samp}.txt", sep="\t", index=True)

    return str(base)


@pytest.fixture(scope="session")
def sample_list():
    return ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------
# Synthetic program loading flat DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_program_loading_flat(synthetic_mdata):
    """Mimics output of compile_Program_loading_score_sheet_flat."""
    mdata = synthetic_mdata
    loadings = pd.DataFrame(
        data=mdata["cNMF"].varm["loadings"],
        columns=mdata["cNMF"].uns["var_names"],
        index=mdata["cNMF"].var_names,
    )
    n_genes = loadings.shape[1]
    top_df = loadings.apply(lambda row: row.nlargest(n_genes).index.tolist(), axis=1)
    result = pd.DataFrame(top_df.tolist(), columns=range(1, n_genes + 1))
    result.index = loadings.index
    result.index.name = "Program"
    return result
