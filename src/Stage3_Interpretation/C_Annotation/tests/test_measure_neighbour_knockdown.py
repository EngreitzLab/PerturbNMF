"""measure_neighbour_knockdown.py on a synthetic h5mu: CSC guide matrix with explicit zeros, and
both control modes (single-target vs NTC; high-MOI complement)."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

SCRIPT = Path(__file__).resolve().parents[1] / "RegulatorGroupAnnotator" / "scripts" / "measure_neighbour_knockdown.py"


def write_sparse(group, name, matrix, encoding):
    node = group.create_group(name)
    node.attrs["encoding-type"] = encoding
    node.attrs["shape"] = matrix.shape
    node.create_dataset("data", data=matrix.data)
    node.create_dataset("indices", data=matrix.indices)
    node.create_dataset("indptr", data=matrix.indptr)


def build_h5mu(path: Path, high_moi: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_cells = 4000
    genes = ["Tgt", "Nbr", "Far", "Other"]
    guides = ["Tgt-1", "Tgt-2", "Other-1", "non-targeting-1"]
    targets = ["Tgt", "Tgt", "Other", "non-targeting"]
    assignment = np.zeros((n_cells, len(guides)))
    if high_moi:  # every cell carries several guides
        assignment = (rng.random((n_cells, len(guides))) < 0.3).astype(float)
    else:
        assignment[np.arange(n_cells), rng.integers(0, len(guides), n_cells)] = 1.0
    carrying = assignment[:, :2].max(axis=1) > 0
    expression = rng.poisson(20, size=(n_cells, len(genes))).astype(np.float32)
    expression[carrying, 0] *= 0.2  # target knocked down
    expression[carrying, 1] *= 0.3  # divergent neighbour knocked down too
    guide_matrix = csc_matrix(assignment)
    # explicit zeros stored in a CSC structure must not count as assignments
    guide_matrix = csc_matrix((np.r_[guide_matrix.data, np.zeros(50)],
                               (np.r_[guide_matrix.nonzero()[0], rng.integers(0, n_cells, 50)],
                                np.r_[guide_matrix.nonzero()[1], np.zeros(50, dtype=int)])),
                              shape=assignment.shape)
    with h5py.File(path, "w") as handle:
        rna = handle.create_group("mod/rna")
        write_sparse(rna, "X", csr_matrix(expression), "csr_matrix")
        var = rna.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(genes, dtype="S"))
        obs = rna.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"c{i}" for i in range(n_cells)], dtype="S"))
        obs.create_dataset("age_sex", data=np.array(["young_F", "aged_M"] * (n_cells // 2), dtype="S"))
        write_sparse(rna.create_group("obsm"), "guide_assignment", guide_matrix, "csc_matrix")
        rna.create_dataset("uns/guide_names", data=np.array(guides, dtype="S"))
        rna.create_dataset("uns/guide_targets", data=np.array(targets, dtype="S"))


def run(tmp_path: Path, high_moi: bool, control: str) -> pd.DataFrame:
    h5mu = tmp_path / "screen.h5mu"
    build_h5mu(h5mu, high_moi)
    coordinates = tmp_path / "coords.tsv"
    coordinates.write_text("Tgt\tchr1\t10000\t20000\t+\tprotein_coding\n"
                           "Nbr\tchr1\t5000\t9500\t-\tprotein_coding\n"
                           "Far\tchr1\t500000\t510000\t+\tprotein_coding\n"
                           "Other\tchr2\t10000\t20000\t+\tprotein_coding\n")
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"groups": [{"group_id": 1, "members": [{"gene": "Tgt"}]}]}))
    out = tmp_path / "knockdown.tsv"
    subprocess.run([sys.executable, str(SCRIPT), "--groups", str(groups), "--gene-coordinates", str(coordinates),
                    "--h5mu", str(h5mu), "--guide-prefix", "mod/rna", "--condition-key", "age_sex",
                    "--control", control, "--output", str(out)], check=True, capture_output=True, text=True)
    return pd.read_csv(out, sep="\t").set_index("gene")


def test_ntc_control_single_guide_cells(tmp_path):
    table = run(tmp_path, high_moi=False, control="ntc")
    assert table.loc["Tgt", "log2fc"] < -1.5 and table.loc["Tgt", "q_value"] < 1e-3
    assert table.loc["Nbr", "relation"] == "neighbour" and table.loc["Nbr", "log2fc"] < -1.0
    assert "Far" not in table.index


def test_complement_control_high_moi(tmp_path):
    table = run(tmp_path, high_moi=True, control="complement")
    assert table.loc["Tgt", "log2fc"] < -1.5 and table.loc["Tgt", "conditions"] == 2
    assert table.loc["Nbr", "log2fc"] < -1.0 and table.loc["Nbr", "q_value"] < 1e-3


def test_complement_uses_more_cells_than_single_target_arm_at_high_moi(tmp_path):
    """Why complement exists: at high MOI the single-target arm keeps only a fraction of the carriers."""
    single = run(tmp_path, high_moi=True, control="ntc").loc["Tgt", "n_target_cells"]
    complement = run(tmp_path, high_moi=True, control="complement").loc["Tgt", "n_target_cells"]
    assert complement > 1.5 * single
