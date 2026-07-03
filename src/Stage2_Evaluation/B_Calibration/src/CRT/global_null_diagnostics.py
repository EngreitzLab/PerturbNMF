"""
Diagnostics helpers for global-null calibration checks.
"""

import json
import logging
import hashlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew

from .adata_utils import union_obs_idx_from_cols
from .crt import crt_betas_for_gene
from .pipeline_helpers import (
    _compute_skew_normal_pvals,
    _fit_propensity,
    _raw_pvals_from_betas,
    _sample_crt_indices,
)
from .propensity import fit_propensity_logistic

logger = logging.getLogger(__name__)


def oracle_propensity_const(C: np.ndarray, y01: np.ndarray) -> np.ndarray:
    """
    Oracle propensity: constant p_i = mean(x) for the unit.
    """
    p = float(np.mean(y01))
    return np.full(C.shape[0], p, dtype=np.float64)


def _stable_seed(base_seed: int, token: str) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(h, "little") ^ int(base_seed)
    return seed & 0xFFFFFFFF


def _propensity_auc(y01: np.ndarray, p_hat: np.ndarray) -> float:
    y = np.asarray(y01, dtype=np.int8)
    if np.unique(y).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, p_hat))
    except Exception:
        return float("nan")


def _x_from_obs_idx(n_cells: int, obs_idx: np.ndarray) -> np.ndarray:
    x = np.zeros(n_cells, dtype=np.float64)
    if obs_idx.size:
        x[obs_idx] = 1.0
    return x


def compute_unit_diagnostics(
    *,
    inputs: Any,
    unit_id: str,
    unit_type: str,
    obs_idx: np.ndarray,
    guide_count: int,
    B: int,
    base_seed: int,
    propensity_model: Callable = fit_propensity_logistic,
    skew_side_code: int = 0,
) -> Dict[str, Any]:
    """
    Compute diagnostics for a single unit (gene or NTC group).
    """
    n_cells = inputs.C.shape[0]
    n_treated = int(obs_idx.size)
    n_control = int(n_cells - n_treated)
    x_mean = float(n_treated) / float(n_cells)

    x = _x_from_obs_idx(n_cells, obs_idx)
    u = inputs.C.T @ x
    u_norm = float(np.linalg.norm(u))
    den = float(n_treated - u.T @ (inputs.A @ u))
    den_is_bad = bool(den <= 1e-12)

    y01 = (x > 0).astype(np.int8)
    p_hat = _fit_propensity(inputs, obs_idx, propensity_model)
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_hat_mean = float(np.mean(p_hat))
    p_hat_min = float(np.min(p_hat))
    p_hat_max = float(np.max(p_hat))
    p_hat_var = float(np.var(p_hat))
    propensity_separation_flag = bool(
        p_hat_min < 1.0e-4 or p_hat_max > 1.0 - 1.0e-4
    )
    propensity_auc = _propensity_auc(y01, p_hat)

    if n_treated == 0 or n_treated == n_cells:
        return {
            "unit_id": unit_id,
            "unit_type": unit_type,
            "guide_count": guide_count,
            "x_mean": x_mean,
            "n_treated": n_treated,
            "n_control": n_control,
            "u_norm": u_norm,
            "den": den,
            "den_is_bad": den_is_bad,
            "beta_obs_nan": True,
            "pmin_raw": float("nan"),
            "pvals_raw_eq1_frac": float("nan"),
            "pmin_skew": float("nan"),
            "pvals_skew_eq1_frac": float("nan"),
            "p_hat_mean": p_hat_mean,
            "p_hat_min": p_hat_min,
            "p_hat_max": p_hat_max,
            "p_hat_var": p_hat_var,
            "propensity_separation_flag": propensity_separation_flag,
            "propensity_auc": propensity_auc,
            "null_nan_frac": float("nan"),
            "beta_null_mean_k0": float("nan"),
            "beta_null_sd_k0": float("nan"),
            "beta_null_skew_k0": float("nan"),
            "z_obs_k0": float("nan"),
            "p_raw_k0": float("nan"),
            "p_skew_k0": float("nan"),
            "seed_base": int(base_seed),
            "skip_reason": "all_or_none",
        }

    seed = _stable_seed(base_seed, f"{unit_type}:{unit_id}")
    indptr, idx = _sample_crt_indices(p_hat, B, seed)
    beta_obs, beta_null = crt_betas_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )
    pvals_raw = _raw_pvals_from_betas(beta_obs, beta_null)
    pvals_skew, _ = _compute_skew_normal_pvals(beta_obs, beta_null, skew_side_code)

    beta_obs_nan = bool(np.any(~np.isfinite(beta_obs)))
    pmin_raw = float(np.min(pvals_raw))
    pvals_raw_eq1_frac = float(np.mean(pvals_raw >= 1.0 - 1e-12))
    pmin_skew = float(np.min(pvals_skew))
    pvals_skew_eq1_frac = float(np.mean(pvals_skew >= 1.0 - 1e-12))

    null_nan_frac = float(np.mean(np.any(~np.isfinite(beta_null), axis=1)))
    beta_null_k0 = beta_null[:, 0]
    beta_null_mean_k0 = float(np.mean(beta_null_k0))
    beta_null_sd_k0 = float(np.std(beta_null_k0))
    if beta_null_sd_k0 > 0.0 and np.isfinite(beta_null_sd_k0):
        z_null = (beta_null_k0 - beta_null_mean_k0) / beta_null_sd_k0
        null_z_mean_k0 = float(np.mean(z_null))
        null_z_sd_k0 = float(np.std(z_null))
        null_z_skew_k0 = float(skew(z_null, bias=False))
        z_obs_k0 = float((beta_obs[0] - beta_null_mean_k0) / beta_null_sd_k0)
    else:
        null_z_mean_k0 = float("nan")
        null_z_sd_k0 = float("nan")
        null_z_skew_k0 = float("nan")
        z_obs_k0 = float("nan")

    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "guide_count": guide_count,
        "x_mean": x_mean,
        "n_treated": n_treated,
        "n_control": n_control,
        "u_norm": u_norm,
        "den": den,
        "den_is_bad": den_is_bad,
        "beta_obs_nan": beta_obs_nan,
        "pmin_raw": pmin_raw,
        "pvals_raw_eq1_frac": pvals_raw_eq1_frac,
        "pmin_skew": pmin_skew,
        "pvals_skew_eq1_frac": pvals_skew_eq1_frac,
        "p_hat_mean": p_hat_mean,
        "p_hat_min": p_hat_min,
        "p_hat_max": p_hat_max,
        "p_hat_var": p_hat_var,
        "propensity_separation_flag": propensity_separation_flag,
        "propensity_auc": propensity_auc,
        "null_nan_frac": null_nan_frac,
        "beta_null_mean_k0": beta_null_mean_k0,
        "beta_null_sd_k0": beta_null_sd_k0,
        "null_z_mean_k0": null_z_mean_k0,
        "null_z_sd_k0": null_z_sd_k0,
        "null_z_skew_k0": null_z_skew_k0,
        "z_obs_k0": z_obs_k0,
        "p_raw_k0": float(pvals_raw[0]),
        "p_skew_k0": float(pvals_skew[0]),
        "seed_base": int(base_seed),
    }


def collect_gene_diagnostics(
    inputs: Any,
    genes: Iterable[str],
    B: int,
    base_seed: int,
    propensity_model: Callable = fit_propensity_logistic,
    skew_side_code: int = 0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gene in genes:
        obs_idx = union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[gene])
        guide_count = len(inputs.gene_to_cols[gene])
        out.append(
            compute_unit_diagnostics(
                inputs=inputs,
                unit_id=gene,
                unit_type="real_gene",
                obs_idx=obs_idx,
                guide_count=guide_count,
                B=B,
                base_seed=base_seed,
                propensity_model=propensity_model,
                skew_side_code=skew_side_code,
            )
        )
    return out


def collect_ntc_group_diagnostics(
    inputs: Any,
    ntc_groups: Mapping[str, Sequence[str]],
    B: int,
    base_seed: int,
    propensity_model: Callable = fit_propensity_logistic,
    skew_side_code: int = 0,
) -> List[Dict[str, Any]]:
    guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
    out: List[Dict[str, Any]] = []
    for group_id, guides in ntc_groups.items():
        cols = [guide_to_col[g] for g in guides if g in guide_to_col]
        obs_idx = union_obs_idx_from_cols(inputs.G, np.asarray(cols, dtype=np.int32))
        out.append(
            compute_unit_diagnostics(
                inputs=inputs,
                unit_id=group_id,
                unit_type="ntc_group",
                obs_idx=obs_idx,
                guide_count=len(cols),
                B=B,
                base_seed=base_seed,
                propensity_model=propensity_model,
                skew_side_code=skew_side_code,
            )
        )
    return out


def summarize_diagnostics(
    diagnostics_df: pd.DataFrame,
    B: int,
    base_seed: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "B": int(B),
        "seed_base": int(base_seed),
        "n_units": int(diagnostics_df.shape[0]),
    }

    def _quantiles(vals: np.ndarray) -> Dict[str, float]:
        return {
            "q10": float(np.quantile(vals, 0.1)),
            "median": float(np.quantile(vals, 0.5)),
            "q90": float(np.quantile(vals, 0.9)),
        }

    for unit_type, sub in diagnostics_df.groupby("unit_type"):
        block: Dict[str, Any] = {
            "n_units": int(sub.shape[0]),
            "x_mean": _quantiles(sub["x_mean"].to_numpy()),
            "n_treated": _quantiles(sub["n_treated"].to_numpy()),
            "den_is_bad_frac": float(np.mean(sub["den_is_bad"])),
            "beta_obs_nan_frac": float(np.mean(sub["beta_obs_nan"])),
            "pvals_raw_eq1_frac": float(np.mean(sub["pvals_raw_eq1_frac"])),
            "pvals_skew_eq1_frac": float(np.mean(sub["pvals_skew_eq1_frac"])),
            "p_hat_mean": _quantiles(sub["p_hat_mean"].to_numpy()),
            "propensity_separation_frac": float(
                np.mean(sub["propensity_separation_flag"])
            ),
        }
        summary[unit_type] = block

    if "real_gene" in summary:
        summary["overall_real_gene_p_raw_eq1_frac"] = summary["real_gene"][
            "pvals_raw_eq1_frac"
        ]
        summary["overall_real_gene_beta_nan_frac"] = summary["real_gene"][
            "beta_obs_nan_frac"
        ]
    if "ntc_group" in summary:
        summary["overall_ntc_p_raw_eq1_frac"] = summary["ntc_group"][
            "pvals_raw_eq1_frac"
        ]
        summary["overall_ntc_beta_nan_frac"] = summary["ntc_group"][
            "beta_obs_nan_frac"
        ]
    return summary


def write_summary(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2))
        handle.write("\n")
