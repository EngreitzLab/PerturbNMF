"""
Skew-normal tail p-values for the CRT null (sceptre's resampling approximation).
Its a port of ceptre's fit_and_evaluate_skew_normal. For one program:

1. Standardize the null betas: z = (beta - mean) / sd.
2. Fit a skew-normal (xi, omega, alpha) to z by the method of moments.
3. Accept the fit only if the null has no extreme outliers and the fitted tail
   matches the empirical tail at the 90.0-99.5% quantiles within a factor of 2,
   on the side where the observed z lies.
4. Two-sided p = 2 x fitted tail beyond z_obs on that side (capped at 1,
   floored at 1e-250).

A rejected fit gives NaN, so the caller can fall back to the empirical p.
"""

from typing import Sequence, Tuple

import numba as nb
import numpy as np
from scipy.stats import skewnorm

MAX_SKEWNESS = 0.995  # |skewness| a skew-normal can reach is < 0.9953
P_FLOOR = 1e-250


@nb.njit(cache=True)
def fit_skew_normal(y):
    """
    Method-of-moments skew-normal fit.

    Args:
        y: 1-D float64 sample.

    Returns:
        [xi, omega, alpha, mean, sd]: skew-normal location, scale and shape, plus
        the sample mean and sd. |skewness| above MAX_SKEWNESS is set to
        0.9 * MAX_SKEWNESS (keeping its sign), as in sceptre.
    """
    n = y.size
    mean = 0.0
    for i in range(n):
        mean += y[i]
    mean /= n
    var = 0.0
    m3 = 0.0
    for i in range(n):
        d = y[i] - mean
        var += d * d
        m3 += d * d * d
    var /= n
    sd = np.sqrt(var)
    gamma1 = (m3 / n) / (sd * sd * sd)
    if np.abs(gamma1) > MAX_SKEWNESS:
        gamma1 = np.copysign(0.9 * MAX_SKEWNESS, gamma1)

    b = np.sqrt(2.0 / np.pi)
    r = np.copysign(1.0, gamma1) * np.power(
        2.0 * np.abs(gamma1) / (4.0 - np.pi), 1.0 / 3.0
    )
    delta = r / (b * np.sqrt(1.0 + r * r))
    alpha = delta / np.sqrt(1.0 - delta * delta)
    mu_z = b * delta
    omega = sd / np.sqrt(1.0 - mu_z * mu_z)
    xi = mean - omega * mu_z

    out = np.empty(5)
    out[0] = xi
    out[1] = omega
    out[2] = alpha
    out[3] = mean
    out[4] = sd
    return out


def _outliers_ok(
    z_sorted: np.ndarray, mean: float, sd: float, ratio_thresh: float = 1.5
) -> bool:
    """False if the null's max or min is far beyond mean +/- sd * sqrt(2 log B) (sceptre's check)."""
    spread = sd * np.sqrt(2.0 * np.log(z_sorted.size))
    return (z_sorted[-1] / (mean + spread) <= ratio_thresh) and (
        z_sorted[0] / (mean - spread) <= ratio_thresh
    )


def _tail_ok(
    z_sorted: np.ndarray,
    xi: float,
    omega: float,
    alpha: float,
    ratio_thresh: float = 2.0,
) -> bool:
    """
    False if, at any of the 90.0, 90.5, ..., 99.0% quantiles of the null, the
    empirical right-tail mass exceeds the fitted one by more than ratio_thresh.
    """
    n = z_sorted.size
    probs = np.arange(180, 199) / 200.0
    idx = np.minimum(np.ceil(n * probs).astype(np.int64), n - 1)
    fitted_tail = skewnorm.sf(z_sorted[idx], alpha, loc=xi, scale=omega)
    if np.any(fitted_tail <= 0.0):
        return False
    return bool(np.all((1.0 - probs) / fitted_tail <= ratio_thresh))


def skew_normal_pvals(
    beta_obs: np.ndarray,
    beta_null: np.ndarray,
    programs: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-sided skew-normal p-values for the given programs.

    Args:
        beta_obs: length-K observed betas.
        beta_null: n_rounds x K null betas; NaN rounds (empty or collinear
            draws) are ignored.
        programs: column indexes to evaluate.

    Returns:
        pvals_sn: length K; NaN for programs not evaluated or whose fit was
            rejected (too few rounds, sd = 0, outliers, or tail mismatch).
        skew_params: K x 3 (xi, omega, alpha) on the standardized scale z;
            NaN where pvals_sn is NaN.
    """
    n_programs = beta_obs.shape[0]
    pvals_sn = np.full(n_programs, np.nan)
    skew_params = np.full((n_programs, 3), np.nan)

    for k in programs:
        null = beta_null[:, k]
        null = null[np.isfinite(null)]
        if null.size < 10 or not np.isfinite(beta_obs[k]):
            continue
        mean, sd = null.mean(), null.std()
        if not np.isfinite(sd) or sd <= 0.0:
            continue
        z_sorted = np.sort((null - mean) / sd)
        z_obs = (beta_obs[k] - mean) / sd

        xi, omega, alpha, z_mean, z_sd = fit_skew_normal(z_sorted)
        if not np.all(np.isfinite([xi, omega, alpha])) or not _outliers_ok(
            z_sorted, z_mean, z_sd
        ):
            continue

        right_tail = z_obs >= z_sorted[(z_sorted.size - 1) // 2]
        if right_tail:
            ok = _tail_ok(z_sorted, xi, omega, alpha)
            tail = skewnorm.sf(z_obs, alpha, loc=xi, scale=omega)
        else:  # mirror so the left tail is checked like a right tail
            ok = _tail_ok(-z_sorted[::-1], -xi, omega, -alpha)
            tail = skewnorm.cdf(z_obs, alpha, loc=xi, scale=omega)
        if not ok or not np.isfinite(tail):
            continue

        pvals_sn[k] = min(1.0, max(2.0 * tail, P_FLOOR))
        skew_params[k] = (xi, omega, alpha)
    return pvals_sn, skew_params
