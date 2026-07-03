"""
Skew-normal moment-matching utilities used for null calibration.
"""

import warnings

import numba as nb
import numpy as np
from scipy.stats import skewnorm


@nb.njit
def _fit_skew_normal_numba_impl(y: np.ndarray) -> np.ndarray:
    """
    Numba implementation of skew-normal moment-matching.
    """
    n = y.size
    n_doub = float(n)

    s_1 = 0.0
    s_2 = 0.0
    for i in range(n):
        s_1 += y[i]
        s_2 += y[i] * y[i]
    m_y = s_1 / n_doub
    sd_y = np.sqrt(s_2 / n_doub - m_y * m_y)

    s_3 = 0.0
    for i in range(n):
        diff = y[i] - m_y
        s_3 += diff * diff * diff
    gamma1 = s_3 / (n_doub * (sd_y * sd_y * sd_y))

    max_gamma_1 = 0.995
    if np.abs(gamma1) > max_gamma_1:
        gamma1 = np.copysign(0.9 * max_gamma_1, gamma1)

    b = np.sqrt(2.0 / np.pi)
    r = np.copysign(1.0, gamma1) * np.power(
        2.0 * np.abs(gamma1) / (4.0 - np.pi), 1.0 / 3.0
    )
    delta = r / (b * np.sqrt(1.0 + r * r))
    alpha = delta / np.sqrt(1.0 - delta * delta)
    mu_z = b * delta
    sd_z = np.sqrt(1.0 - mu_z * mu_z)
    omega = sd_y / sd_z
    xi = m_y - omega * mu_z

    out = np.empty(5, dtype=np.float64)
    out[0] = xi
    out[1] = omega
    out[2] = alpha
    out[3] = m_y
    out[4] = sd_y
    return out


def fit_skew_normal(y: np.ndarray) -> np.ndarray:
    """
    Public wrapper for the numba implementation.
    """
    return _fit_skew_normal_numba_impl(np.asarray(y, dtype=np.float64))


def fit_skew_normal_funct_numba(y: np.ndarray) -> np.ndarray:
    """
    Deprecated: use fit_skew_normal instead.
    """
    warnings.warn(
        "fit_skew_normal_funct_numba is deprecated; use fit_skew_normal instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_skew_normal(y)


def compute_empirical_p_value(
    null_statistics: np.ndarray, z_orig: float, side: int
) -> float:
    """
    Empirical p-value with +1 smoothing (side: -1 left, 0 two-sided, 1 right).
    """
    null_statistics = np.asarray(null_statistics, dtype=np.float64)
    B = float(null_statistics.size)
    if side in (-1, 1):
        if side == -1:
            counter = np.sum(z_orig >= null_statistics)
        else:
            counter = np.sum(z_orig <= null_statistics)
        return (1.0 + counter) / (1.0 + B)
    return 2.0 * min(
        compute_empirical_p_value(null_statistics, z_orig, -1),
        compute_empirical_p_value(null_statistics, z_orig, 1),
    )


def check_sn_tail(
    y_sorted: np.ndarray,
    xi_hat: float,
    omega_hat: float,
    alpha_hat: float,
    ratio_thresh: float = 2.0,
) -> bool:
    """
    Validate skew-normal fit in the right tail using quantile ratios.
    """
    y_sorted = np.asarray(y_sorted, dtype=np.float64)
    n = y_sorted.size
    if n == 0:
        return False

    dist = skewnorm(alpha_hat, loc=xi_hat, scale=omega_hat)
    for i in range(180, 199):
        p = i / 200.0
        idx = int(np.ceil(n * p))
        if idx >= n:
            idx = n - 1
        quantile = y_sorted[idx]
        sn_tail_prob = dist.sf(quantile)
        if sn_tail_prob <= 0.0:
            return False
        ratio = (1.0 - p) / sn_tail_prob
        if ratio > ratio_thresh:
            return False
    return True


def check_for_outliers(
    null_sorted: np.ndarray,
    mu: float,
    sd: float,
    ratio_thresh: float = 1.5,
) -> bool:
    """
    Detect extreme tail outliers using max/min ratios.
    """
    null_sorted = np.asarray(null_sorted, dtype=np.float64)
    if null_sorted.size == 0:
        return False

    min_z = null_sorted[0]
    max_z = null_sorted[-1]
    B = float(null_sorted.size)
    denom = sd * np.sqrt(2.0 * np.log(B))
    R_max = max_z / (mu + denom)
    R_min = min_z / (mu - denom)
    return (R_max <= ratio_thresh) and (R_min <= ratio_thresh)


def fit_and_evaluate_skew_normal(
    z_orig: float,
    null_statistics: np.ndarray,
    side_code: int,
) -> np.ndarray:
    """
    Fit skew-normal to nulls, check fit, and return [xi, omega, alpha, p].
    """
    p_val = -1.0
    fitted_params = fit_skew_normal(np.asarray(null_statistics, dtype=np.float64))
    if np.all(np.isfinite(fitted_params)):
        null_sorted = np.asarray(null_statistics, dtype=np.float64).copy()
        null_sorted.sort()

        median_idx = (null_sorted.size - 1) // 2
        median = null_sorted[median_idx]
        check_right_tail = z_orig >= median

        outlier_ok = check_for_outliers(null_sorted, fitted_params[3], fitted_params[4])
        fit_ok = False
        if outlier_ok:
            if check_right_tail:
                fit_ok = check_sn_tail(
                    null_sorted, fitted_params[0], fitted_params[1], fitted_params[2]
                )
            else:
                null_left = -null_sorted[::-1]
                fit_ok = check_sn_tail(
                    null_left, -fitted_params[0], fitted_params[1], -fitted_params[2]
                )

        use_sn = outlier_ok and fit_ok
        if use_sn:
            dist = skewnorm(
                fitted_params[2], loc=fitted_params[0], scale=fitted_params[1]
            )
            if side_code == 0:
                p_tail = dist.sf(z_orig) if check_right_tail else dist.cdf(z_orig)
                p_val = 2.0 * p_tail
            elif side_code == 1:
                p_val = dist.sf(z_orig)
            else:
                p_val = dist.cdf(z_orig)

            if p_val <= 1.0e-250:
                p_val = 1.0e-250

    return np.array(
        [fitted_params[0], fitted_params[1], fitted_params[2], p_val],
        dtype=np.float64,
    )
