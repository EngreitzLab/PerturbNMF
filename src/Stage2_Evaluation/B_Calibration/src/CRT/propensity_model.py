"""
Propensity score models for guide-union indicators.
"""

from typing import Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression


def propensity_model(
    Covariates: np.ndarray,
    Cell_treatment_indicator: np.ndarray,
    has_intercept: bool = True,
    penalty: Optional[str] = None,
    C: float = 1.0,
    max_iter: Optional[int] = None,
    tol: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[LogisticRegression]]:
    """
    Fit a logistic regression P(treated = 1 | Covariates); unpenalized by default
    (sceptre: stats::glm.fit(family = binomial)).

    The propensity model always has exactly one intercept, which carries the
    target's baseline rate. If Covariates already holds the intercept column
    (has_intercept=True, CRT_obj.has_intercept), sklearn's own intercept is off;
    otherwise sklearn adds one. Skipping the intercept in the outcome regression
    therefore does not remove it here.

    No penalty and a tight tolerance give the exact MLE, whose probabilities sum
    to the number of treated cells; a penalty or the default tol=1e-4 inflate
    sum(p) and make null treated sets too large.

    penalty="l2" (old PerturbNMF: L2, C=1) shrinks the covariate coefficients
    only: the intercept column is dropped and sklearn fits its own, unpenalized
    intercept, so its score equation still gives sum(p) = number treated.

    penalty="old_perturbnmf" reproduces PerturbNMF's fit_propensity_logistic for
    comparison only: L2 with C (old default 1) on the full design including its
    intercept column, sklearn's default tol=1e-4, max_iter=200, p clipped to
    [1e-6, 1 - 1e-6], and no convergence check. lbfgs stops after ~11 iterations
    on a high-MOI screen, with sum(p) ~1.15x the number treated (anti-conservative null).
    Passing tol / max_iter overrides the old settings (e.g. tol=1e-10 converges
    it: sum(p) = number treated with or without the L2).

    An explicit tol (any mode) is a tolerance test: the sum(p) convergence check
    is skipped, since a loose tol is expected to fail it.

    Args:
        Covariates: design matrix (N x p).
        Cell_treatment_indicator: 0/1 treated indicator for one gene (length N).
        has_intercept: whether column 0 of Covariates is the intercept.
        penalty: None (MLE), "l2" (ridge on the covariate coefficients) or
            "old_perturbnmf" (old PerturbNMF fit, see above).
        C: inverse L2 strength (sklearn C; smaller = stronger shrinkage); penalty="l2" only.
        max_iter: maximum lbfgs iterations; None = 10,000 (old_perturbnmf: 200).
        tol: lbfgs convergence tolerance; None = 1e-10 with the sum(p) check
            (old_perturbnmf: sklearn's 1e-4, no check). An explicit value skips the check.
        kwargs: extra arguments for sklearn LogisticRegression.

    Returns:
        p: propensity scores (length N), unclipped (exact 0 or 1 is fine for
            Bernoulli resampling, as in sceptre's crt_index_sampler_fast).
        model: fitted LogisticRegression model, or None when the model is
            intercept-only (p is then the constant treated fraction).

    Raises:
        RuntimeError: if sum(p) does not match the number of treated cells
            (fit not converged).
    """
    y = np.asarray(Cell_treatment_indicator, dtype=np.int8)
    if penalty not in (None, "l2", "old_perturbnmf"):
        raise ValueError(
            f'penalty must be None, "l2" or "old_perturbnmf"; got {penalty!r}.'
        )

    if penalty == "old_perturbnmf":
        # PerturbNMF B_Calibration/src/CRT/propensity.py fit_propensity_logistic (its settings unless tol / max_iter given)
        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=200 if max_iter is None else max_iter,
            tol=1e-4 if tol is None else tol,
            **kwargs,
        )
        model.fit(Covariates, y)
        p = np.clip(model.predict_proba(Covariates)[:, 1], 1e-6, 1 - 1e-6)
        return p, model

    check_sum = tol is None  # explicit tol = tolerance test, no convergence check
    tol = 1e-10 if tol is None else tol
    max_iter = 10_000 if max_iter is None else max_iter

    # no covariate columns other than the intercept
    n_covariate_cols = Covariates.shape[1] - int(has_intercept)
    if n_covariate_cols == 0:
        # intercept-only model: the MLE is the treated fraction for every cell
        p = np.full(y.shape[0], y.mean(), dtype=np.float64)
        return p, None

    if penalty == "l2":
        # penalize covariates only: sklearn's own intercept is not penalized
        X = Covariates[:, 1:] if has_intercept else Covariates
        model = LogisticRegression(
            penalty="l2",
            C=C,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=max_iter,
            tol=tol,
            **kwargs,
        )
    else:
        X = Covariates
        model = LogisticRegression(
            penalty=None,
            fit_intercept=not has_intercept,
            solver="lbfgs",
            max_iter=max_iter,
            tol=tol,
            **kwargs,
        )
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    if check_sum and not np.isclose(p.sum(), y.sum(), rtol=1e-3):
        raise RuntimeError(
            f"Propensity fit did not converge: sum(p) = {p.sum():.3f}, "
            f"n treated = {int(y.sum())}."
        )
    return p, model
