"""
simulate.py
===========

Generates synthetic high-dimensional, multicollinear, outlier-contaminated
regression data, following the same general simulation structure described
in the published thesis: correlated predictors, a sparse true coefficient
vector, and a fraction of response values contaminated with large outliers.

This is an independent re-creation of that simulation design for the
purpose of this open-source demo, not a reproduction of the original
paper's exact datasets.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulatedDataset:
    X: np.ndarray
    y: np.ndarray
    true_beta: np.ndarray
    outlier_mask: np.ndarray


def generate(
    n: int = 100,
    p: int = 150,
    rho: float = 0.9,
    sigma: float = 5.0,
    n_active: int = 8,
    contamination_fraction: float = 0.2,
    contamination_magnitude: float = 10.0,
    random_state: int = 0,
) -> SimulatedDataset:
    """
    n: number of observations
    p: number of predictors (p > n by default, matching the thesis's
       high-dimensional setting)
    rho: pairwise correlation decay parameter, corr(i, j) ~ rho^|i-j|
    sigma: noise standard deviation
    n_active: number of predictors with a genuinely non-zero coefficient
    contamination_fraction: proportion of observations whose response is
       contaminated with a large additive outlier
    contamination_magnitude: multiplier controlling outlier size, following
       the thesis's contamination rule y_i <- m * max(y) + y_i
    """
    rng = np.random.default_rng(random_state)

    # Build a correlated design matrix via an AR(1)-style covariance structure.
    idx = np.arange(p)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    true_beta = np.zeros(p)
    active_idx = rng.choice(p, size=n_active, replace=False)
    true_beta[active_idx] = rng.choice([-10, -5, -3, 3, 5, 10], size=n_active, replace=True)

    noise = rng.normal(0, sigma, size=n)
    y = X @ true_beta + noise

    n_outliers = int(np.floor(contamination_fraction * n))
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    outlier_mask = np.zeros(n, dtype=bool)
    outlier_mask[outlier_idx] = True
    y[outlier_idx] = contamination_magnitude * np.max(y) + y[outlier_idx]

    return SimulatedDataset(X=X, y=y, true_beta=true_beta, outlier_mask=outlier_mask)
