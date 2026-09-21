"""
estimators.py
=============

Implements the two-step hybrid regularization approach described in:

    Nwaeme, C.N. & Lukman, A.F., "Robust Hybrid Algorithms for
    Regularization and Variable Selection in QSAR Studies",
    Journal of the Nigerian Society of Physical Sciences.

Step 1 (variable selection): either standard LASSO, or a trimmed-LASSO
robust variable selector -- a practical, iterative approximation of the
Sparse Least Trimmed Squares (Sparse LTS) idea of Alfons, Croux & Gelper
(2013): repeatedly fit an L1-penalized regression, discard the points
with the largest residuals, and refit on the remaining "clean" subset,
until the selected variable set stabilizes. This is not a reimplementation
of the exact FAST-LTS concentration-step algorithm used in the original
paper's `robustHD` R package -- it is a from-scratch, simplified robust
selector built to serve the same purpose (variable selection that is not
thrown off by outliers in the response).

Step 2 (prediction): the reduced variable set from Step 1 is used to
train a second-stage regressor -- Ridge, Lasso, Random Forest, or Support
Vector Regression -- selected by cross-validated error, matching the
two-step structure described in the published thesis.

Author: Christian Nnaemeka Nwaeme
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin


def trimmed_lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    h_fraction: float = 0.75,
    alpha: Optional[float] = None,
    max_iter: int = 30,
    tol: int = 0,
) -> np.ndarray:
    """
    Robust variable selection via iterative trimming + LASSO refitting.

    Starting from all n observations, repeatedly:
      1. Fit a LASSO (with alpha chosen by cross-validation ON THE
         CURRENTLY RETAINED SUBSET ONLY, so contaminated points excluded
         in earlier iterations do not influence the penalty choice) on
         the current retained subset.
      2. Compute residuals for ALL n observations against that fit.
      3. Retain the h = floor(h_fraction * n) observations with the
         smallest squared residuals.
      4. Stop when the retained subset stops changing (or max_iter is hit).

    Returns the boolean mask of selected (non-zero-coefficient) variables
    from the final fit on the converged clean subset.
    """
    n, p = X.shape
    h = int(np.floor(h_fraction * n))
    if h < p + 1:
        h = min(n, max(p // 4, 10))

    retained = np.arange(n)  # start with everyone
    prev_retained = None
    final_model = None

    for _ in range(max_iter):
        n_retained = len(retained)
        cv_folds = min(5, n_retained) if n_retained >= 2 else 2
        try:
            cv_model = LassoCV(cv=cv_folds, max_iter=20000).fit(X[retained], y[retained])
            fitted_alpha = cv_model.alpha_
        except ValueError:
            fitted_alpha = 1.0

        model = Lasso(alpha=fitted_alpha, max_iter=20000)
        model.fit(X[retained], y[retained])
        final_model = model

        residuals = (y - model.predict(X)) ** 2
        order = np.argsort(residuals)
        new_retained = np.sort(order[:h])

        if prev_retained is not None and np.array_equal(new_retained, prev_retained):
            retained = new_retained
            break

        prev_retained = retained = new_retained

    return np.abs(final_model.coef_) > 1e-10


def lasso_select(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standard (non-robust) LASSO variable selection via cross-validated alpha."""
    model = LassoCV(cv=5, max_iter=20000).fit(X, y)
    return np.abs(model.coef_) > 1e-10


_SECOND_STAGE_MODELS = {
    "ridge": lambda: Ridge(),
    "lasso": lambda: Lasso(max_iter=20000),
    "rf": lambda: RandomForestRegressor(n_estimators=200, random_state=0),
    "svr": lambda: SVR(),
}

_SECOND_STAGE_GRIDS = {
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "rf": {"max_depth": [None, 5, 10, 20]},
    "svr": {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.5]},
}


@dataclass
class HybridEstimator(BaseEstimator, RegressorMixin):
    """
    Two-step hybrid estimator, matching the structure described in the
    thesis: a variable-selection step ('lasso' or 'trimmed_lasso'),
    followed by a second-stage regressor ('ridge', 'lasso', 'rf', 'svr')
    fit only on the selected variables, with its hyperparameters chosen
    by k-fold cross-validation.
    """
    selector: Literal["lasso", "trimmed_lasso"] = "trimmed_lasso"
    second_stage: Literal["ridge", "lasso", "rf", "svr"] = "ridge"
    h_fraction: float = 0.75
    cv_folds: int = 5
    random_state: int = 0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.selector == "trimmed_lasso":
            self.selected_mask_ = trimmed_lasso_select(X, y, h_fraction=self.h_fraction)
        elif self.selector == "lasso":
            self.selected_mask_ = lasso_select(X, y)
        else:
            raise ValueError(f"Unknown selector: {self.selector}")

        if not self.selected_mask_.any():
            # Degenerate case: nothing selected, fall back to all variables
            self.selected_mask_ = np.ones(X.shape[1], dtype=bool)

        X_reduced = X[:, self.selected_mask_]

        best_score = np.inf
        best_params = None
        grid = _SECOND_STAGE_GRIDS[self.second_stage]
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        param_names = list(grid.keys())
        import itertools
        for combo in itertools.product(*grid.values()):
            params = dict(zip(param_names, combo))
            fold_errors = []
            for train_idx, val_idx in kf.split(X_reduced):
                model = _SECOND_STAGE_MODELS[self.second_stage]()
                model.set_params(**params)
                model.fit(X_reduced[train_idx], y[train_idx])
                pred = model.predict(X_reduced[val_idx])
                fold_errors.append(np.sqrt(np.mean((pred - y[val_idx]) ** 2)))
            mean_err = float(np.mean(fold_errors))
            if mean_err < best_score:
                best_score = mean_err
                best_params = params

        self.best_params_ = best_params
        self.model_ = _SECOND_STAGE_MODELS[self.second_stage]()
        self.model_.set_params(**best_params)
        self.model_.fit(X_reduced, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        X_reduced = X[:, self.selected_mask_]
        return self.model_.predict(X_reduced)

    @property
    def n_selected_(self) -> int:
        return int(self.selected_mask_.sum())
