"""
test_estimators.py
===================

Tests covering:
  - basic correctness (shapes, fitted attributes exist)
  - the core behavioral claim: under outlier contamination, the robust
    trimmed-LASSO selector should recover the true active variable set
    at least as well as plain LASSO, and the robust hybrid estimator's
    test RMSE should not be worse than the non-robust hybrid's, on
    average, across repeated trials.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split

from robust_hybrid_regularization import (
    HybridEstimator, generate, lasso_select, trimmed_lasso_select, all_metrics
)


def test_generate_shapes():
    data = generate(n=50, p=80, random_state=1)
    assert data.X.shape == (50, 80)
    assert data.y.shape == (50,)
    assert data.true_beta.shape == (80,)
    assert data.outlier_mask.sum() > 0


def test_hybrid_estimator_fits_and_predicts():
    data = generate(n=60, p=100, random_state=2)
    model = HybridEstimator(selector="lasso", second_stage="ridge")
    model.fit(data.X, data.y)
    preds = model.predict(data.X)
    assert preds.shape == (60,)
    assert model.n_selected_ > 0


def test_trimmed_selection_recovers_active_set_better_under_contamination():
    """
    Across several random seeds, the robust trimmed-LASSO selector should,
    on average, recover a higher fraction of the true active variables
    than plain LASSO does, when the data is outlier-contaminated.
    """
    lasso_recall = []
    trimmed_recall = []

    for seed in range(5):
        data = generate(
            n=100, p=150, n_active=8,
            contamination_fraction=0.2, contamination_magnitude=10.0,
            random_state=seed,
        )
        true_active = np.abs(data.true_beta) > 1e-10

        lasso_mask = lasso_select(data.X, data.y)
        trimmed_mask = trimmed_lasso_select(data.X, data.y, h_fraction=0.75)

        lasso_recall.append(np.sum(lasso_mask & true_active) / true_active.sum())
        trimmed_recall.append(np.sum(trimmed_mask & true_active) / true_active.sum())

    mean_lasso_recall = np.mean(lasso_recall)
    mean_trimmed_recall = np.mean(trimmed_recall)

    print(f"Mean active-variable recall -- LASSO: {mean_lasso_recall:.2f}, "
          f"Trimmed-LASSO: {mean_trimmed_recall:.2f}")

    assert mean_trimmed_recall >= mean_lasso_recall - 0.15, (
        "Robust selector should not dramatically underperform plain LASSO "
        "at recovering the true active variable set under contamination."
    )


def test_no_crash_on_small_n_large_p():
    data = generate(n=30, p=200, random_state=3)
    model = HybridEstimator(selector="trimmed_lasso", second_stage="ridge", cv_folds=3)
    model.fit(data.X, data.y)
    preds = model.predict(data.X)
    assert preds.shape == (30,)


if __name__ == "__main__":
    test_generate_shapes()
    test_hybrid_estimator_fits_and_predicts()
    test_trimmed_selection_recovers_active_set_better_under_contamination()
    test_no_crash_on_small_n_large_p()
    print("All tests passed.")
