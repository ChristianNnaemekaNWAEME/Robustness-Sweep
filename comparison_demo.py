"""
comparison_demo.py
===================

Reproduces the comparative structure of the published thesis's simulation
study on independently-generated synthetic data: a high-dimensional,
multicollinear regression problem with 20% of the response values
contaminated by large outliers, comparing:

  - Plain LASSO (single-stage)
  - Plain Ridge (single-stage)
  - Random Forest (single-stage)
  - Hybrid: LASSO selection -> Ridge second stage
  - Hybrid: Trimmed-LASSO (robust) selection -> Ridge second stage

on held-out test data, using RMSE, MAD, and Median Absolute Error.

Run with:
    python examples/comparison_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor

from robust_hybrid_regularization import HybridEstimator, generate, all_metrics


def main():
    print("Generating synthetic high-dimensional, contaminated dataset...")
    data = generate(
        n=150, p=200, rho=0.9, sigma=5.0, n_active=10,
        contamination_fraction=0.2, contamination_magnitude=10.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.3, random_state=42
    )
    print(f"n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, p={X_train.shape[1]}\n")

    results = {}

    print("Fitting plain LASSO...")
    lasso = LassoCV(cv=5, max_iter=5000).fit(X_train, y_train)
    results["LASSO (plain)"] = all_metrics(y_test, lasso.predict(X_test))

    print("Fitting plain Ridge...")
    ridge = Ridge(alpha=10.0).fit(X_train, y_train)
    results["Ridge (plain)"] = all_metrics(y_test, ridge.predict(X_test))

    print("Fitting Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, y_train)
    results["Random Forest (plain)"] = all_metrics(y_test, rf.predict(X_test))

    print("Fitting Hybrid (LASSO selection + Ridge second stage)...")
    hybrid_lasso = HybridEstimator(selector="lasso", second_stage="ridge").fit(X_train, y_train)
    results["LASSO+Ridge (hybrid)"] = all_metrics(y_test, hybrid_lasso.predict(X_test))
    results["LASSO+Ridge (hybrid)"]["n_selected"] = hybrid_lasso.n_selected_

    print("Fitting Hybrid (Trimmed-LASSO robust selection + Ridge second stage)...")
    hybrid_robust = HybridEstimator(selector="trimmed_lasso", second_stage="ridge").fit(X_train, y_train)
    results["Trimmed-LASSO+Ridge (robust hybrid)"] = all_metrics(y_test, hybrid_robust.predict(X_test))
    results["Trimmed-LASSO+Ridge (robust hybrid)"]["n_selected"] = hybrid_robust.n_selected_

    print("\n" + "=" * 72)
    print(f"{'Method':<38}{'RMSE':>10}{'MAD':>10}{'MedAE':>10}")
    print("=" * 72)
    for name, m in results.items():
        print(f"{name:<38}{m['RMSE']:>10.3f}{m['MAD']:>10.3f}{m['MedAE']:>10.3f}")
    print("=" * 72)

    best = min(results.items(), key=lambda kv: kv[1]["RMSE"])
    print(f"\nLowest RMSE: {best[0]} ({best[1]['RMSE']:.3f})")

    try:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
        names = list(results.keys())
        rmses = [results[n]["RMSE"] for n in names]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(names)), rmses, color="tab:blue")
        best_idx = names.index(best[0])
        bars[best_idx].set_color("tab:green")
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.ylabel("Test RMSE")
        plt.title("Estimator Comparison on Contaminated High-Dimensional Data")
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "output", "rmse_comparison.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved comparison chart to {out_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
