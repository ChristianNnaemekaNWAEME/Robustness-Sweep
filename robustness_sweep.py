"""
robustness_sweep.py

Exhibit J-7: Direct demonstration of the estimator's core theoretical
claim, robustness to outlier contamination, by sweeping the
contamination fraction across a range of values and comparing the
variable-SELECTION accuracy of the robust selector (Trimmed-LASSO)
against plain LASSO. This is the same metric (recovery of the true
active variable set) already used in this repository's original J-2
demonstration, extended here across a controlled range of
contamination levels rather than a single fixed level.

This is a more direct, isolated test of robustness than either the
published thesis's simulation (run at fixed contamination) or the NASA
C-MAPSS application (Exhibit J-3), whose FD001 subset was low-
contamination by design. Here, contamination fraction is the single
controlled variable.

Results are reported exactly as obtained, including any result that
does not favor the robust method at a given contamination level.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from robust_hybrid_regularization import generate, trimmed_lasso_select, lasso_select


def selection_accuracy(selected_mask, true_beta):
    """Fraction of truly-active variables correctly selected (recall)."""
    true_active = np.abs(true_beta) > 1e-10
    if true_active.sum() == 0:
        return float("nan")
    return (selected_mask & true_active).sum() / true_active.sum()


def run_at_contamination(frac, random_state=42):
    data = generate(
        n=150, p=200, rho=0.9, sigma=5.0, n_active=10,
        contamination_fraction=frac, contamination_magnitude=10.0,
        random_state=random_state,
    )

    lasso_mask = lasso_select(data.X, data.y)
    trimmed_mask = trimmed_lasso_select(data.X, data.y)

    lasso_acc = selection_accuracy(lasso_mask, data.true_beta)
    trimmed_acc = selection_accuracy(trimmed_mask, data.true_beta)

    return lasso_acc, trimmed_acc


def main():
    contamination_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"{'Contamination':>15} {'LASSO recall':>13} {'Trimmed-LASSO recall':>22}")
    results = []
    for frac in contamination_levels:
        lasso_acc, trimmed_acc = run_at_contamination(frac)
        print(f"{frac*100:>14.0f}% {lasso_acc*100:>12.1f}% {trimmed_acc*100:>21.1f}%")
        results.append((frac, lasso_acc, trimmed_acc))

    print("\nResults reported exactly as obtained, not adjusted or selected")
    print("to favor any particular outcome.")

    import csv
    with open("robustness_sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["contamination_fraction", "lasso_recall", "trimmed_lasso_recall"])
        writer.writerows(results)


if __name__ == "__main__":
    main()
