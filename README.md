# Robust Hybrid Regularization

An open-source implementation of a **two-step hybrid regularization and
variable-selection approach** for high-dimensional, multicollinear,
outlier-contaminated regression problems, based on the two-step structure
described in my published thesis:

> Nwaeme, C.N. & Lukman, A.F., "Robust Hybrid Algorithms for Regularization
> and Variable Selection in QSAR Studies," *Journal of the Nigerian Society
> of Physical Sciences.*

## What this is, and isn't

This is an **independent, from-scratch re-implementation** of the general
two-step idea described in the paper (robust variable selection, followed by
a second-stage predictive model fit only on the selected variables), written
in Python using `scikit-learn`. It is **not** a line-for-line reproduction of
the original paper's exact Sparse LTS algorithm, which uses the FAST-LTS
concentration-step procedure from the R `robustHD` package. Here, the robust
selection step (`trimmed_lasso_select`) is a simpler, original alternative:
iteratively fit an L1-penalized regression, discard the observations with the
largest residuals, and refit on the remaining "clean" subset, until the
selected variable set stabilizes. It serves the same purpose (variable
selection that isn't thrown off by outliers in the response) via a different,
simpler mechanism.

All results below are from this implementation running on independently
generated synthetic data, not a reproduction of the original paper's dataset
or results.

## What's included

- **`estimators.py`** — `trimmed_lasso_select` (the robust selector),
  `lasso_select` (a plain-LASSO baseline selector), and `HybridEstimator`,
  a scikit-learn-compatible two-step estimator combining a selector with a
  cross-validated second-stage regressor (Ridge, LASSO, Random Forest, or
  SVR).
- **`simulate.py`** — generates synthetic high-dimensional, multicollinear,
  outlier-contaminated regression data (correlated predictors, a sparse true
  coefficient vector, and a configurable fraction of response values
  contaminated with large additive outliers), following the same general
  simulation design described in the thesis.
- **`metrics.py`** — RMSE, MAD, and Median Absolute Error, the three metrics
  reported in the thesis.
- **`tests/`** — correctness tests, including a direct test of the core
  claim: that the robust selector recovers the true set of active variables
  at least as well as plain LASSO under contamination.
- **`examples/comparison_demo.py`** — compares plain LASSO, plain Ridge,
  Random Forest, a non-robust hybrid (LASSO selection + Ridge), and the
  robust hybrid (trimmed-LASSO selection + Ridge) on a contaminated,
  high-dimensional synthetic dataset.

## Results (from the included demo)

Running `examples/comparison_demo.py` on synthetic data with n=150, p=200,
90% pairwise predictor correlation decay, 10 truly active variables, and 20%
of observations contaminated with large outliers (test split, n=45):

| Method | RMSE | MAD | MedAE |
|---|---|---|---|
| Ridge (plain) | 372.16 | 293.07 | 253.49 |
| Random Forest (plain) | 305.45 | 244.18 | 220.07 |
| LASSO+Ridge (hybrid) | 304.42 | 242.79 | 203.84 |
| LASSO (plain) | 291.47 | 226.15 | 153.35 |
| **Trimmed-LASSO+Ridge (robust hybrid)** | **288.07** | **229.14** | **179.99** |

The robust hybrid achieves the lowest test RMSE of the five methods compared.

More strikingly, in the included test suite (`tests/test_estimators.py`),
averaged over five random seeds at 20% contamination, **plain LASSO recovers
0% of the true active variables**, while the **trimmed-LASSO selector
recovers 45%** — a direct demonstration of the practical cost of ignoring
outliers during variable selection in this setting.

![RMSE comparison](examples/output/rmse_comparison.png)

## Installation

```bash
git clone https://github.com/ChristianNnaemekaNWAEME/robust-hybrid-regularization.git
cd robust-hybrid-regularization
pip install -r requirements.txt
```

## Usage

```python
from robust_hybrid_regularization import HybridEstimator, generate

data = generate(n=150, p=200, contamination_fraction=0.2, random_state=0)

model = HybridEstimator(selector="trimmed_lasso", second_stage="ridge")
model.fit(data.X, data.y)

print(f"Selected {model.n_selected_} of {data.X.shape[1]} variables")
predictions = model.predict(data.X)
```

Run the full comparison demo:

```bash
python examples/comparison_demo.py
```

Run the test suite:

```bash
python tests/test_estimators.py
```

## Roadmap

- [ ] Implement a closer approximation of the original FAST-LTS
      concentration-step algorithm for direct comparison against the
      simpler trimmed-LASSO selector used here
- [ ] Add Sparse LTS as a third selector option
- [ ] Benchmark against real (not only synthetic) high-dimensional datasets
- [ ] Package as a scikit-learn-compatible estimator with `GridSearchCV`
      support out of the box

## License

MIT — see [LICENSE](LICENSE).
