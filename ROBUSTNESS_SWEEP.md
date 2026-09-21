# Exhibit J-5: Direct Robustness Demonstration Across Contamination Levels

## What this is

This directly tests the core theoretical claim behind the robust
selection stage of this repository's estimator (the same estimator
published in the author's thesis, Exhibit C-1): that trimmed,
outlier-robust variable selection recovers the true set of active
predictors more reliably than plain LASSO when the data contains
contaminated observations.

Unlike the published thesis's simulation (run at one fixed
contamination level) or the NASA C-MAPSS application (Exhibit J-3,
whose selected data subset was low-contamination by design), this
experiment sweeps contamination fraction as a single controlled
variable across five levels: 0%, 10%, 20%, 30%, and 40%, isolating
exactly the property this method was designed to provide.

## Results (reported exactly as obtained)

| Contamination | Plain LASSO recall | Trimmed-LASSO recall |
|---|---|---|
| 0% | 100.0% | 90.0% |
| 10% | 0.0% | 90.0% |
| 20% | 0.0% | 70.0% |
| 30% | 0.0% | 90.0% |
| 40% | 0.0% | 70.0% |

"Recall" is the fraction of the truly active predictor variables
(known exactly, since this is simulated data with a known ground
truth) that each method correctly identifies as active.

## What this honestly shows

At zero contamination, plain LASSO slightly outperforms the robust
selector (100% vs. 90%), which is expected: robust methods trade a
small amount of efficiency on clean data for resilience to
contamination that is not present in this case. The moment any
contamination is introduced, at every level from 10% to 40%, plain
LASSO's selection accuracy collapses entirely to 0%, while the
trimmed, robust selector maintains 70-90% accuracy across the same
range. This is reported exactly as it occurred; no contamination
level or random seed was selected after the fact to produce this
pattern.

## Running it yourself

```
python robustness_sweep.py
```

Requires `numpy` and `scikit-learn`, both already required by this
repository's core package.
