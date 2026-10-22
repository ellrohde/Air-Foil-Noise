# Results summary

Generated automatically by `run_all.py`. Do not edit by hand.

## Data
- 1503 rows from 106 wind-tunnel test runs (0 missing values, 0 duplicate rows).
- Development set: 1186 rows / 84 runs. Final test set: 317 rows / 22 runs (whole runs held out).

## Inputs and target: range in the data

| Variable | Role | Unit | Min | Max | Distinct values |
|---|---|---|---|---|---|
| Frequency | input | Hz | 200 | 20,000 | 21 |
| Angle of attack | input | deg | 0 | 22.2 | 27 |
| Chord length | input | m | 0.0254 | 0.3048 | 6 |
| Free-stream velocity | input | m/s | 31.7 | 71.3 | 4 |
| Suction-side displacement thickness | input | m | 0.000400682 | 0.0584113 | 105 |
| Scaled sound pressure level | target | dB | 103.38 | 140.987 | 1456 |

## Degree sweep (5-fold cross-validation on the development runs)

| Degree | Terms | Train RMSE (dB) | Random-split RMSE (dB) | Grouped-split RMSE (dB) | Grouped R² | Fit time (ms) |
|---|---|---|---|---|---|---|
| Mean baseline (DummyRegressor) | 0 | – | 6.84 | 6.84 | -0.01 | – |
| 1 | 5 | 4.76 | 4.80 | 4.81 | 0.49 | 4.6 |
| 2 | 20 | 3.98 | 4.08 | 4.19 | 0.62 | 6.1 |
| 3 | 55 | 3.27 | 3.55 | 4.29 | 0.59 | 9.5 |
| 4 | 125 | 2.51 | 3.16 | 1551.83 | -166828.69 | 18.7 |
| 5 | 251 | 2.10 | 4.52 | 2279.00 | -302249.09 | 41.4 |

- **Selected degree (grouped split, simplest within 5% of best): 2**
- A random row split alone would have selected degree 4.

## Final test (held-out runs, used once)

| Model | RMSE (dB) | MAE (dB) | R² | Underpredicted by >3 dB |
|---|---|---|---|---|
| Mean baseline | 7.15 | 5.68 | -0.05 | 30% |
| Linear regression | 4.98 | 3.69 | 0.49 | 21% |
| Polynomial degree 2 (selected) | 4.88 | 3.51 | 0.51 | 21% |
| Polynomial degree 4 (random-split pick) | 834.53 | 314.10 | -14337.51 | 31% |

## Physical check: noise vs velocity (31.7 → 71.3 m/s)
- 144 matched pairs (same chord, angle and frequency).
- Model predicts an increase in 100% of pairs (measured: 76%).
- Mean increase: predicted +3.9 dB, measured +3.3 dB.

## Sensitivity (permutation importance on the final test runs)

| Input | RMSE increase when shuffled (dB) |
|---|---|
| frequency | 5.53 |
| thickness | 4.03 |
| aoa | 2.64 |
| chord | 2.17 |
| velocity | 0.41 |

## Numerical stability (condition number of the design matrix)

| Degree | With StandardScaler | Without scaling |
|---|---|---|
| 1 | 3.4e+00 | 5.4e+05 |
| 2 | 4.3e+01 | 9.0e+11 |
| 3 | 3.1e+03 | 7.9e+18 |
| 4 | 1.2e+17 | 1.8e+30 |
| 5 | 2.0e+18 | 1.2e+43 |
