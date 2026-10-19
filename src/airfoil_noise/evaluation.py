"""
Metrics, cross-validation, degree sweep, timing and the physical trend check.

Which split answers which question (Week 5.6)
---------------------------------------------
- **Random row split** (`KFold(shuffle=True)`): "how well does the model do on
  more frequency bands from runs it has already seen?" Rows from one run land
  on both sides, so this is the easy test.
- **Grouped split by test run** (`GroupKFold`): "how well does it do on a
  condition nobody has tested?" Every run is kept whole. This matches the
  deployment population, so model selection uses it.
- **Time-ordered split**: not used. The NASA runs have no timestamps, and the
  task is not forecasting future operation; each run is an independent
  wind-tunnel condition.

The difference between the random and grouped scores shows how optimistic the
random split is.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline

from .data import FEATURES
from .models import estimate_fit_flops, n_poly_terms

UNDERPREDICTION_LIMIT_DB = 3.0
"""An underprediction larger than 3 dB counts as costly: +3 dB is a doubling
of acoustic energy, so a design could exceed a noise limit it appeared to
meet (the "decision and cost" part of the prediction contract)."""


def underprediction_rate(y_true, y_pred) -> float:
    """Fraction of rows where noise is underpredicted by more than 3 dB.

    Week 5.5 notes that under- and over-prediction can have different costs.
    For noise screening, underprediction is the costly direction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Positive residual (y - y_hat > 0) means the model guessed too quiet.
    return float(np.mean((y_true - y_pred) > UNDERPREDICTION_LIMIT_DB))


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, R2 and the >3 dB underprediction rate for one set of predictions.

    Course link: Week 5.5 regression metrics. With residual
    e = y - y_hat (measured minus predicted):

    | Metric | Formula                              | Meaning                          |
    |--------|--------------------------------------|----------------------------------|
    | MAE    | mean(abs(e))                         | typical error, in dB             |
    | RMSE   | sqrt(mean(e**2))                     | like MAE but punishes big misses more, in dB |
    | R2     | 1 - sum(e**2) / sum((y - mean(y))**2) | 1 = perfect, 0 = no better than the mean, negative = worse than the mean |
    | under3 | fraction of rows with e > 3 dB       | costly underpredictions          |

    Returns
    -------
    dict
        Keys `rmse`, `mae`, `r2`, `under3`.
    """
    return {
        # Week 5.5 table: "square root of MSE" gives RMSE in target units.
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "under3": underprediction_rate(y_true, y_pred),
    }


# The scorers handed to cross_validate. scikit-learn always *maximises*
# scores, so error metrics come back negative ("neg_..."); they are flipped
# back to positive in cross_validate_model() below.
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
    "under3": make_scorer(underprediction_rate, greater_is_better=False),
}


# ---------------------------------------------------------------------------
# Course link: Week 5.6 splitting strategies
# ---------------------------------------------------------------------------
def random_splitter(n_splits: int = 5, seed: int = 42) -> KFold:
    """Random row split: shuffles rows and ignores which run they came from.

    `shuffle=True` matters: the UCI file is stored run by run, so without
    shuffling KFold would accidentally cut whole runs out, i.e. a grouped
    split in disguise.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def grouped_splitter(n_splits: int = 5) -> GroupKFold:
    """Grouped split: whole test runs are held out together ("unseen pumps")."""
    return GroupKFold(n_splits=n_splits)


def hold_out_test_runs(
    df: pd.DataFrame, groups: pd.Series, test_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Set aside ~20% of test runs as a final test set, used once at the end.

    Course link: Week 5.5 "training, validation and final-test roles": if you
    keep checking the test set and adjusting, it becomes another validation
    set and its score is optimistic. `GroupShuffleSplit` picks whole runs at
    random, so the test set is also a grouped split.

    Returns
    -------
    (dev_idx, test_idx) : tuple of numpy arrays
        Row positions for model development and for the final test.
        No run appears in both.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    # split() yields (train, test) index pairs; next() takes the only one.
    dev_idx, test_idx = next(gss.split(df, groups=groups))
    return dev_idx, test_idx


def cross_validate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Score one model with k-fold cross-validation.

    Course link: Week 5.5 `cross_validate` ("evaluate multiple metrics across
    folds"). For each fold it clones the model (a fresh, unfitted copy),
    fits it on the training part, then scores it on the held-out part.

    `return_train_score=True` also scores each fold on its own training
    data. The gap between training and validation error is the overfitting
    signal from Week 3.

    Parameters
    ----------
    model : Pipeline
        An unfitted pipeline, e.g. `make_polynomial(3)`.
    X, y : DataFrame, Series
        Development data only (never the final test set).
    splitter : KFold or GroupKFold
    groups : Series, optional
        Run ids. Passed only to `GroupKFold`; `KFold` ignores groups.

    Returns
    -------
    pandas.DataFrame
        One row per fold: training and validation metrics, and fit time (s).
    """
    split_groups = groups if isinstance(splitter, GroupKFold) else None
    cv = sk_cross_validate(
        model, X, y, groups=split_groups, cv=splitter,
        scoring=SCORING, return_train_score=True,
    )
    # cv is a dict like {"test_rmse": [-4.1, -4.3, ...], "fit_time": [...], ...}.
    # Flip the sign of the error metrics back to positive values.
    out = pd.DataFrame({
        "train_rmse": -cv["train_rmse"],
        "val_rmse": -cv["test_rmse"],
        "val_mae": -cv["test_mae"],
        "val_r2": cv["test_r2"],
        "val_under3": -cv["test_under3"],
        "fit_s": cv["fit_time"],
    })
    out.index.name = "fold"
    return out


def time_fit(model_factory: Callable[[], Pipeline], X, y, repeats: int = 7) -> dict:
    """Median wall-clock time to fit and to predict, over several repeats.

    Course link: Week 2 performance analysis. `time.perf_counter()` is a
    high-resolution timer. The median is used because single timings jump
    around with whatever else the computer is doing.
    """
    fit_times, pred_times = [], []
    for _ in range(repeats):
        model = model_factory()
        t0 = time.perf_counter()
        model.fit(X, y)
        t1 = time.perf_counter()
        model.predict(X)
        t2 = time.perf_counter()
        fit_times.append(t1 - t0)
        pred_times.append(t2 - t1)
    return {
        "fit_ms": 1000 * float(np.median(fit_times)),
        "predict_ms": 1000 * float(np.median(pred_times)),
    }


def degree_sweep(
    model_factory: Callable[[int], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    degrees=range(1, 6),
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate each polynomial degree with both splitting strategies.

    Course link: Week 3 "training vs validation error" plot, where training
    error keeps falling with degree while validation error eventually
    explodes. This records, per degree:
    accuracy (mean validation RMSE/R2 and spread across folds),
    overfitting (validation minus training RMSE), and
    cost (number of terms, estimated FLOPs, matrix memory, fit time).

    Returns
    -------
    pandas.DataFrame
        One row per (degree, split) combination.
    """
    rows = []
    for d in degrees:
        n_terms = n_poly_terms(len(FEATURES), d)
        # "lambda: model_factory(d)" is a tiny function that builds a fresh
        # model each time it is called, so each timing repeat starts unfitted.
        timing = time_fit(lambda: model_factory(d), X, y)
        for split_name, splitter in [
            ("random", random_splitter(n_splits, seed)),
            ("grouped", grouped_splitter(n_splits)),
        ]:
            cv = cross_validate_model(model_factory(d), X, y, splitter, groups)
            rows.append({
                "degree": d,
                "split": split_name,
                "n_terms": n_terms,
                "train_rmse": cv["train_rmse"].mean(),
                "val_rmse": cv["val_rmse"].mean(),
                "val_rmse_std": cv["val_rmse"].std(),
                "val_r2": cv["val_r2"].mean(),
                "val_mae": cv["val_mae"].mean(),
                "val_under3": cv["val_under3"].mean(),
                "overfit_gap": cv["val_rmse"].mean() - cv["train_rmse"].mean(),
                "fit_ms": timing["fit_ms"],
                "predict_ms": timing["predict_ms"],
                "est_mflops": estimate_fit_flops(len(X), n_terms) / 1e6,
                # 8 bytes per float64 number in an n x p matrix.
                "design_matrix_kb": len(X) * n_terms * 8 / 1024,
            })
    return pd.DataFrame(rows)


def select_degree(sweep: pd.DataFrame, tolerance: float = 0.05) -> int:
    """Pick the simplest degree whose grouped validation RMSE is close to the best.

    Course link: Week 5.2 workbook rule ("accept trees within 5% of the best
    validation MAE, then select the shallowest"), applied to polynomial
    degree with RMSE. Week 3: the best model is not the one with the lowest
    training error; it balances accuracy, simplicity and plausibility.

    Only the grouped split is used, because it matches deployment.
    """
    g = sweep[sweep["split"] == "grouped"]
    best = g["val_rmse"].min()
    ok = g[g["val_rmse"] <= (1 + tolerance) * best]
    return int(ok["degree"].min())


def out_of_fold_predictions(
    model_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """Predict every row using a model that never saw that row's test run.

    Each row gets its prediction from the one fold where its run was held
    out, so the physical check below is an honest test, not a re-read of
    training data.
    """
    pred = np.empty(len(X))
    for tr, va in grouped_splitter(n_splits).split(X, y, groups):
        model = model_factory().fit(X.iloc[tr], y.iloc[tr])
        pred[va] = model.predict(X.iloc[va])
    return pred


def velocity_trend_check(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Physical check: does predicted noise rise with flow speed like the data?

    Course link: Weeks 6, 7 and 11 "engineering checks" (do physical
    constraints and trends hold?) and Week 5.1 "physical meaning". A model
    can score well and still behave unphysically.

    Physics: trailing-edge noise intensity scales roughly with U**5, so
    overall SPL rises by about 50*log10(U2/U1) dB, ~17.6 dB from 31.7 to
    71.3 m/s. That figure is reported for context only and is NOT used as a
    pass mark: the UCI target is a *scaled* SPL, and in the data the
    measured per-band increase is only ~3-5 dB. The check therefore asks
    (1) does the model predict an increase, and (2) is the size of the
    increase similar to what was measured?

    Method: find matched pairs of rows with the same chord, angle and
    frequency, measured at the lowest (31.7 m/s) and highest (71.3 m/s)
    velocity. Compare the measured SPL increase with the predicted increase,
    using out-of-fold predictions.

    Returns
    -------
    dict
        Number of pairs, fraction where the model predicts an increase,
        mean measured and predicted increase (dB), and the U**5 reference.
    """
    d = df.assign(pred=pred)
    lo, hi = d["velocity"].min(), d["velocity"].max()
    keys = ["chord", "aoa", "frequency"]
    # Index both tables by (chord, angle, frequency), then join: rows that
    # match on all three become one "pair" with _lo and _hi columns.
    a = d[d["velocity"] == lo].set_index(keys)
    b = d[d["velocity"] == hi].set_index(keys)
    pairs = a.join(b, lsuffix="_lo", rsuffix="_hi", how="inner")
    meas = pairs["spl_hi"] - pairs["spl_lo"]
    model = pairs["pred_hi"] - pairs["pred_lo"]
    return {
        "pairs": int(len(pairs)),
        "model_increase_fraction": float((model > 0).mean()),
        "measured_increase_fraction": float((meas > 0).mean()),
        "mean_measured_increase_db": float(meas.mean()),
        "mean_predicted_increase_db": float(model.mean()),
        "u5_law_increase_db": float(50 * np.log10(hi / lo)),
    }
