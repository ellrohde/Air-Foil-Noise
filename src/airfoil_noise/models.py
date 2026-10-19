"""
Model definitions: the mean baseline, the linear baseline and polynomial
regression, plus cost and numerical-stability helpers.

All models are scikit-learn `Pipeline`s. A pipeline chains steps (scale ->
build polynomial terms -> fit) into one object, so `fit` and `predict` run
every step in order. Its main job here is preventing preprocessing leakage
(Week 5.3/5.6): inside cross-validation, the scaler is fitted on the training
fold only, never on the validation fold.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .data import FEATURES, check_inputs


def make_mean_baseline() -> Pipeline:
    """Predict the training-set mean SPL for every row.

    Course link: Week 5.5 `DummyRegressor` baseline. It ignores every input,
    so any model that cannot beat it has learned nothing useful.
    """
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def make_linear() -> Pipeline:
    """Linear regression baseline: SPL = b0 + b1*x1 + ... + b5*x5.

    Course link: Week 5.1 multiple linear regression. One flat "plane" through
    the data; it cannot bend or capture interactions.

    The scaler does not change linear predictions, but it makes the fitted
    coefficients comparable (dB per standard deviation of each input).
    """
    return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])


def make_polynomial(degree: int) -> Pipeline:
    """Polynomial regression of a given degree.

    Course link: Week 3 (least-squares polynomial fitting, over/underfitting)
    and Week 5.1 ("transformed features").

    What the three steps do
    -----------------------
    1. `StandardScaler`: for each input, subtract its mean and divide by its
       standard deviation, so every input is roughly between -2 and +2.
    2. `PolynomialFeatures`: build new columns from products of inputs, up
       to `degree`. For degree 2 with inputs (f, U) it makes
       f, U, f**2, f*U, U**2. The product terms (like f*U) let the model
       capture interactions, e.g. "the effect of frequency depends on speed".
    3. `LinearRegression`: ordinary least squares on those columns. The model
       is still *linear in its coefficients*, which is why plain linear
       regression can fit a curve.

    Why scale first (Week 3 and Week 9.1)
    -------------------------------------
    Week 3 noted that raw powers of x are numerically unstable. Raw frequency
    reaches 20,000 Hz while thickness is ~0.001 m, so frequency**5 and
    thickness**5 differ by ~10**36. The least-squares matrix becomes
    ill-conditioned (Week 9.1: large condition number), and tiny rounding
    errors get amplified. Scaling first keeps every column a similar size.
    `condition_number()` below measures this.

    Parameters
    ----------
    degree : int
        Highest total power of the polynomial (1 reproduces linear regression).
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            # include_bias=False: no column of 1s, because LinearRegression
            # already fits its own intercept (b0).
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )


def n_poly_terms(n_inputs: int, degree: int) -> int:
    """Number of polynomial terms (excluding the intercept).

    Equals C(n_inputs + degree, degree) - 1. With 5 inputs: degree 1 -> 5,
    2 -> 20, 3 -> 55, 4 -> 125, 5 -> 251. This rapid growth is why high
    degrees overfit (more coefficients than the data can pin down) and cost
    more to fit.
    """
    return comb(n_inputs + degree, degree) - 1


def estimate_fit_flops(n_samples: int, n_terms: int) -> float:
    """Rough floating-point operation count for one least-squares fit.

    Course link: Week 2 FLOPs. scikit-learn solves least squares with an
    SVD-based LAPACK routine whose cost is dominated by about 2*n*p**2
    operations (n rows, p terms), plus about p**3 for the small p x p part.
    Going from p = 20 (degree 2) to p = 251 (degree 5) multiplies p**2 by
    ~160, which is why fit time climbs with degree. This is an
    order-of-magnitude estimate, not an exact count.
    """
    n, p = float(n_samples), float(n_terms)
    return 2.0 * n * p**2 + p**3


def condition_number(X: pd.DataFrame, degree: int, scaled: bool = True) -> float:
    """Condition number of the polynomial design matrix.

    Course link: Week 9.1 ill-conditioned matrices. The condition number
    (kappa) measures how much small input errors can be amplified in the
    least-squares solution. Roughly, you lose log10(kappa) of the ~16 digits
    a double-precision number holds. Comparing scaled=True vs scaled=False
    shows why the StandardScaler step matters for numerical stability.
    """
    Z = StandardScaler().fit_transform(X[FEATURES]) if scaled else X[FEATURES].to_numpy()
    P = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(Z)
    return float(np.linalg.cond(P))


def predict_spl(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Predict SPL for new conditions, with input checks.

    Parameters
    ----------
    model : Pipeline
        A fitted pipeline from this module.
    X : pandas.DataFrame
        New conditions with the columns in `airfoil_noise.data.FEATURES`.

    Returns
    -------
    pandas.DataFrame
        Columns `spl_pred` (dB) and `in_tested_range` (bool). Rows flagged
        False are extrapolations and should not be used for decisions.
    """
    in_range = check_inputs(X)  # raises on bad input, warns on extrapolation
    pred = model.predict(X[FEATURES])
    return pd.DataFrame(
        {"spl_pred": np.asarray(pred, dtype=float), "in_tested_range": in_range},
        index=X.index,
    )
