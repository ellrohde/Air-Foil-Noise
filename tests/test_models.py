"""
Tests for `airfoil_noise.models`.

Many of these are *verification* tests (Week 6 and 7.5): check the code on a
problem whose answer is known exactly, or check that two independent ways of
computing the same thing agree. Verification shows the code is implemented
correctly; it does not show the model describes real airfoils (that is
validation, done by cross-validation and the final test set).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from airfoil_noise.data import FEATURES, TARGET
from airfoil_noise.models import (
    condition_number,
    estimate_fit_flops,
    make_linear,
    make_mean_baseline,
    make_polynomial,
    n_poly_terms,
    predict_spl,
)


def test_mean_baseline_predicts_training_mean(toy_runs):
    """Week 5.5 DummyRegressor: every prediction equals the training mean."""
    X, y = toy_runs[FEATURES], toy_runs[TARGET]
    pred = make_mean_baseline().fit(X, y).predict(X)
    assert np.allclose(pred, y.mean())


def test_degree_1_polynomial_equals_linear_regression(toy_runs):
    """Two routes to the same model must agree (manual vs library check)."""
    X, y = toy_runs[FEATURES], toy_runs[TARGET]
    p_lin = make_linear().fit(X, y).predict(X)
    p_poly1 = make_polynomial(1).fit(X, y).predict(X)
    assert np.allclose(p_lin, p_poly1)


def test_degree_2_recovers_a_known_quadratic():
    """Known-answer test: noise-free data from an exact quadratic.

    If the pipeline is wired correctly, degree 2 must reproduce it almost
    perfectly, while degree 1 (a flat plane) cannot.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "frequency": rng.uniform(200, 20000, 200),
        "aoa": rng.uniform(0, 20, 200),
        "chord": rng.uniform(0.03, 0.3, 200),
        "velocity": rng.uniform(32, 71, 200),
        "thickness": rng.uniform(0.001, 0.05, 200),
    })
    y = 100 + 0.002 * X["velocity"] ** 2 + 0.3 * X["aoa"] * X["chord"]
    p2 = make_polynomial(2).fit(X, y).predict(X)
    p1 = make_linear().fit(X, y).predict(X)
    assert np.max(np.abs(p2 - y)) < 1e-6        # degree 2: essentially exact
    assert np.max(np.abs(p1 - y)) > 1e-3        # degree 1: misses the curve


def test_invalid_degree_raises():
    with pytest.raises(ValueError):
        make_polynomial(0)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_term_count_matches_scikit_learn(degree):
    """Our formula C(5+d, d) - 1 must match what PolynomialFeatures builds.

    `parametrize` runs this one test five times, once per degree.
    """
    built = PolynomialFeatures(degree, include_bias=False).fit_transform(np.ones((1, 5)))
    assert n_poly_terms(5, degree) == built.shape[1]


def test_flops_grow_with_number_of_terms():
    """Week 2: more terms means more floating-point work for the same rows."""
    assert estimate_fit_flops(1000, 251) > estimate_fit_flops(1000, 20)


def test_scaling_improves_conditioning(toy_runs):
    """Week 9.1: scaling first should give a much smaller condition number."""
    scaled = condition_number(toy_runs, 2, scaled=True)
    unscaled = condition_number(toy_runs, 2, scaled=False)
    assert scaled < unscaled / 1000


def test_predict_spl_returns_predictions_and_range_flag(toy_runs):
    X, y = toy_runs[FEATURES], toy_runs[TARGET]
    model = make_polynomial(2).fit(X, y)
    out = predict_spl(model, X)
    assert list(out.columns) == ["spl_pred", "in_tested_range"]
    assert len(out) == len(X)
    assert out["in_tested_range"].all()
