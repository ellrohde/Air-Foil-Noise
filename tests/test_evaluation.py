"""
Tests for `airfoil_noise.evaluation`: metrics, splitting, model selection and
the physical trend check.

Course links: Week 5.5 (metrics, worked example), Week 5.6 ("verify that
zero fan units are shared between training and validation"), Week 5.2
(5% selection rule).
"""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.data import FEATURES, TARGET, make_run_groups
from airfoil_noise.evaluation import (
    cross_validate_model,
    grouped_splitter,
    hold_out_test_runs,
    random_splitter,
    regression_metrics,
    select_degree,
    underprediction_rate,
    velocity_trend_check,
)
from airfoil_noise.models import make_polynomial


# --- Metrics: hand-checked against the Week 5.5 worked example --------------
def test_four_equal_errors_give_mae_2_rmse_2():
    """Week 5.5: four errors of 2 -> MAE 2, RMSE 2."""
    m = regression_metrics([10, 10, 10, 10], [8, 8, 8, 8])
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(2.0)


def test_one_large_error_gives_mae_2_rmse_4():
    """Week 5.5: errors (0, 0, 0, 8) -> MAE 2 but RMSE 4.

    Same MAE, but RMSE doubles because squaring punishes the big miss.
    """
    m = regression_metrics([10, 10, 10, 10], [10, 10, 10, 2])
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(4.0)


def test_perfect_prediction_has_r2_of_1():
    y = [110.0, 120.0, 130.0]
    assert regression_metrics(y, y)["r2"] == pytest.approx(1.0)


def test_predicting_the_mean_has_r2_of_0():
    y = np.array([110.0, 120.0, 130.0])
    assert regression_metrics(y, np.full(3, y.mean()))["r2"] == pytest.approx(0.0)


def test_underprediction_only_counts_too_quiet_by_more_than_3db():
    """Residual = measured - predicted. Positive and > 3 dB is costly."""
    measured = [120, 120, 120, 120]
    predicted = [116, 118, 124, 120]  # residuals: +4, +2, -4, 0
    assert underprediction_rate(measured, predicted) == pytest.approx(0.25)


# --- Splitting (Week 5.6) ----------------------------------------------------
def test_grouped_split_never_shares_a_run(toy_runs):
    """No test run may appear in both training and validation."""
    groups = make_run_groups(toy_runs)
    for tr, va in grouped_splitter(3).split(toy_runs, groups=groups):
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])


def test_random_split_does_mix_runs(toy_runs):
    """The random split puts bands of the same run on both sides (the leak)."""
    groups = make_run_groups(toy_runs)
    shared = [
        len(set(groups.iloc[tr]) & set(groups.iloc[va]))
        for tr, va in random_splitter(3).split(toy_runs)
    ]
    assert max(shared) > 0


def test_final_test_set_shares_no_runs_and_is_repeatable(toy_runs):
    groups = make_run_groups(toy_runs)
    dev1, test1 = hold_out_test_runs(toy_runs, groups, 0.34, seed=42)
    dev2, test2 = hold_out_test_runs(toy_runs, groups, 0.34, seed=42)
    assert set(groups.iloc[dev1]).isdisjoint(groups.iloc[test1])
    assert np.array_equal(test1, test2)  # same seed -> same split


def test_cross_validate_returns_one_row_per_fold(toy_runs):
    X, y = toy_runs[FEATURES], toy_runs[TARGET]
    groups = make_run_groups(toy_runs)
    cv = cross_validate_model(make_polynomial(1), X, y, grouped_splitter(3), groups)
    assert len(cv) == 3
    assert (cv["val_rmse"] >= 0).all()  # sign flipped back to positive


# --- Model selection (Week 5.2 rule) ----------------------------------------
def test_select_degree_picks_simplest_within_5_percent():
    """Best grouped RMSE is 4.00 at degree 3; degree 2 (4.15) is within 5%."""
    sweep = pd.DataFrame({
        "degree": [1, 2, 3, 1, 2, 3],
        "split": ["grouped"] * 3 + ["random"] * 3,
        "val_rmse": [5.0, 4.15, 4.0, 3.0, 2.0, 1.0],
    })
    assert select_degree(sweep, tolerance=0.05) == 2


def test_select_degree_ignores_random_split():
    """Random-split scores must not influence the choice."""
    sweep = pd.DataFrame({
        "degree": [1, 2, 1, 2],
        "split": ["grouped", "grouped", "random", "random"],
        "val_rmse": [4.0, 5.0, 9.0, 1.0],
    })
    assert select_degree(sweep) == 1


# --- Physical check ----------------------------------------------------------
def test_velocity_check_on_known_u5_law(toy_runs):
    """toy_runs follows SPL ~ 50*log10(U), so 31.7 -> 71.3 m/s must give +17.6 dB."""
    perfect = toy_runs[TARGET].to_numpy()
    out = velocity_trend_check(toy_runs, perfect)
    assert out["pairs"] == 10  # 2 chords with both speeds x 5 frequencies
    assert out["model_increase_fraction"] == 1.0
    assert out["mean_measured_increase_db"] == pytest.approx(17.6, abs=0.1)
    assert out["mean_predicted_increase_db"] == pytest.approx(17.6, abs=0.1)
