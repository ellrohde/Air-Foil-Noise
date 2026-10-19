"""
Tests for `airfoil_noise.data`: column handling, data audit, run grouping and
input validation.

Course links: Week 5.1 (X is 2D, y is 1D), Week 5.6 (data audit, grouping),
project brief section 2 (handling invalid, missing or unsupported inputs).
"""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.data import (
    FEATURES,
    TARGET,
    _standardise_columns,
    audit,
    check_inputs,
    make_run_groups,
)


# --- Shapes and columns (Week 5.1) -----------------------------------------
def test_x_is_2d_and_y_is_1d(toy_runs):
    """Double brackets give a 2D table for X; single brackets a 1D Series for y."""
    X, y = toy_runs[FEATURES], toy_runs[TARGET]
    assert X.shape == (30, 5)  # 30 samples, 5 features
    assert y.shape == (30,)    # 1D: no second dimension


def test_standardise_columns_renames_by_position():
    """The six UCI columns are renamed to our short names, in order."""
    raw = pd.DataFrame(np.ones((2, 6)), columns=list("abcdef"))
    out = _standardise_columns(raw)
    assert list(out.columns) == FEATURES + [TARGET]


def test_standardise_columns_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        _standardise_columns(pd.DataFrame(np.ones((2, 5))))


# --- Data audit (Week 5.6) ---------------------------------------------------
def test_audit_clean_data(toy_runs):
    info = audit(toy_runs)
    assert info["rows"] == 30
    assert info["missing_values"] == 0
    assert info["duplicate_rows"] == 0
    assert info["test_runs"] == 6


def test_audit_finds_missing_and_duplicate_rows(toy_runs):
    """Plant one NaN and one copied row; the audit must count both."""
    bad = pd.concat([toy_runs, toy_runs.iloc[[0]]], ignore_index=True)
    bad.loc[5, "velocity"] = np.nan
    info = audit(bad)
    assert info["missing_values"] == 1
    assert info["duplicate_rows"] == 1


# --- Run grouping (Week 5.6 grouped split) -----------------------------------
def test_same_run_gets_same_id(toy_runs):
    """All five frequency bands of one run must share one run id."""
    groups = make_run_groups(toy_runs)
    assert groups.iloc[0:5].nunique() == 1


def test_different_runs_get_different_ids(toy_runs):
    groups = make_run_groups(toy_runs)
    assert groups.nunique() == 6
    assert groups.iloc[0] != groups.iloc[5]  # run 1 vs run 2


# --- Input validation (project brief section 2) ------------------------------
def test_valid_inputs_pass(toy_runs):
    in_range = check_inputs(toy_runs)
    assert in_range.all()


def test_missing_column_raises(toy_runs):
    with pytest.raises(ValueError, match="Missing"):
        check_inputs(toy_runs.drop(columns="velocity"))


def test_nan_raises(toy_runs):
    X = toy_runs.copy()
    X.loc[0, "chord"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        check_inputs(X)


def test_non_numeric_raises(toy_runs):
    X = toy_runs.copy()
    X["aoa"] = "high"
    with pytest.raises(ValueError, match="numeric"):
        check_inputs(X)


def test_negative_velocity_raises(toy_runs):
    """A negative speed is physically impossible, so it is rejected outright."""
    X = toy_runs.copy()
    X.loc[0, "velocity"] = -10.0
    with pytest.raises(ValueError):
        check_inputs(X)


def test_out_of_range_warns_and_is_flagged(toy_runs):
    """100 m/s was never tested: predict, but warn and flag it (extrapolation)."""
    X = toy_runs.copy()
    X.loc[0, "velocity"] = 100.0
    with pytest.warns(UserWarning, match="outside the tested range"):
        in_range = check_inputs(X)
    assert not in_range[0]
    assert in_range[1:].all()
