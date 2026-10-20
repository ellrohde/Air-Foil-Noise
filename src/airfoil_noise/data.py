"""
Data loading, auditing, run grouping and input validation.

Column names are shortened and carry their units so that code stays readable:

| Column        | Meaning                                   | Unit |
|---------------|-------------------------------------------|------|
| `frequency`   | one-third-octave band centre frequency    | Hz   |
| `aoa`         | angle of attack                           | deg  |
| `chord`       | chord length                              | m    |
| `velocity`    | free-stream velocity                      | m/s  |
| `thickness`   | suction-side displacement thickness       | m    |
| `spl`         | scaled sound pressure level (the target)  | dB   |
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Course link: Week 5.1 (features and target) and Week 5.6 (prediction contract)
#
# FEATURES are the inputs (X) and TARGET is the output (y). Week 5.6 says
# features must be "available at prediction time": all five here are set or
# known before the noise is measured, so there is no target leakage.
# ---------------------------------------------------------------------------
FEATURES = ["frequency", "aoa", "chord", "velocity", "thickness"]
"""Input columns, in the order used by every model."""

TARGET = "spl"
"""Target column: scaled sound pressure level (dB)."""

UNITS = {
    "frequency": "Hz",
    "aoa": "deg",
    "chord": "m",
    "velocity": "m/s",
    "thickness": "m",
    "spl": "dB",
}
"""Units for each column (used in plot labels and the README)."""

# ---------------------------------------------------------------------------
# Course link: Week 3 (extrapolation is risky) and Week 5.2 (tree flattens
# outside the measured domain)
#
# This box is the "measured domain". A polynomial will happily return a number
# for 100 m/s, but nothing in the data supports it, so check_inputs() warns
# whenever a prediction falls outside these limits.
# ---------------------------------------------------------------------------
TESTED_RANGE = {
    "frequency": (200.0, 20000.0),
    "aoa": (0.0, 22.2),
    "chord": (0.0254, 0.3048),
    "velocity": (31.7, 71.3),
    "thickness": (0.0004, 0.0585),
}
"""Min and max of each input in the NASA data (thickness rounded outward to
avoid float-rounding false alarms). Outside this box the model is
extrapolating and its predictions are not supported."""

UCI_ID = 291
"""UCI Machine Learning Repository id of the Airfoil Self-Noise dataset."""

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "airfoil_self_noise.csv"
"""Local copy written after the first download, so later runs work offline."""


def load_airfoil(cache_path: str | Path | None = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the airfoil self-noise data as one tidy DataFrame.

    The first call downloads the data with `ucimlrepo` and saves a CSV copy at
    `cache_path`. Later calls read the CSV, so the workflow is repeatable even
    if the UCI server is unavailable.

    Parameters
    ----------
    cache_path : str, Path or None
        Where to read/write the local CSV copy. `None` always downloads and
        never writes a file.

    Returns
    -------
    pandas.DataFrame
        1,503 rows with columns `FEATURES + [TARGET]`.

    Raises
    ------
    ConnectionError
        If there is no cached copy and the download fails.
    """
    if cache_path is not None and Path(cache_path).exists():
        df = pd.read_csv(cache_path)
        return _standardise_columns(df)

    # Imported here (not at the top) so the tests can run without internet.
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=UCI_ID)
    # ucimlrepo gives the inputs and the target as two separate tables;
    # concat(axis=1) glues them side by side into one table.
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    df = _standardise_columns(df)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


UCI_NAMES = {
    "frequency": "frequency",
    "attack-angle": "aoa",
    "chord-length": "chord",
    "free-stream-velocity": "velocity",
    "suction-side-displacement-thickness": "thickness",
    "scaled-sound-pressure": "spl",
}
"""UCI column name -> short name used in this project."""


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the six UCI columns to the short names in `FEATURES + [TARGET]`.

    If the columns carry the UCI names they are renamed by name (safest).
    If they already carry the short names they are kept. Otherwise they are
    renamed by position, in the UCI order (frequency, angle, chord,
    velocity, thickness, SPL).
    """
    if df.shape[1] != 6:
        raise ValueError(f"Expected 6 columns, got {df.shape[1]}")
    out = df.copy()  # copy so the caller's DataFrame is never changed
    if set(out.columns) == set(UCI_NAMES):
        out = out.rename(columns=UCI_NAMES)
    elif set(out.columns) != set(FEATURES + [TARGET]):
        out.columns = FEATURES + [TARGET]
    # Same column order every time, whatever order the file used.
    return out[FEATURES + [TARGET]].astype(float)


def audit(df: pd.DataFrame) -> dict:
    """Summarise data quality before any model is fitted.

    Course link: Week 5.6 data audit ("checks schema, units, missing values,
    duplicates, validity rules and coverage"). Problems are found here,
    before they can influence a fitted model.

    Returns
    -------
    dict
        Counts that are printed by `run_all.py` and quoted in the report.
    """
    groups = make_run_groups(df)
    return {
        "rows": len(df),
        # isna() marks each missing cell True; summing twice counts them all.
        "missing_values": int(df.isna().sum().sum()),
        # duplicated() is True for a row identical to an earlier row.
        # Week 5.6: copies must not count as new evidence.
        "duplicate_rows": int(df.duplicated().sum()),
        # Validity rule: frequency, chord, speed and thickness must be > 0.
        "non_positive_physical_values": int(
            (df[["frequency", "chord", "velocity", "thickness"]] <= 0).sum().sum()
        ),
        # Coverage: how many independent test runs and distinct conditions.
        "test_runs": int(groups.nunique()),
        "rows_per_run_min": int(groups.value_counts().min()),
        "rows_per_run_max": int(groups.value_counts().max()),
        "distinct_frequencies": int(df["frequency"].nunique()),
        "distinct_aoa": int(df["aoa"].nunique()),
        "distinct_chords": int(df["chord"].nunique()),
        "distinct_velocities": int(df["velocity"].nunique()),
    }


LONG_NAMES = {
    "frequency": "Frequency",
    "aoa": "Angle of attack",
    "chord": "Chord length",
    "velocity": "Free-stream velocity",
    "thickness": "Suction-side displacement thickness",
    "spl": "Scaled sound pressure level",
}
"""Full names of each column, for tables in the report."""


def input_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Table of the range covered by each input (and the target) in the data.

    Course link: Week 5.6 "coverage" (which operating conditions the data
    represents). The min and max define the measured domain: inside it the
    model interpolates, outside it the model extrapolates (Week 3). The number
    of distinct values shows how finely each input was sampled; for example,
    velocity was tested at only 4 speeds, which limits how many powers of
    velocity a polynomial can support.

    Returns
    -------
    pandas.DataFrame
        One row per variable with columns `variable`, `column`, `role`
        (input or target), `unit`, `min`, `max`, `mean` and `distinct_values`.
    """
    rows = []
    for col in FEATURES + [TARGET]:
        s = df[col]
        rows.append({
            "variable": LONG_NAMES[col],
            "column": col,
            "role": "target" if col == TARGET else "input",
            "unit": UNITS[col],
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            # nunique() counts how many different values appear.
            "distinct_values": int(s.nunique()),
        })
    return pd.DataFrame(rows)


def make_run_groups(df: pd.DataFrame) -> pd.Series:
    """Label each row with the wind-tunnel test run it came from.

    Course link: Week 5.6 grouped split. In the pump example the group was
    `pump_id`; here the group is the test run. One run is one airfoil (chord)
    at one angle and one velocity, and the microphone records a whole
    spectrum, giving one row per frequency band. Rows from the same run share
    every input except frequency, so they are closely related.

    The NASA data has no run-id column, so one is built from the run
    conditions. Displacement thickness is included in the key because it is
    set by the run conditions and the boundary-layer trip.

    Returns
    -------
    pandas.Series
        Integer run id per row, aligned with `df.index`.
    """
    # Glue the run conditions into one text label per row, e.g.
    # "0.3048|0.0|71.3|0.00266". Rounding stops tiny float differences from
    # splitting one run into two.
    key = (
        df["chord"].round(4).astype(str)
        + "|" + df["aoa"].round(1).astype(str)
        + "|" + df["velocity"].round(1).astype(str)
        + "|" + df["thickness"].round(7).astype(str)
    )
    # factorize() turns each distinct label into an integer 0, 1, 2, ...
    codes, _ = pd.factorize(key)
    return pd.Series(codes, index=df.index, name="run_id")


def check_inputs(X: pd.DataFrame) -> np.ndarray:
    """Validate inputs before a prediction and flag unsupported rows.

    This is how the project handles "invalid, missing or unsupported inputs"
    (project brief, section 2).

    Rules
    -----
    - Missing columns, non-numeric values or NaNs -> `ValueError` (no
      prediction is better than a silent wrong one).
    - Frequency, chord, velocity or thickness <= 0 -> `ValueError`
      (physically impossible).
    - Values outside `TESTED_RANGE` -> a `UserWarning`; the row is still
      predicted but marked unsupported (extrapolation).

    Parameters
    ----------
    X : pandas.DataFrame
        Must contain every column in `FEATURES` (extra columns are ignored).

    Returns
    -------
    numpy.ndarray of bool
        `True` where the row lies inside the tested range.
    """
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    # Double brackets keep a 2D table (Week 5.1: X must be 2D).
    Xf = X[FEATURES]
    if not all(pd.api.types.is_numeric_dtype(Xf[c]) for c in FEATURES):
        raise ValueError("All inputs must be numeric")
    if Xf.isna().any().any():
        raise ValueError("Inputs contain missing values (NaN)")
    positive = ["frequency", "chord", "velocity", "thickness"]
    if (Xf[positive] <= 0).any().any():
        raise ValueError(f"{positive} must all be > 0")

    # Start by assuming every row is in range, then knock out any row that
    # fails any one limit. "&=" means "stay True only if this is also True".
    in_range = np.ones(len(Xf), dtype=bool)
    for col, (lo, hi) in TESTED_RANGE.items():
        in_range &= Xf[col].between(lo, hi).to_numpy()
    if not in_range.all():
        warnings.warn(
            f"{(~in_range).sum()} row(s) are outside the tested range; "
            "predictions there are extrapolations and not supported.",
            UserWarning,
            stacklevel=2,
        )
    return in_range
