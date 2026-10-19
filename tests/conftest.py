"""
Shared test data ("fixtures") for the test files.

pytest runs any function named `test_*` in files named `test_*.py`. A
fixture is a function marked `@pytest.fixture`; when a test lists the
fixture's name as an argument, pytest calls the fixture and passes in its
result. This lets every test start from the same small, known dataset.

The tests use small *synthetic* data instead of the real NASA file, so they
run in under a second, need no internet, and have answers we know exactly.
"""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.data import FEATURES, TARGET


@pytest.fixture
def toy_runs() -> pd.DataFrame:
    """Six fake wind-tunnel runs, each measured at five frequencies.

    Values sit inside the real tested range. SPL follows a simple made-up
    rule so the numbers are realistic but fully known.
    """
    rows = []
    runs = [  # (chord m, angle deg, velocity m/s, thickness m)
        (0.0254, 0.0, 31.7, 0.0004),
        (0.0254, 0.0, 71.3, 0.0005),
        (0.1016, 4.0, 39.6, 0.0020),
        (0.1016, 4.0, 55.5, 0.0018),
        (0.3048, 7.3, 31.7, 0.0100),
        (0.3048, 7.3, 71.3, 0.0090),
    ]
    for chord, aoa, vel, thick in runs:
        for f in [500.0, 1000.0, 2000.0, 4000.0, 8000.0]:
            spl = 120 + 50 * np.log10(vel / 31.7) - 3 * np.log10(f / 500) + 10 * chord
            rows.append([f, aoa, chord, vel, thick, spl])
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])
