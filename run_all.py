"""
Run the whole project end to end and write every result and figure.

Usage (from the repository folder):

    python run_all.py

Outputs go to `results/` (CSV/JSON/TXT tables) and `figures/` (PNG plots).
The script is deterministic: a fixed random seed (42) is used everywhere, so
re-running gives the same numbers.

Workflow
--------
1. Load and audit the data; label each row with its wind-tunnel test run.
2. Hold out ~20% of test runs as a final test set (touched once, in step 5).
3. Degree sweep 1-5 on the remaining runs with random and grouped 5-fold CV.
4. Select the simplest degree within 5% of the best *grouped* RMSE.
5. Fit baseline, linear and selected polynomial on all development runs and
   score them once on the final test runs.
6. Physical check (velocity trend), sensitivity (permutation importance)
   and a profile of the most expensive fit.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Let `python run_all.py` find the package in src/ without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from airfoil_noise import plots  # noqa: E402
from airfoil_noise.data import (  # noqa: E402
    FEATURES, TARGET, audit, load_airfoil, make_run_groups,
)
from airfoil_noise.evaluation import (  # noqa: E402
    degree_sweep,
    hold_out_test_runs,
    out_of_fold_predictions,
    regression_metrics,
    select_degree,
    velocity_trend_check,
)
from airfoil_noise.models import (  # noqa: E402
    condition_number, make_linear, make_mean_baseline, make_polynomial,
)

ROOT = Path(__file__).resolve().parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SEED = 42


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # 1. Data audit ---------------------------------------------------------
    df = load_airfoil()
    df["run_id"] = make_run_groups(df)
    info = audit(df)
    print("Audit:", info)

    # 2. Final test set: whole runs, set aside before any modelling ----------
    dev_idx, test_idx = hold_out_test_runs(df, df["run_id"], 0.2, SEED)
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    assert not set(dev["run_id"]) & set(test["run_id"]), "run leaked into test set"
    info.update(dev_rows=len(dev), dev_runs=dev["run_id"].nunique(),
                test_rows=len(test), test_runs=test["run_id"].nunique())
    (RESULTS / "audit.json").write_text(json.dumps(info, indent=2))

    Xd, yd, gd = dev[FEATURES], dev[TARGET], dev["run_id"]
    Xt, yt = test[FEATURES], test[TARGET]

    # 3-4. Degree sweep and selection ---------------------------------------
    sweep = degree_sweep(make_polynomial, Xd, yd, gd, degrees=range(1, 6), seed=SEED)
    sweep.to_csv(RESULTS / "degree_sweep.csv", index=False)
    chosen = select_degree(sweep, tolerance=0.05)
    rnd = sweep[sweep["split"] == "random"]
    random_choice = int(rnd.loc[rnd["val_rmse"].idxmin(), "degree"])
    print(sweep.round(3).to_string())
    print(f"Selected degree (grouped CV, 5% rule): {chosen}; "
          f"random split alone would pick: {random_choice}")

    # 5. Final test, used once ------------------------------------------------
    candidates = {
        "Mean baseline": make_mean_baseline(),
        "Linear regression": make_linear(),
        f"Polynomial degree {chosen} (selected)": make_polynomial(chosen),
    }
    if random_choice != chosen:
        candidates[f"Polynomial degree {random_choice} (random-split pick)"] = (
            make_polynomial(random_choice))
    test_rows, test_preds = [], {}
    for name, model in candidates.items():
        model.fit(Xd, yd)
        p = model.predict(Xt)
        test_preds[name] = p
        test_rows.append({"model": name, **regression_metrics(yt, p)})
    final = pd.DataFrame(test_rows)
    final.to_csv(RESULTS / "final_test.csv", index=False)
    print(final.round(3).to_string())
    selected_name = f"Polynomial degree {chosen} (selected)"
    selected_model = candidates[selected_name]

    # 6a. Physical check: noise should rise with velocity --------------------
    checks = {}
    for name, factory in [("linear", make_linear),
                          (f"poly{chosen}", lambda: make_polynomial(chosen))]:
        oof = out_of_fold_predictions(factory, Xd, yd, gd)
        checks[name] = velocity_trend_check(dev, oof)
        if name != "linear":
            plots.velocity_check_plot(dev, oof, FIGURES / "velocity_check.png")
    (RESULTS / "velocity_check.json").write_text(json.dumps(checks, indent=2))
    print("Velocity check:", json.dumps(checks, indent=2))

    # 6b. Sensitivity: permutation importance on the test runs ---------------
    pi = permutation_importance(selected_model, Xt, yt, n_repeats=20,
                                random_state=SEED, scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": FEATURES, "rmse_increase_db": pi.importances_mean,
                        "std": pi.importances_std})
    imp.sort_values("rmse_increase_db", ascending=False).to_csv(
        RESULTS / "permutation_importance.csv", index=False)

    # 6c. Numerical stability (Week 9.1): condition number with/without scaling
    cond = pd.DataFrame([
        {"degree": d,
         "kappa_scaled": condition_number(Xd, d, scaled=True),
         "kappa_unscaled": condition_number(Xd, d, scaled=False)}
        for d in range(1, 6)
    ])
    cond.to_csv(RESULTS / "condition_numbers.csv", index=False)
    print(cond.to_string())

    # 6d. Profile the most expensive fit (degree 5) -------------------------
    prof = cProfile.Profile()
    prof.enable()
    make_polynomial(5).fit(Xd, yd)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(15)
    (RESULTS / "profile_degree5.txt").write_text(buf.getvalue())

    # Figures ------------------------------------------------------------------
    plots.degree_sweep_plot(sweep, chosen, FIGURES / "degree_sweep.png")
    plots.runtime_plot(sweep, FIGURES / "runtime.png")
    plots.observed_vs_predicted_plot(
        yt, {k: v for k, v in test_preds.items() if k != "Mean baseline"},
        FIGURES / "observed_vs_predicted.png")
    plots.residual_frequency_plot(test, test_preds[selected_name],
                                  FIGURES / "residuals_vs_frequency.png")
    example_runs = (test.groupby("run_id").size().sort_values(ascending=False)
                    .index[:3].tolist())
    plots.spectra_plot(test, {selected_name.split(" (")[0]: test_preds[selected_name],
                              "Linear": test_preds["Linear regression"]},
                       example_runs, FIGURES / "example_spectra.png")
    plots.importance_plot(imp, FIGURES / "permutation_importance.png")
    print(f"Done. Results in {RESULTS}, figures in {FIGURES}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
