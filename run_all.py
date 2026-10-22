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
    FEATURES, TARGET, audit, input_ranges, load_airfoil, make_run_groups,
)
from airfoil_noise.evaluation import (  # noqa: E402
    cross_validate_model,
    degree_sweep,
    grouped_splitter,
    random_splitter,
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
    ranges = input_ranges(df)
    ranges.to_csv(RESULTS / "input_ranges.csv", index=False)
    print(ranges.to_string(index=False))

    # 2. Final test set: whole runs, set aside before any modelling ----------
    dev_idx, test_idx = hold_out_test_runs(df, df["run_id"], 0.2, SEED)
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    assert not set(dev["run_id"]) & set(test["run_id"]), "run leaked into test set"
    info.update(dev_rows=len(dev), dev_runs=dev["run_id"].nunique(),
                holdout_rows=len(test), holdout_runs=test["run_id"].nunique())
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

    # 4b. Mean baseline in cross-validation (Week 5.5 DummyRegressor) --------
    # Predicts the training-fold mean SPL for every row, ignoring all inputs.
    # Scored with the same folds as the sweep, so it is a fair reference line:
    # any model that cannot beat it has learned nothing useful.
    baseline_rows = []
    for split_name, splitter in [("random", random_splitter(5, SEED)),
                                 ("grouped", grouped_splitter(5))]:
        cv = cross_validate_model(make_mean_baseline(), Xd, yd, splitter, gd)
        baseline_rows.append({"model": "Mean baseline", "split": split_name,
                              "val_rmse": cv["val_rmse"].mean(),
                              "val_rmse_std": cv["val_rmse"].std(),
                              "val_mae": cv["val_mae"].mean(),
                              "val_r2": cv["val_r2"].mean()})
    baseline_cv = pd.DataFrame(baseline_rows)
    baseline_cv.to_csv(RESULTS / "baseline_cv.csv", index=False)
    print(baseline_cv.round(3).to_string(index=False))

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
    grouped_baseline = float(
        baseline_cv.loc[baseline_cv["split"] == "grouped", "val_rmse"].iloc[0])
    plots.degree_sweep_plot(sweep, chosen, FIGURES / "degree_sweep.png",
                            baseline_rmse=grouped_baseline)
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

    write_summary(info, ranges, sweep, chosen, random_choice, final,
                  checks[f"poly{chosen}"], imp, cond, baseline_cv)
    print(f"Done. Results in {RESULTS}, figures in {FIGURES}")


def write_summary(info, ranges, sweep, chosen, random_choice, final, vcheck, imp,
                  cond, baseline_cv) -> None:
    """Write results/summary.md: the headline numbers in one readable page.

    Regenerated on every run, so the numbers always match the latest data.
    """
    g = sweep[sweep["split"] == "grouped"].set_index("degree")
    r = sweep[sweep["split"] == "random"].set_index("degree")
    lines = [
        "# Results summary",
        "",
        "Generated automatically by `run_all.py`. Do not edit by hand.",
        "",
        "## Data",
        f"- {info['rows']} rows from {info['test_runs']} wind-tunnel test runs "
        f"({info['missing_values']} missing values, {info['duplicate_rows']} duplicate rows).",
        f"- Development set: {info['dev_rows']} rows / {info['dev_runs']} runs. "
        f"Final test set: {info['holdout_rows']} rows / {info['holdout_runs']} runs (whole runs held out).",
        "",
        "## Inputs and target: range in the data",
        "",
        "| Variable | Role | Unit | Min | Max | Distinct values |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in ranges.iterrows():
        lines.append(f"| {row['variable']} | {row['role']} | {row['unit']} | "
                     f"{row['min']:,.6g} | {row['max']:,.6g} | {row['distinct_values']} |")
    lines += [
        "",
        "## Degree sweep (5-fold cross-validation on the development runs)",
        "",
        "| Degree | Terms | Train RMSE (dB) | Random-split RMSE (dB) | Grouped-split RMSE (dB) | Grouped R² | Fit time (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    # Initial reference model first: the mean baseline (DummyRegressor).
    b = baseline_cv.set_index("split")
    lines.append(
        f"| Mean baseline (DummyRegressor) | 0 | – | {b.loc['random', 'val_rmse']:.2f} | "
        f"{b.loc['grouped', 'val_rmse']:.2f} | {b.loc['grouped', 'val_r2']:.2f} | – |")
    for d in g.index:
        lines.append(
            f"| {d} | {g.loc[d, 'n_terms']} | {g.loc[d, 'train_rmse']:.2f} | "
            f"{r.loc[d, 'val_rmse']:.2f} | {g.loc[d, 'val_rmse']:.2f} | "
            f"{g.loc[d, 'val_r2']:.2f} | {g.loc[d, 'fit_ms']:.1f} |")
    lines += [
        "",
        f"- **Selected degree (grouped split, simplest within 5% of best): {chosen}**",
        f"- A random row split alone would have selected degree {random_choice}.",
        "",
        "## Final test (held-out runs, used once)",
        "",
        "| Model | RMSE (dB) | MAE (dB) | R² | Underpredicted by >3 dB |",
        "|---|---|---|---|---|",
    ]
    for _, row in final.iterrows():
        lines.append(f"| {row['model']} | {row['rmse']:.2f} | {row['mae']:.2f} | "
                     f"{row['r2']:.2f} | {100 * row['under3']:.0f}% |")
    lines += [
        "",
        "## Physical check: noise vs velocity (31.7 → 71.3 m/s)",
        f"- {vcheck['pairs']} matched pairs (same chord, angle and frequency).",
        f"- Model predicts an increase in {100 * vcheck['model_increase_fraction']:.0f}% "
        f"of pairs (measured: {100 * vcheck['measured_increase_fraction']:.0f}%).",
        f"- Mean increase: predicted {vcheck['mean_predicted_increase_db']:+.1f} dB, "
        f"measured {vcheck['mean_measured_increase_db']:+.1f} dB.",
        "",
        "## Sensitivity (permutation importance on the final test runs)",
        "",
        "| Input | RMSE increase when shuffled (dB) |",
        "|---|---|",
    ]
    for _, row in imp.sort_values("rmse_increase_db", ascending=False).iterrows():
        lines.append(f"| {row['feature']} | {row['rmse_increase_db']:.2f} |")
    lines += [
        "",
        "## Numerical stability (condition number of the design matrix)",
        "",
        "| Degree | With StandardScaler | Without scaling |",
        "|---|---|---|",
    ]
    for _, row in cond.iterrows():
        lines.append(f"| {int(row['degree'])} | {row['kappa_scaled']:.1e} | "
                     f"{row['kappa_unscaled']:.1e} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
