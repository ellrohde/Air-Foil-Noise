"""
Figures for the report. Each function saves one PNG and returns its path.

Colours: blue = grouped split / selected model, orange = random split /
comparison. Every figure also labels its series in text, so colour is never
the only cue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8984", "#2b2b2a"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _hz_ticks(ax) -> None:
    """Plain frequency labels (200, 1k, 5k, 20k) on a log axis, no clutter."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ticks = [200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def degree_sweep_plot(sweep: pd.DataFrame, selected: int, path: Path,
                      baseline_rmse: float | None = None) -> Path:
    """Validation RMSE vs polynomial degree for random and grouped splits.

    Log y-axis, because grouped errors at degree 4-5 are ~10x larger than the
    rest. Training RMSE (dashed) shows the overfitting gap opening up. If
    `baseline_rmse` is given, the mean baseline (DummyRegressor, grouped
    split) is drawn as a horizontal reference line: points above it are
    worse than predicting the average.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if baseline_rmse is not None:
        ax.axhline(baseline_rmse, color=INK, lw=1, ls=":",
                   label=f"Mean baseline, grouped ({baseline_rmse:.2f} dB)")
    for split, colour, label in [
        ("random", ORANGE, "Random row split (validation)"),
        ("grouped", BLUE, "Grouped split by test run (validation)"),
    ]:
        s = sweep[sweep["split"] == split]
        ax.errorbar(s["degree"], s["val_rmse"], yerr=s["val_rmse_std"], color=colour,
                    marker="o", ms=6, capsize=3, label=label)
    tr = sweep[sweep["split"] == "grouped"]
    ax.plot(tr["degree"], tr["train_rmse"], color=GREY, ls="--", marker="o", ms=5,
            label="Training error")
    sel = sweep[(sweep["split"] == "grouped") & (sweep["degree"] == selected)]
    ax.annotate(f"selected: degree {selected}", (selected, float(sel["val_rmse"].iloc[0])),
                xytext=(12, -28), textcoords="offset points", color=INK,
                arrowprops={"arrowstyle": "-", "color": GREY})
    ax.set_yscale("log")
    ax.set_xticks(sweep["degree"].unique())
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE (dB, log scale)")
    ax.set_title("Accuracy vs model complexity (5-fold CV, mean ± std)", loc="left")
    ax.legend(loc="upper left")
    return _save(fig, path)


def runtime_plot(sweep: pd.DataFrame, path: Path) -> Path:
    """Measured fit time and number of polynomial terms vs degree."""
    s = sweep[sweep["split"] == "grouped"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(s["degree"], s["fit_ms"], color=BLUE, width=0.6)
    for b, n in zip(bars, s["n_terms"]):
        ax.annotate(f"{n} terms", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", color=INK)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Median fit time (ms)")
    ax.set_title("Cost of complexity: fit time on ~1,200 rows", loc="left")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def observed_vs_predicted_plot(y, preds: dict, path: Path) -> Path:
    """Observed vs predicted SPL on the final test runs, one panel per model."""
    fig, axes = plt.subplots(1, len(preds), figsize=(3.2 * len(preds), 3.3),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = float(np.min(y)) - 2, float(np.max(y)) + 2
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(y, p, s=10, color=BLUE, alpha=0.6, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(name, loc="left")
        ax.set_xlabel("Measured SPL (dB)")
    axes[0].set_ylabel("Predicted SPL (dB)")
    return _save(fig, path)


def residual_frequency_plot(df_test: pd.DataFrame, pred, path: Path) -> Path:
    """Residuals (measured - predicted) against frequency on the test runs.

    Points above zero are underpredictions (the costly direction).
    """
    resid = df_test["spl"].to_numpy() - np.asarray(pred)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.scatter(df_test["frequency"], resid, s=10, color=BLUE, alpha=0.6, edgecolor="none")
    ax.axhline(0, color=GREY, lw=1)
    ax.axhline(3, color=ORANGE, lw=1, ls="--", label="+3 dB: costly underprediction above")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xscale("log")
    _hz_ticks(ax)
    ax.set_xlabel("Frequency (Hz, log scale)")
    ax.set_ylabel("Residual (dB)")
    ax.set_title("Where the selected model is wrong (final test runs)", loc="left")
    return _save(fig, path)


def spectra_plot(df_test: pd.DataFrame, preds: dict, run_ids, path: Path) -> Path:
    """Measured vs predicted noise spectra for a few held-out test runs."""
    fig, axes = plt.subplots(1, len(run_ids), figsize=(3.2 * len(run_ids), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    styles = [(BLUE, "-"), (ORANGE, "--")]
    d = df_test.copy()
    for name, p in preds.items():
        d[name] = np.asarray(p)  # keep each prediction attached to its row
    for ax, rid in zip(axes, run_ids):
        r = d[d["run_id"] == rid].sort_values("frequency")
        ax.plot(r["frequency"], r["spl"], color=INK, marker="o", ms=4, lw=1.5, label="Measured")
        for name, (c, ls) in zip(preds, styles):
            ax.plot(r["frequency"], r[name], color=c, ls=ls, label=name)
        row = r.iloc[0]
        ax.set_title(f"c={row.chord:.3f} m, α={row.aoa:g}°, U={row.velocity:g} m/s",
                     loc="left", fontsize=8)
        ax.set_xscale("log")
        _hz_ticks(ax)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("SPL (dB)")
    axes[0].legend(loc="lower left", fontsize=7)
    return _save(fig, path)


def velocity_check_plot(df: pd.DataFrame, pred, path: Path) -> Path:
    """Measured vs predicted SPL increase from 31.7 to 71.3 m/s, per matched pair."""
    d = df.assign(pred=np.asarray(pred))
    lo, hi = d["velocity"].min(), d["velocity"].max()
    keys = ["chord", "aoa", "frequency"]
    p = d[d["velocity"] == lo].set_index(keys).join(
        d[d["velocity"] == hi].set_index(keys), lsuffix="_lo", rsuffix="_hi", how="inner")
    meas, model = p["spl_hi"] - p["spl_lo"], p["pred_hi"] - p["pred_lo"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.scatter(meas, model, s=12, color=BLUE, alpha=0.7, edgecolor="none")
    lim = [min(meas.min(), model.min()) - 2, max(meas.max(), model.max()) + 2]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1)
    ax.annotate("perfect agreement", (lim[1] - 1, lim[1] - 1), ha="right", va="top",
                color=GREY, fontsize=8)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.annotate("below 0: model says\nfaster = quieter", (lim[1] - 1, -0.5), ha="right",
                va="top", color=INK, fontsize=8)
    ax.set_xlabel(f"Measured increase, {lo:g}→{hi:g} m/s (dB)")
    ax.set_ylabel("Predicted increase (dB)")
    ax.set_title("Physical check: noise vs velocity", loc="left")
    return _save(fig, path)


def importance_plot(imp: pd.DataFrame, path: Path) -> Path:
    """Permutation importance: RMSE increase when one input is shuffled."""
    imp = imp.sort_values("rmse_increase_db")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.barh(imp["feature"], imp["rmse_increase_db"], xerr=imp["std"], color=BLUE,
            height=0.6, capsize=3)
    ax.set_xlabel("Increase in test RMSE when shuffled (dB)")
    ax.set_title("Sensitivity: which inputs matter most", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
