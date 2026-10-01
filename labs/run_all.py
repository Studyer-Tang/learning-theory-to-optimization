"""Generate synthetic experiments, plots, CSV measurements, and a run report.

Run from the repository root with: python labs/run_all.py
No network or input dataset is used. Seeds are separate for each experiment.
"""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import (gaussian_test_mse, quadratic_gaps, quadratic_gd, ridge,
                  scalar_clt_variance, scalar_stationary_objective)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "labs" / "results"
BLUE, ORANGE, GREEN, GRAY = "#2563eb", "#e67e22", "#159a80", "#64748b"


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def csv_save(name, columns, rows):
    with (RESULTS / f"{name}.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def ridge_lab():
    rng = np.random.default_rng(1101)
    n, d = 40, 65
    truth = np.zeros(d)
    truth[:5] = [2, 1.5, 1, 0.5, 0.2]
    X = rng.normal(size=(n, d))
    y = X @ truth + rng.normal(scale=2, size=n)
    V = rng.normal(size=(500, d))
    vy = V @ truth + rng.normal(scale=2, size=500)
    Z = rng.normal(size=(2000, d))
    zy = Z @ truth + rng.normal(scale=2, size=2000)
    penalties = np.r_[0.0, np.logspace(-5, 2, 65)]
    fits = [ridge(X, y, penalty) for penalty in penalties]
    train = np.array([np.mean((X @ fit - y) ** 2) / 2 for fit in fits])
    validation = np.array([np.mean((V @ fit - vy) ** 2) / 2 for fit in fits])
    selected = int(np.argmin(validation))
    # Test labels are evaluated only after the validation choice is frozen.
    test_loss = float(np.mean((Z @ fits[selected] - zy) ** 2) / 2)
    csv_save("ridge", ["lambda", "training_half_mse", "validation_half_mse"],
             zip(penalties, train, validation))
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.plot(penalties, train, label="Training", color=BLUE)
    ax.plot(penalties, validation, label="Validation", color=ORANGE)
    ax.axvline(penalties[selected], color=GREEN, ls="--", label="Validation choice")
    ax.set(xscale="symlog", xlabel="Ridge penalty (linear near zero)",
           ylabel="Half mean squared error", title="Better training fit need not predict better")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlim(0, float(penalties[-1]))
    ax.legend()
    save(fig, "ridge")
    return {"seed": 1101, "n": n, "d": d,
            "selected_lambda": float(penalties[selected]), "test_half_mse": test_loss,
            "minimum_training_increment": float(np.min(np.diff(train)))}


def double_descent_lab():
    rng = np.random.default_rng(1102)
    n, max_d, repetitions, noise = 30, 80, 240, 0.25
    truth = np.zeros(max_d)
    truth[:5] = [2, 1.5, 1, 0.7, 0.4]
    dims = np.array([1, 2, 3, 5, 8, 15, 22, 26, 28, 29, 30, 31, 32, 34, 40, 50, 65, 80])
    samples = np.zeros((repetitions, len(dims)))
    training = np.zeros_like(samples)
    for r in range(repetitions):
        X = rng.normal(size=(n, max_d))
        y = X @ truth + rng.normal(scale=np.sqrt(noise), size=n)
        for j, d in enumerate(dims):
            fit = np.linalg.lstsq(X[:, :d], y, rcond=None)[0]
            samples[r, j] = noise + np.sum(truth[d:] ** 2) + np.sum((fit - truth[:d]) ** 2)
            training[r, j] = np.mean((X[:, :d] @ fit - y) ** 2)
    theoretical = np.array([gaussian_test_mse(n, int(d), truth, noise) for d in dims])
    means, medians = samples.mean(axis=0), np.median(samples, axis=0)
    csv_save("double_descent", ["d", "theoretical_test_mse", "finite_run_mean_test_mse",
                                "finite_run_median_test_mse", "mean_training_mse"],
             zip(dims, theoretical, means, medians, training.mean(axis=0)))
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    # NaN creates an actual break across the infinite-expectation dimensions.
    ax.plot(dims, np.where(np.isfinite(theoretical), theoretical, np.nan),
            color=BLUE, label="Finite theoretical expectation")
    ax.scatter(dims, means, color=ORANGE, s=25, label="Mean of 240 realized risks")
    ax.plot(dims, medians, color=GREEN, ls=":", label="Median (a different quantity)")
    ax.axvspan(n-1, n+1, alpha=0.12, color=ORANGE, label="Infinite expected MSE")
    ax.set(yscale="log", xlabel="Retained coordinates d; fixed n = 30", ylabel="Test MSE (log scale)",
           title="Double descent for a fixed Gaussian signal")
    ax.legend(fontsize=8)
    save(fig, "double-descent")
    finite_far = (dims <= 22) | (dims >= 40)
    relative = np.abs(means[finite_far] / theoretical[finite_far] - 1)
    return {"seed": 1102, "n": n, "repetitions": repetitions,
            "critical_dimensions": [29, 30, 31],
            "max_relative_mean_error_away_from_peak": float(np.max(relative)),
            "max_training_mse_after_interpolation": float(training[:, dims >= n].max())}


def gd_lab():
    H = np.diag([1.0, 9.0])
    initial = np.array([1.0, 1.0])
    steps = [0.08, 1/9, 0.2, 2/9, 0.24]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    rows, summaries = [], []
    for step in steps:
        points = quadratic_gd(H, initial, step, 60)
        gaps = quadratic_gaps(H, points)
        label = f"step = {step:.3f}"
        axes[0].semilogy(gaps, label=label)
        axes[1].plot(np.arange(16), points[:16, 1], label=label)
        q = float(max(abs(1-step), abs(1-9*step)))
        summaries.append({"step": step, "q": q, "final_gap": float(gaps[-1])})
        rows.extend((step, t, *point, gap) for t, (point, gap) in enumerate(zip(points, gaps)))
    axes[0].set(title="Objective gap", xlabel="Updates", ylabel="Gap (log scale)")
    axes[1].set(title="High-curvature coordinate", xlabel="Updates", ylabel="Error coordinate")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color=GRAY, lw=0.7)
    save(fig, "gd-stability")
    csv_save("gd", ["step", "update", "error_low", "error_high", "objective_gap"], rows)
    return {"eigenvalues": [1, 9], "steps": summaries}


def sgd_lab():
    rng = np.random.default_rng(1103)
    paths, updates, h, sigma2, constant = 12000, 300, 2.0, 1.0, 0.15
    x_fixed = np.full(paths, 3.0)
    x_decay = x_fixed.copy()
    analytic_fixed = analytic_decay = 9.0
    rows = [(0, 9.0, 9.0, 9.0, 9.0)]  # h/2 = 1 in this experiment
    for t in range(1, updates+1):
        noise = rng.normal(size=paths)
        eta = 1 / (t + 4)
        x_fixed = (1-constant*h) * x_fixed - constant*noise
        x_decay = (1-eta*h) * x_decay - eta*noise
        analytic_fixed = (1-constant*h)**2 * analytic_fixed + constant**2*sigma2
        analytic_decay = (1-eta*h)**2 * analytic_decay + eta**2*sigma2
        rows.append((t, np.mean(x_fixed*x_fixed), np.mean(x_decay*x_decay),
                     analytic_fixed, analytic_decay))
    data = np.asarray(rows)
    floor = scalar_stationary_objective(h, constant, sigma2)
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.semilogy(data[:, 0], data[:, 1], color=ORANGE, label="Fixed-step Monte Carlo")
    ax.semilogy(data[:, 0], data[:, 2], color=BLUE, label="Decreasing-step Monte Carlo")
    ax.semilogy(data[:, 0], data[:, 3], color=GRAY, ls="--", label="Exact fixed-step expectation")
    ax.semilogy(data[:, 0], data[:, 4], color=GREEN, ls=":", label="Exact decreasing-step expectation")
    ax.axhline(floor, color=ORANGE, alpha=0.6, ls=":")
    ax.set(xlabel="Updates", ylabel="Expected objective gap", title="Contraction with noise: a solvable scalar example")
    ax.legend(fontsize=8)
    save(fig, "sgd-noise")
    csv_save("sgd", ["update", "mc_fixed", "mc_decay", "exact_fixed", "exact_decay"], rows)
    return {"seed": 1103, "paths": paths, "updates": updates,
            "exact_stationary_objective": floor,
            "mc_final_fixed": float(data[-1, 1]), "mc_final_decay": float(data[-1, 2]),
            "exact_final_decay": float(data[-1, 4])}


def uncertainty_lab():
    rng = np.random.default_rng(1104)
    paths, updates, h = 5000, 5000, 2.0
    x = np.ones((3, paths))
    average_sum = np.zeros(paths)
    # Final x is the result after 'updates' updates; average uses pre-update iterates.
    for t in range(1, updates+1):
        average_sum += x[2]
        steps = np.array([0.5/t, 1.0/t, 0.5*t**(-0.7)])[:, None]
        noise = rng.normal(size=(3, paths))
        x -= steps * (h*x + noise)
    scaled = np.array([x[0], x[1], average_sum/updates]) * np.sqrt(updates)
    limits = [scalar_clt_variance(h, 0.5, 1), scalar_clt_variance(h, 1, 1), 1/h**2]
    names = ["Final: gain 0.5 / t", "Final: gain 1.0 / t", "PR average: 0.5 / t^0.7"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    summaries = []
    for i, ax in enumerate(axes):
        grid = np.linspace(-2.2, 2.2, 300)
        v = limits[i]
        density = np.exp(-grid**2/(2*v)) / np.sqrt(2*np.pi*v)
        ax.hist(scaled[i], bins=42, density=True, color=BLUE, alpha=0.45, label="Finite run")
        ax.plot(grid, density, color=ORANGE, label="Asymptotic reference")
        ax.set(title=names[i], xlabel="sqrt(T) × parameter error", ylabel="Density")
        summaries.append({"method": names[i], "theoretical_limit_variance": v,
                          "empirical_scaled_variance": float(np.var(scaled[i], ddof=1)),
                          "empirical_scaled_mean": float(np.mean(scaled[i]))})
    axes[0].legend(fontsize=8)
    save(fig, "uncertainty")
    csv_save("uncertainty", ["replicate", "final_gain_half", "final_gain_one", "pr_average"],
             ((i, *scaled[:, i]) for i in range(paths)))
    return {"seed": 1104, "paths": paths, "updates": updates, "methods": summaries}


def main():
    ASSETS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": 0.16})
    metrics = {"runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "matplotlib": matplotlib.__version__},
               "ridge": ridge_lab(), "double_descent": double_descent_lab(),
               "gd": gd_lab(), "sgd": sgd_lab(), "uncertainty": uncertainty_lab()}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, name, title in zip(axes.flat,
                              ["ridge", "double-descent", "gd-stability", "sgd-noise"],
                              ["Generalization", "Interpolation", "Deterministic progress", "Stochastic progress"]):
        ax.imshow(plt.imread(ASSETS / f"{name}.png"))
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    save(fig, "overview")
    r, dd, sg = metrics["ridge"], metrics["double_descent"], metrics["sgd"]
    report = f"""# Reproducibility report

Generated by `python labs/run_all.py`. All data are synthetic; seeds and raw measurements are in [results/metrics.json](results/metrics.json) and adjacent CSV files.

Runtime: Python {platform.python_version()}, NumPy {np.__version__}, Matplotlib {matplotlib.__version__}. This report describes a local execution, not an independently replicated result or a hosted CI run.

| Experiment | Measured result | Interpretation |
|---|---|---|
| Ridge | Validation selected lambda = {r['selected_lambda']:.6g}; held-out test half-MSE = {r['test_half_mse']:.6g} | Test evaluation occurs after selection |
| Double descent | Largest relative mean discrepancy away from the peak = {dd['max_relative_mean_error_away_from_peak']:.3%} | Finite Monte Carlo, not an exact expectation |
| Interpolation | Largest training MSE for d >= n = {dd['max_training_mse_after_interpolation']:.3g} | Numerical interpolation to floating-point accuracy |
| Constant-step SGD | Final mean gap = {sg['mc_final_fixed']:.6g}; exact stationary gap = {sg['exact_stationary_objective']:.6g} | Compare realized Monte Carlo with a solvable expectation |
| Decreasing-step SGD | Final mean gap = {sg['mc_final_decay']:.6g}; exact finite-time expectation = {sg['exact_final_decay']:.6g} | Decreasing steps reduce the residual error |

## Finite-horizon uncertainty experiment

| Output | Limit variance | Empirical scaled variance | Empirical scaled mean |
|---|---:|---:|---:|
"""
    for item in metrics["uncertainty"]["methods"]:
        report += f"| {item['method']} | {item['theoretical_limit_variance']:.6g} | {item['empirical_scaled_variance']:.6g} | {item['empirical_scaled_mean']:.6g} |\n"
    report += """
The normal curves are asymptotic references, not exact finite-time distributions. Initialization, discretization, and finite Monte Carlo sampling cause visible discrepancies. The PR average uses exponent 0.7, distinct from the 1/t last-iterate runs.

## Limits of these checks

The critical Gaussian dimensions have infinite expected test risk when effective noise is positive. A finite average of 240 realized risks remains finite and cannot estimate a finite value that does not exist. The median is intentionally labeled as a different statistic. Numerical GD trajectories illustrate a single quadratic. None of these experiments establish universal rates for neural networks or Adam. Platform-dependent linear algebra can change final digits even with identical seeds.
"""
    (ROOT / "labs" / "RESULTS.md").write_text(report, encoding="utf-8")
    print("Generated five experiments, six figures, CSV measurements, and RESULTS.md.")


if __name__ == "__main__":
    main()
