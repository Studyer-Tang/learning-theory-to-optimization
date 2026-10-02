"""Small numerical implementations corresponding to the companion's mathematics."""
from __future__ import annotations

import numpy as np


def ridge(X, y, penalty):
    """Minimize ||X theta-y||^2/(2n) + penalty*||theta||^2/2.

    Use a thin SVD and a numerical rank cutoff at zero penalty.
    """
    if penalty < 0:
        raise ValueError("penalty must be nonnegative")
    if penalty == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((s / (s * s + len(y) * penalty)) * (U.T @ y))


def quadratic_gd(H, initial, step, updates):
    points = [np.asarray(initial, dtype=float).copy()]
    for _ in range(updates):
        points.append(points[-1] - step * (H @ points[-1]))
    return np.asarray(points)


def quadratic_gaps(H, points):
    return 0.5 * np.einsum("...i,ij,...j->...", points, H, points)


def gaussian_test_mse(n, d, coefficients, noise_variance):
    """Finite-dimensional specialization with zero coefficients past the vector.

    np.inf denotes the positive-noise critical dimensions. This implementation
    explicitly rejects the degenerate critical zero-noise case instead of
    silently multiplying zero by an infinite inverse moment.
    """
    if n < 2 or d < 1 or noise_variance < 0:
        raise ValueError("require n >= 2, d >= 1, and nonnegative noise")
    included = float(np.sum(np.asarray(coefficients)[:d] ** 2))
    omitted = float(np.sum(np.asarray(coefficients)[d:] ** 2))
    effective = noise_variance + omitted
    if d in (n - 1, n, n + 1):
        if effective > 0:
            return np.inf
        raise ValueError("critical zero-noise case requires separate analysis")
    if d < n - 1:
        return effective * (1 + d / (n - d - 1))
    return included * (1 - n / d) + effective * (1 + n / (d - n - 1))


def realized_isotropic_test_mse(fit, coefficients, noise_variance):
    """Conditional test MSE for independent standard-normal test coordinates.

    This identity does not apply unchanged to correlated or non-unit-variance
    features: their coefficient-error term uses the test covariance matrix.
    """
    fit = np.asarray(fit, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if fit.ndim != 1 or coefficients.ndim != 1 or len(fit) > len(coefficients):
        raise ValueError("require vectors and no fit beyond the coefficient vector")
    if noise_variance < 0:
        raise ValueError("noise variance must be nonnegative")
    d = len(fit)
    return float(noise_variance + np.sum(coefficients[d:] ** 2)
                 + np.sum((fit - coefficients[:d]) ** 2))


def risk_diagnostics(samples):
    """Describe realized risks without asserting a finite population mean.

    No standard error or confidence interval is inferred from these values.
    The top 1% share uses ceil(0.01 * repetitions), including at least one risk.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError("require a nonempty finite vector of realized risks")
    if np.any(samples < 0):
        raise ValueError("risks must be nonnegative")
    running = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    total = float(np.sum(samples))
    count = max(1, int(np.ceil(0.01 * len(samples))))
    return {"running_mean": running,
            "mean": float(running[-1]),
            "median": float(np.quantile(samples, 0.5)),
            "q90": float(np.quantile(samples, 0.9)),
            "q99": float(np.quantile(samples, 0.99)),
            "maximum": float(np.max(samples)),
            "top_one_percent_share": float(np.sum(np.sort(samples)[-count:]) / total)
            if total else 0.0}


def scalar_stationary_objective(h, step, noise_variance):
    if h <= 0 or not 0 < step < 2 / h:
        raise ValueError("unstable scalar step")
    return step * noise_variance / (2 * (2 - step * h))


def scalar_clt_variance(h, gain, noise_variance):
    if h <= 0 or 2 * gain * h <= 1:
        raise ValueError("CLT gain condition fails")
    return gain * gain * noise_variance / (2 * gain * h - 1)


def heavy_ball(H, initial, step, momentum, updates):
    current = np.asarray(initial, dtype=float).copy()
    velocity = np.zeros_like(current)
    points = [current.copy()]
    for _ in range(updates):
        velocity = momentum * velocity + H @ current
        current = current - step * velocity
        points.append(current.copy())
    return np.asarray(points)


def scheduled_momentum(H, initial, base_step, updates):
    current = np.asarray(initial, dtype=float).copy()
    velocity = np.zeros_like(current)
    points = [current.copy()]
    for t in range(updates):
        velocity = t / (t + 2) * velocity + H @ current
        current = current - 2 * base_step / (t + 3) * velocity
        points.append(current.copy())
    return np.asarray(points)


def moving_average_form(H, initial, base_step, updates):
    current = np.asarray(initial, dtype=float).copy()
    z = current.copy()
    points = [current.copy()]
    for t in range(updates):
        z = z - base_step * (H @ current)
        lam_next = (t + 1) / 2
        current = (lam_next * current + z) / (lam_next + 1)
        points.append(current.copy())
    return np.asarray(points)
