"""Numerical checks of invariants used in the explanations, not theorem proofs."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs"))
from core import (gaussian_test_mse, heavy_ball, moving_average_form, quadratic_gaps,
                  quadratic_gd, ridge, scalar_clt_variance,
                  realized_isotropic_test_mse, risk_diagnostics,
                  scalar_stationary_objective, scheduled_momentum)


class MathematicalChecks(unittest.TestCase):
    def test_ridge_solves_normal_equation_with_n_scaling(self):
        rng = np.random.default_rng(4)
        X, y, penalty = rng.normal(size=(9, 13)), rng.normal(size=9), 0.37
        theta = ridge(X, y, penalty)
        residual = X.T @ (X @ theta-y)/len(y) + penalty*theta
        np.testing.assert_allclose(residual, 0, atol=2e-14)

    def test_ridge_training_residual_monotonicity(self):
        rng = np.random.default_rng(6)
        X, y = rng.normal(size=(20, 30)), rng.normal(size=20)
        losses = [np.linalg.norm(X @ ridge(X, y, p)-y)**2
                  for p in [0, 0.001, 0.1, 1, 10]]
        self.assertTrue(np.all(np.diff(losses) >= -1e-12))

    def test_minimum_norm_and_null_space(self):
        X, y = np.array([[1., 1.]]), np.array([2.])
        minimum = ridge(X, y, 0)
        np.testing.assert_allclose(minimum, [1, 1])
        self.assertGreater(np.linalg.norm(minimum + [3, -3]), np.linalg.norm(minimum))
        theta = np.array([3., -1.])
        for _ in range(10):
            theta -= 0.25 * X.T @ (X @ theta-y)
        np.testing.assert_allclose(theta, [3, -1])

    def test_quadratic_exact_spectral_rate(self):
        H, initial, step = np.diag([1., 9.]), np.array([2., -1.]), 0.2
        points = quadratic_gd(H, initial, step, 30)
        expected = initial * (1-step*np.diag(H))[None, :]**np.arange(31)[:, None]
        np.testing.assert_allclose(points, expected, rtol=1e-13, atol=1e-14)
        gaps = quadratic_gaps(H, points)
        np.testing.assert_allclose(gaps, gaps[0]*0.8**(2*np.arange(31)), rtol=1e-12)

    def test_stability_boundary_is_strict(self):
        H, x = np.array([[3.]]), np.array([1.])
        boundary = quadratic_gd(H, x, 2/3, 20)
        np.testing.assert_allclose(np.abs(boundary), 1)
        unstable = quadratic_gd(H, x, 0.8, 20)
        self.assertGreater(abs(unstable[-1, 0]), 100)

    def test_double_descent_threshold_and_feature_criterion(self):
        truth = np.array([2., 1., 1., 0.])
        # At d=3, omitted energy is zero, effective variance 8; adding zero signal hurts.
        self.assertAlmostEqual(gaussian_test_mse(12, 3, truth, 8), 11)
        self.assertGreater(gaussian_test_mse(12, 4, truth, 8), 11)
        for d in [11, 12, 13]:
            self.assertTrue(np.isinf(gaussian_test_mse(12, d, truth, 1)))
        with self.assertRaises(ValueError):
            gaussian_test_mse(12, 12, truth, 0)

    def test_scalar_noise_floor_satisfies_stationary_equation(self):
        h, eta, s2 = 2., 0.15, 1.
        objective = scalar_stationary_objective(h, eta, s2)
        u = 2*objective/h
        self.assertAlmostEqual(u, (1-eta*h)**2*u + eta**2*s2)

    def test_realized_risk_matches_an_exact_isotropic_test_population(self):
        # The uniform distribution on +/- sqrt(3) e_j has covariance I_3.
        # Add independent +/- 0.5 label noise: variance 0.25.
        truth, fit = np.array([2., -1., 3.]), np.array([1., 0.5])
        test_x = np.vstack([np.sqrt(3) * np.eye(3), -np.sqrt(3) * np.eye(3)])
        errors = np.concatenate([test_x @ truth - test_x[:, :2] @ fit + noise
                                 for noise in [-0.5, 0.5]])
        self.assertAlmostEqual(realized_isotropic_test_mse(fit, truth, 0.25),
                               float(np.mean(errors ** 2)))

    def test_tail_diagnostics_keep_one_extreme_draw_visible(self):
        risks = np.r_[np.ones(99), 10000.]
        summary = risk_diagnostics(risks)
        self.assertAlmostEqual(summary["running_mean"][98], 1)
        self.assertAlmostEqual(summary["mean"], 100.99)
        self.assertAlmostEqual(summary["median"], 1)
        self.assertAlmostEqual(summary["maximum"], 10000)
        self.assertAlmostEqual(summary["top_one_percent_share"], 10000 / 10099)
        for invalid in [[], [np.inf], [-1]]:
            with self.assertRaises(ValueError):
                risk_diagnostics(invalid)

    def test_scalar_clt_lyapunov_and_optimal_gain(self):
        h, s2 = 2., 3.
        for a in [0.3, 0.5, 1.0]:
            v = scalar_clt_variance(h, a, s2)
            self.assertAlmostEqual(2*(a*h-0.5)*v, a*a*s2)
        best = scalar_clt_variance(h, 1/h, s2)
        self.assertAlmostEqual(best, s2/h**2)
        self.assertLess(best, scalar_clt_variance(h, 0.3, s2))
        with self.assertRaises(ValueError):
            scalar_clt_variance(h, 0.25, s2)

    def test_heavy_ball_matches_second_order_recurrence(self):
        H, x, eta, rho = np.diag([1., 9.]), np.array([1., 2.]), 0.25, 0.25
        points = heavy_ball(H, x, eta, rho, 20)
        for t in range(1, 20):
            np.testing.assert_allclose(points[t+1], points[t]-eta*H@points[t]
                                       +rho*(points[t]-points[t-1]), atol=1e-14)

    def test_stochastic_momentum_reparameterization(self):
        H, x, eta = np.diag([1., 4.]), np.array([2., -1.]), 0.03
        np.testing.assert_allclose(scheduled_momentum(H, x, eta, 35),
                                   moving_average_form(H, x, eta, 35), atol=1e-14)

    def test_adam_constant_gradient_correction(self):
        g, m, v, b1, b2 = np.array([2., -4.]), np.zeros(2), np.zeros(2), 0.9, 0.999
        for t in range(1, 21):
            m, v = b1*m+(1-b1)*g, b2*v+(1-b2)*g*g
            np.testing.assert_allclose(m/(1-b1**t), g, rtol=1e-13)
            np.testing.assert_allclose(v/(1-b2**t), g*g, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
