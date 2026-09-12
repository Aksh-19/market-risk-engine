"""
Tests for risk_engine.volatility.garch

The centerpiece here is test_parameter_recovery_on_simulated_data: we
GENERATE a return series from a GARCH(1,1) process with KNOWN true
parameters, then fit our estimator to it and check the recovered
parameters are close to the truth. This is the gold-standard test for any
MLE-based estimator -- unlike testing against real market data (where we
never know the "true" parameters), simulation gives us ground truth to
check against. If this test passes, the optimizer and likelihood function
are demonstrably doing their job, not just running without crashing.
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.volatility.garch import (
    fit_garch11,
    forecast_variance,
    garch_volatility_series,
)


def simulate_garch11(n, omega, alpha, beta, seed=42):
    """Generate a synthetic return series that truly follows GARCH(1,1)
    with the given parameters, for use as ground truth in tests."""
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = rng.normal(0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return r, sigma2


def test_parameter_recovery_on_simulated_data():
    true_omega, true_alpha, true_beta = 1e-6, 0.08, 0.88
    r, _ = simulate_garch11(3000, true_omega, true_alpha, true_beta, seed=7)
    returns = pd.Series(r, index=pd.bdate_range("2010-01-01", periods=3000))

    fitted = fit_garch11(returns)

    assert fitted.converged
    # MLE on a finite sample won't recover parameters exactly, but should
    # land in a reasonable neighborhood given 3000 observations.
    assert fitted.alpha == pytest.approx(true_alpha, abs=0.05)
    assert fitted.beta == pytest.approx(true_beta, abs=0.08)
    assert fitted.persistence < 1.0


def test_stationarity_constraint_respected():
    """Even on data that might tempt an unconstrained optimizer toward
    alpha+beta >= 1, the fit must respect the stationarity constraint."""
    rng = np.random.default_rng(1)
    # Deliberately volatility-clustered-looking series (crude approximation)
    r = np.concatenate([rng.normal(0, 0.005, 200), rng.normal(0, 0.03, 200)])
    returns = pd.Series(r, index=pd.bdate_range("2020-01-01", periods=400))

    fitted = fit_garch11(returns)
    assert fitted.persistence < 1.0
    assert fitted.omega > 0
    assert fitted.alpha >= 0
    assert fitted.beta >= 0


def test_unconditional_variance_matches_formula():
    true_omega, true_alpha, true_beta = 2e-6, 0.06, 0.90
    r, _ = simulate_garch11(2000, true_omega, true_alpha, true_beta, seed=3)
    returns = pd.Series(r, index=pd.bdate_range("2015-01-01", periods=2000))
    fitted = fit_garch11(returns)

    expected_uncond_var = fitted.omega / (1 - fitted.alpha - fitted.beta)
    assert fitted.unconditional_variance == pytest.approx(expected_uncond_var)


def test_forecast_converges_to_unconditional_variance():
    true_omega, true_alpha, true_beta = 1e-6, 0.08, 0.88
    r, _ = simulate_garch11(2000, true_omega, true_alpha, true_beta, seed=7)
    returns = pd.Series(r, index=pd.bdate_range("2010-01-01", periods=2000))
    fitted = fit_garch11(returns)

    long_horizon_forecast = forecast_variance(fitted, horizon=500)
    # After 500 days (persistence^500 is essentially zero), the forecast
    # should have converged almost exactly to the unconditional variance --
    # this is the mean-reversion property EWMA structurally lacks.
    assert long_horizon_forecast[-1] == pytest.approx(fitted.unconditional_variance, rel=1e-6)

    # And forecasts should move monotonically toward that level from
    # wherever today's variance happens to sit.
    diffs = np.abs(long_horizon_forecast - fitted.unconditional_variance)
    assert np.all(np.diff(diffs) <= 1e-12)  # monotonically non-increasing


def test_garch_volatility_series_is_labeled_and_aligned():
    r, _ = simulate_garch11(500, 1e-6, 0.08, 0.88, seed=5)
    returns = pd.Series(r, index=pd.bdate_range("2022-01-01", periods=500))
    fitted = fit_garch11(returns)
    vol = garch_volatility_series(returns, fitted)

    assert (vol.index == returns.index).all()
    assert (vol > 0).all()
    assert vol.name == "garch_vol"


def test_garch_likelihood_beats_constant_variance_on_clustered_data():
    """
    A meaningful sanity check: on genuinely volatility-clustered data, GARCH
    should find a materially better (higher) log-likelihood than a naive
    constant-variance (i.e. alpha=beta=0) model would achieve on the same
    data -- proving the fitted alpha/beta are doing real work, not just
    landing near zero.
    """
    true_omega, true_alpha, true_beta = 1e-6, 0.10, 0.85
    r, _ = simulate_garch11(2000, true_omega, true_alpha, true_beta, seed=11)
    returns = pd.Series(r, index=pd.bdate_range("2010-01-01", periods=2000))

    fitted = fit_garch11(returns)

    # Constant-variance log-likelihood: alpha=beta=0, sigma^2 = sample variance
    sample_var = np.var(r)
    const_ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sample_var) + r**2 / sample_var)

    assert fitted.log_likelihood > const_ll
