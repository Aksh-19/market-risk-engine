"""
Tests for src/risk_engine/var/filtered_historical.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.var.filtered_historical import (
    filtered_historical_var,
    filtered_historical_expected_shortfall,
    _standardized_residuals_and_forecast,
)
from risk_engine.var.historical import historical_var


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    # GARCH fitting needs a reasonably long series to converge sensibly;
    # 250 days across 3 assets, fixed seed for reproducibility.
    rng = np.random.default_rng(seed=17)
    n = 250
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(loc=0.0, scale=0.01, size=(n, 3))
    return pd.DataFrame(data, columns=["X", "Y", "Z"], index=dates)


def test_var_is_positive(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = filtered_historical_var(sample_returns, w)
    assert var[0.01] > 0
    assert var[0.05] > 0


def test_1pct_var_exceeds_5pct_var(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = filtered_historical_var(sample_returns, w)
    assert var[0.01] > var[0.05]


def test_es_exceeds_var_at_same_confidence(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = filtered_historical_var(sample_returns, w)
    es = filtered_historical_expected_shortfall(sample_returns, w)
    assert es[0.01] >= var[0.01]
    assert es[0.05] >= var[0.05]


def test_standardized_residuals_have_roughly_unit_variance(sample_returns):
    # Core theoretical property of GARCH standardization: z_t = r_t/sigma_t
    # should have variance close to 1 if GARCH is doing its job of
    # absorbing time-varying volatility. Not exactly 1 (finite sample,
    # imperfect fit) but should be in a sane neighborhood -- this is the
    # check that would catch a broken standardization (e.g. dividing by
    # the wrong sigma, or a shape mismatch silently broadcasting wrong).
    Z, sigma_next = _standardized_residuals_and_forecast(sample_returns)
    variances = Z.var()
    for col in Z.columns:
        assert 0.5 < variances[col] < 2.0


def test_sigma_next_is_positive_and_correct_shape(sample_returns):
    Z, sigma_next = _standardized_residuals_and_forecast(sample_returns)
    assert sigma_next.shape == (3,)
    assert np.all(sigma_next > 0)


def test_deterministic_no_randomness_involved(sample_returns):
    # Unlike Monte Carlo, FHS has no random simulation step -- it's built
    # entirely from historical data and GARCH's deterministic fit. Same
    # input must give bit-identical output every call, no seed needed.
    w = [1 / 3, 1 / 3, 1 / 3]
    var1 = filtered_historical_var(sample_returns, w)
    var2 = filtered_historical_var(sample_returns, w)
    assert var1[0.01] == var2[0.01]
    assert var1[0.05] == var2[0.05]


def test_differs_from_plain_historical_sim(sample_returns):
    # FHS rescales historical shocks by TODAY's GARCH volatility forecast,
    # while plain Historical Sim uses raw historical returns unscaled --
    # these should generally diverge, confirming FHS is doing real work
    # and not silently collapsing to the same computation.
    w = [1 / 3, 1 / 3, 1 / 3]
    fhs_var = filtered_historical_var(sample_returns, w)
    hist_var = historical_var(sample_returns, w)
    assert not np.isclose(fhs_var[0.01], hist_var[0.01], rtol=1e-6)


def test_one_hot_weights_isolate_single_asset(sample_returns):
    # One-hot weights should reduce FHS to that single asset's own
    # GARCH-standardized-and-rescaled historical shocks -- same
    # single-asset special-case invariant checked throughout this project.
    from risk_engine.var.filtered_historical import _simulate_fhs_portfolio_returns

    w_onehot = [1.0, 0.0, 0.0]
    r_p_sim = _simulate_fhs_portfolio_returns(sample_returns, w_onehot)

    Z, sigma_next = _standardized_residuals_and_forecast(sample_returns)
    expected = Z["X"].to_numpy() * sigma_next[0]

    np.testing.assert_allclose(r_p_sim.to_numpy(), expected, rtol=1e-10)


def test_zero_variance_asset_handled_without_crash():
    # Degenerate case: a near-constant series. GARCH fitting on a
    # zero/near-zero-variance series is a known edge case -- this test
    # just confirms it doesn't crash outright with a shape or div-by-zero
    # error; it's not asserting a specific "correct" numeric answer since
    # GARCH itself isn't well-defined on degenerate data.
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(seed=3)
    df = pd.DataFrame(
        {
            "A": rng.normal(0, 0.01, 100),
            "B": [0.0001] * 100,  # near-constant
        },
        index=dates,
    )
    var = filtered_historical_var(df, [0.5, 0.5])
    assert np.isfinite(var[0.01])
    assert np.isfinite(var[0.05])
