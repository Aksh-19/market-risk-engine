"""
Tests for src/risk_engine/var/portfolio.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.var.portfolio import (
    validate_weights,
    portfolio_returns,
    portfolio_variance,
    portfolio_volatility,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, 0.00, -0.01],
            "B": [0.02, 0.01, -0.01, 0.03, -0.02],
        },
        index=dates,
    )


def test_validate_weights_accepts_correct_shape():
    w = validate_weights([0.5, 0.5], n_assets=2)
    assert w.shape == (2,)


def test_validate_weights_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        validate_weights([0.5, 0.3, 0.2], n_assets=2)


def test_validate_weights_rejects_non_unit_sum():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_weights([0.5, 0.4], n_assets=2)


def test_validate_weights_allows_short_positions():
    # Weights don't need to be non-negative -- shorts are legitimate as
    # long as they still sum to 1.
    w = validate_weights([1.5, -0.5], n_assets=2)
    assert np.isclose(w.sum(), 1.0)


def test_portfolio_returns_matches_manual_weighted_sum(sample_returns):
    w = [0.5, 0.5]
    r_p = portfolio_returns(sample_returns, w)
    expected = sample_returns["A"] * 0.5 + sample_returns["B"] * 0.5
    pd.testing.assert_series_equal(r_p, expected, check_names=False)


def test_portfolio_returns_one_hot_matches_single_asset(sample_returns):
    # One-hot weight vector should reproduce that asset's return series
    # exactly -- this is the "single-asset VaR is a special case" invariant
    # the whole project design leans on.
    w = [1.0, 0.0]
    r_p = portfolio_returns(sample_returns, w)
    pd.testing.assert_series_equal(r_p, sample_returns["A"], check_names=False)


def test_portfolio_returns_demean_subtracts_mean(sample_returns):
    w = [0.5, 0.5]
    r_p = portfolio_returns(sample_returns, w, demean=True)
    assert np.isclose(r_p.mean(), 0.0, atol=1e-12)


def test_portfolio_returns_no_demean_by_default(sample_returns):
    w = [0.5, 0.5]
    r_p = portfolio_returns(sample_returns, w)
    manual_mean = (sample_returns["A"] * 0.5 + sample_returns["B"] * 0.5).mean()
    assert np.isclose(r_p.mean(), manual_mean)


def test_portfolio_variance_matches_quadratic_form():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    expected = w @ cov @ w
    assert np.isclose(portfolio_variance(w, cov), expected)


def test_portfolio_variance_one_hot_matches_diagonal():
    # One-hot weights should isolate exactly that asset's own variance --
    # same invariant as the returns test above, at the covariance level.
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert np.isclose(portfolio_variance([1.0, 0.0], cov), 0.04)
    assert np.isclose(portfolio_variance([0.0, 1.0], cov), 0.09)


def test_portfolio_volatility_is_sqrt_of_variance():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    var = portfolio_variance(w, cov)
    vol = portfolio_volatility(w, cov)
    assert np.isclose(vol, np.sqrt(var))


def test_portfolio_variance_rejects_mismatched_weight_shape():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="shape"):
        portfolio_variance([0.5, 0.5], cov)
