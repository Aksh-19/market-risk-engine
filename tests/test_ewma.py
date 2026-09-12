"""
Tests for risk_engine.volatility.ewma

Each test targets a specific claim from the derivation:
  - halflife <-> lambda round-trips correctly (checks the formula, not just
    that it runs)
  - the recursion matches a hand-computed value on a tiny example (proves
    the implementation is the formula, not an approximation of it)
  - the covariance matrix's diagonal matches the single-asset scalar EWMA
    exactly (proves the matrix generalization is consistent with the
    scalar special case, not a different unrelated calculation)
  - correlation values are bounded in [-1, 1] and the diagonal is exactly 1
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.volatility.ewma import (
    covariance_to_correlation,
    ewma_covariance_matrix,
    ewma_volatility,
    halflife_to_lambda,
    lambda_to_halflife,
)


def test_halflife_lambda_roundtrip():
    for hl in [5, 11.2, 23, 60]:
        lam = halflife_to_lambda(hl)
        hl_back = lambda_to_halflife(lam)
        assert hl_back == pytest.approx(hl, rel=1e-9)


def test_riskmetrics_lambda_gives_expected_halflife():
    # This is the exact number from the derivation: lambda=0.94 -> ~11.2 days
    hl = lambda_to_halflife(0.94)
    assert hl == pytest.approx(11.2, abs=0.1)


def test_ewma_recursion_matches_hand_computation():
    """
    Tiny 6-point series, min_periods=3, lambda=0.9.
    Seed variance = sample variance of first 3 returns.
    Then manually unroll the recursion for 2 more steps and compare.
    """
    r = np.array([0.01, -0.02, 0.015, 0.03, -0.01, 0.02])
    returns = pd.Series(r, index=pd.bdate_range("2024-01-01", periods=6))
    lam = 0.9
    min_periods = 3

    vol = ewma_volatility(returns, lambda_=lam, min_periods=min_periods)

    seed_var = np.var(r[:3])
    assert vol.iloc[2] ** 2 == pytest.approx(seed_var)

    # sigma_3^2 = lambda*sigma_2^2 + (1-lambda)*r_2^2   (r_2 = r[2] = 0.015, 0-indexed prior day)
    expected_var_3 = lam * seed_var + (1 - lam) * r[2] ** 2
    assert vol.iloc[3] ** 2 == pytest.approx(expected_var_3)

    expected_var_4 = lam * expected_var_3 + (1 - lam) * r[3] ** 2
    assert vol.iloc[4] ** 2 == pytest.approx(expected_var_4)

    # First min_periods-1 entries must be NaN, not fabricated numbers
    assert vol.iloc[: min_periods - 1].isna().all()


def test_annualization_scales_by_sqrt_trading_days():
    r = np.random.default_rng(1).normal(0, 0.01, 50)
    returns = pd.Series(r, index=pd.bdate_range("2024-01-01", periods=50))
    daily = ewma_volatility(returns, min_periods=20, annualize=False)
    annual = ewma_volatility(returns, min_periods=20, annualize=True)
    ratio = (annual / daily).dropna()
    assert ratio.round(6).nunique() == 1  # constant scale factor everywhere
    assert ratio.iloc[0] == pytest.approx(np.sqrt(252))


def test_covariance_diagonal_matches_scalar_ewma():
    """
    The (i, i) diagonal entry of the EWMA covariance matrix at each date
    should exactly equal the single-asset scalar EWMA variance for asset i.
    This proves the matrix version is a genuine generalization, not a
    different, inconsistent calculation.
    """
    rng = np.random.default_rng(7)
    n = 80
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({"A": rng.normal(0, 0.01, n), "B": rng.normal(0, 0.02, n)}, index=dates)
    min_periods = 20
    lam = 0.94

    cov_series = ewma_covariance_matrix(df, lambda_=lam, min_periods=min_periods)
    vol_a = ewma_volatility(df["A"], lambda_=lam, min_periods=min_periods)
    vol_b = ewma_volatility(df["B"], lambda_=lam, min_periods=min_periods)

    # Spot-check a handful of dates across the series
    for date in list(cov_series.keys())[::15]:
        sigma = cov_series[date]
        assert sigma[0, 0] == pytest.approx(vol_a.loc[date] ** 2, rel=1e-9)
        assert sigma[1, 1] == pytest.approx(vol_b.loc[date] ** 2, rel=1e-9)


def test_covariance_to_correlation_bounds_and_diagonal():
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    corr = covariance_to_correlation(cov)
    assert np.allclose(np.diag(corr), 1.0)
    assert (-1 <= corr).all() and (corr <= 1).all()
    # off-diagonal sign should match the covariance's sign
    assert corr[0, 1] > 0


def test_rejects_invalid_lambda():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50))
    with pytest.raises(ValueError, match="lambda_"):
        ewma_volatility(r, lambda_=1.5, min_periods=20)
    with pytest.raises(ValueError, match="lambda_"):
        ewma_volatility(r, lambda_=0.0, min_periods=20)
