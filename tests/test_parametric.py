"""
Tests for src/risk_engine/var/parametric.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_engine.var.parametric import parametric_var, parametric_expected_shortfall


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    # Enough observations for EWMA's min_periods (default 30) and a stable
    # GARCH fit -- 200 days across 3 assets, fixed seed for reproducibility.
    rng = np.random.default_rng(seed=7)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(loc=0.0, scale=0.01, size=(n, 3))
    return pd.DataFrame(data, columns=["X", "Y", "Z"], index=dates)


@pytest.mark.parametrize("cov_method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_var_is_positive(sample_returns, cov_method):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = parametric_var(sample_returns, w, cov_method=cov_method)
    assert var[0.01] > 0
    assert var[0.05] > 0


@pytest.mark.parametrize("cov_method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_1pct_var_exceeds_5pct_var(sample_returns, cov_method):
    # z_0.01 (2.326) > z_0.05 (1.645) in magnitude, so 1% VaR must always
    # be larger -- this holds by construction under the normal formula,
    # unlike Historical Sim where it's an empirical fact about the data.
    w = [1 / 3, 1 / 3, 1 / 3]
    var = parametric_var(sample_returns, w, cov_method=cov_method)
    assert var[0.01] > var[0.05]


def test_var_matches_manual_zscore_formula(sample_returns):
    from risk_engine.var.portfolio import portfolio_volatility
    from risk_engine.var.covariance_selector import get_covariance_matrix

    w = [1 / 3, 1 / 3, 1 / 3]
    cov = get_covariance_matrix(sample_returns, method="ewma")
    sigma_p = portfolio_volatility(w, cov)

    var = parametric_var(sample_returns, w, confidence_levels=(0.01,), cov_method="ewma")
    expected = -norm.ppf(0.01) * sigma_p

    assert np.isclose(var[0.01], expected)


def test_es_exceeds_var_at_same_confidence(sample_returns):
    # Same invariant as Historical Sim: ES must be >= VaR, since ES
    # averages the tail beyond the VaR threshold (here, the theoretical
    # normal tail rather than an empirical one).
    w = [1 / 3, 1 / 3, 1 / 3]
    var = parametric_var(sample_returns, w, confidence_levels=(0.01, 0.05))
    es = parametric_expected_shortfall(sample_returns, w, confidence_levels=(0.01, 0.05))

    assert es[0.01] >= var[0.01]
    assert es[0.05] >= var[0.05]


def test_es_matches_manual_normal_tail_formula(sample_returns):
    from risk_engine.var.portfolio import portfolio_volatility
    from risk_engine.var.covariance_selector import get_covariance_matrix

    w = [1 / 3, 1 / 3, 1 / 3]
    cov = get_covariance_matrix(sample_returns, method="ledoit_wolf")
    sigma_p = portfolio_volatility(w, cov)

    es = parametric_expected_shortfall(
        sample_returns, w, confidence_levels=(0.05,), cov_method="ledoit_wolf"
    )
    z = norm.ppf(0.05)
    expected = sigma_p * norm.pdf(z) / 0.05

    assert np.isclose(es[0.05], expected)


def test_one_hot_weights_isolate_single_asset_variance(sample_returns):
    # One-hot weight vector should give VaR driven purely by that asset's
    # own variance (diagonal of Sigma) -- same "single-asset is a special
    # case" invariant checked throughout portfolio.py and historical.py.
    from risk_engine.var.covariance_selector import get_covariance_matrix

    w = [1.0, 0.0, 0.0]
    cov = get_covariance_matrix(sample_returns, method="ewma")
    expected_sigma_p = np.sqrt(cov[0, 0])

    var = parametric_var(sample_returns, w, confidence_levels=(0.01,), cov_method="ewma")
    expected_var = -norm.ppf(0.01) * expected_sigma_p

    assert np.isclose(var[0.01], expected_var)


def test_different_cov_methods_give_different_var(sample_returns):
    # Not a strict mathematical guarantee for all inputs, but on real or
    # realistic synthetic data EWMA/Ledoit-Wolf/GARCH-hybrid should not
    # coincidentally produce identical Sigma -- if they do, something's
    # likely wired wrong upstream (e.g. covariance_selector silently
    # returning the same matrix regardless of method).
    w = [1 / 3, 1 / 3, 1 / 3]
    var_ewma = parametric_var(sample_returns, w, cov_method="ewma")
    var_lw = parametric_var(sample_returns, w, cov_method="ledoit_wolf")
    var_garch = parametric_var(sample_returns, w, cov_method="garch_hybrid")

    assert not np.isclose(var_ewma[0.01], var_lw[0.01])
    assert not np.isclose(var_ewma[0.01], var_garch[0.01])


def test_invalid_cov_method_raises(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    with pytest.raises(ValueError, match="Unknown covariance method"):
        parametric_var(sample_returns, w, cov_method="bogus")
