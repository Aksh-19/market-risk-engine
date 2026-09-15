"""
Tests for src/risk_engine/var/covariance_selector.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.var.covariance_selector import get_covariance_matrix


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    # Need enough observations for EWMA's min_periods (default 30) and for
    # GARCH to fit without warnings -- 200 days of synthetic-but-plausible
    # returns across 3 assets.
    rng = np.random.default_rng(seed=42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(loc=0.0, scale=0.01, size=(n, 3))
    return pd.DataFrame(data, columns=["X", "Y", "Z"], index=dates)


@pytest.mark.parametrize("method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_returns_correct_shape(sample_returns, method):
    cov = get_covariance_matrix(sample_returns, method=method)
    assert cov.shape == (3, 3)


@pytest.mark.parametrize("method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_returns_symmetric(sample_returns, method):
    cov = get_covariance_matrix(sample_returns, method=method)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)


@pytest.mark.parametrize("method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_returns_positive_semidefinite(sample_returns, method):
    # A valid covariance matrix must have non-negative eigenvalues -- if
    # this ever fails it means Sigma is malformed and Cholesky (Monte Carlo,
    # Phase 3 next step) would crash downstream anyway. Catch it here first.
    cov = get_covariance_matrix(sample_returns, method=method)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10)


@pytest.mark.parametrize("method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_diagonal_is_positive_variance(sample_returns, method):
    cov = get_covariance_matrix(sample_returns, method=method)
    assert np.all(np.diag(cov) > 0)


def test_unknown_method_raises(sample_returns):
    with pytest.raises(ValueError, match="Unknown covariance method"):
        get_covariance_matrix(sample_returns, method="not_a_real_method")


def test_garch_hybrid_diagonal_matches_latest_garch_vol(sample_returns):
    # The hybrid's diagonal should be EXACTLY each asset's own latest GARCH
    # variance -- not some blended or re-derived value. This is the
    # invariant that makes it a legitimate "D_garch @ R_ewma @ D_garch"
    # construction rather than an approximation of one.
    from risk_engine.volatility.garch import fit_garch11

    cov = get_covariance_matrix(sample_returns, method="garch_hybrid")
    for i, col in enumerate(sample_returns.columns):
        fitted = fit_garch11(sample_returns[col])
        expected_var = fitted.conditional_variance[-1]
        assert np.isclose(cov[i, i], expected_var, rtol=1e-6)
