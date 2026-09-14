"""
Tests for risk_engine.volatility.covariance

test_ledoit_wolf_matches_sklearn is the centerpiece: it validates our
from-scratch implementation of the Ledoit-Wolf (2004) closed-form shrinkage
formula against sklearn's `ledoit_wolf`, a widely used reference
implementation of the same paper. Matching to near machine precision is
strong evidence the formula was implemented correctly, not just "runs and
produces plausible-looking numbers."
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import empirical_covariance, ledoit_wolf as sklearn_ledoit_wolf

from risk_engine.volatility.covariance import ledoit_wolf_shrinkage, sample_covariance


def _correlated_returns(n=300, p=6, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (p, p))
    true_cov = A @ A.T / p
    X = rng.multivariate_normal(np.zeros(p), true_cov, size=n)
    return pd.DataFrame(X, columns=[f"A{i}" for i in range(p)]), X


def test_sample_covariance_matches_sklearn_empirical_covariance():
    returns, X = _correlated_returns()
    ours = sample_covariance(returns)
    sk = empirical_covariance(X)
    np.testing.assert_allclose(ours, sk, rtol=1e-10)


def test_sample_covariance_is_symmetric_and_positive_semidefinite():
    returns, _ = _correlated_returns()
    S = sample_covariance(returns)
    np.testing.assert_allclose(S, S.T)
    eigenvalues = np.linalg.eigvalsh(S)
    assert (eigenvalues >= -1e-10).all()  # small negative tolerance for float error


def test_ledoit_wolf_matches_sklearn():
    returns, X = _correlated_returns(n=300, p=6, seed=42)
    our_shrunk, our_shrinkage = ledoit_wolf_shrinkage(returns)
    sk_shrunk, sk_shrinkage = sklearn_ledoit_wolf(X)

    assert our_shrinkage == pytest.approx(sk_shrinkage, rel=1e-6)
    np.testing.assert_allclose(our_shrunk, sk_shrunk, rtol=1e-6)


def test_ledoit_wolf_matches_sklearn_across_multiple_datasets():
    """Confirm the match isn't a one-off coincidence on a single seed/shape."""
    for seed, n, p in [(1, 200, 4), (7, 500, 8), (99, 150, 10)]:
        returns, X = _correlated_returns(n=n, p=p, seed=seed)
        our_shrunk, our_shrinkage = ledoit_wolf_shrinkage(returns)
        sk_shrunk, sk_shrinkage = sklearn_ledoit_wolf(X)
        assert our_shrinkage == pytest.approx(sk_shrinkage, rel=1e-5)
        np.testing.assert_allclose(our_shrunk, sk_shrunk, rtol=1e-5)


def test_shrinkage_intensity_bounded_zero_to_one():
    returns, _ = _correlated_returns()
    _, shrinkage = ledoit_wolf_shrinkage(returns)
    assert 0.0 <= shrinkage <= 1.0


def test_shrinkage_pulls_toward_scaled_identity():
    """
    The shrunk covariance's off-diagonal entries should be smaller in
    magnitude than the sample covariance's -- since the identity target has
    NO off-diagonal structure, any shrinkage necessarily pulls off-diagonals
    toward zero. This checks the mechanism does what it claims, not just
    that it agrees with sklearn.
    """
    returns, _ = _correlated_returns(
        n=50, p=8, seed=3
    )  # small n relative to p -> non-trivial shrinkage
    S = sample_covariance(returns)
    shrunk, shrinkage = ledoit_wolf_shrinkage(returns)

    assert shrinkage > 0.01  # ensure this test case actually exercises shrinkage
    off_diag_sample = np.abs(S[np.triu_indices(8, k=1)]).mean()
    off_diag_shrunk = np.abs(shrunk[np.triu_indices(8, k=1)]).mean()
    assert off_diag_shrunk < off_diag_sample
