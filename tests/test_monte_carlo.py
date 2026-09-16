"""
Tests for src/risk_engine/var/monte_carlo.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.var.monte_carlo import monte_carlo_var, monte_carlo_expected_shortfall
from risk_engine.var.parametric import parametric_var, parametric_expected_shortfall


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(seed=11)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(loc=0.0, scale=0.01, size=(n, 3))
    return pd.DataFrame(data, columns=["X", "Y", "Z"], index=dates)


@pytest.mark.parametrize("cov_method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_var_is_positive(sample_returns, cov_method):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = monte_carlo_var(sample_returns, w, cov_method=cov_method, n_sims=10_000, seed=1)
    assert var[0.01] > 0
    assert var[0.05] > 0


def test_1pct_var_exceeds_5pct_var(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = monte_carlo_var(sample_returns, w, n_sims=10_000, seed=1)
    assert var[0.01] > var[0.05]


def test_es_exceeds_var_at_same_confidence(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    var = monte_carlo_var(sample_returns, w, n_sims=10_000, seed=1)
    es = monte_carlo_expected_shortfall(sample_returns, w, n_sims=10_000, seed=1)
    assert es[0.01] >= var[0.01]
    assert es[0.05] >= var[0.05]


def test_seed_gives_reproducible_results(sample_returns):
    # Same seed -> identical output. Critical for tests and for anyone
    # trying to reproduce a reported VaR number later.
    w = [1 / 3, 1 / 3, 1 / 3]
    var1 = monte_carlo_var(sample_returns, w, n_sims=5_000, seed=42)
    var2 = monte_carlo_var(sample_returns, w, n_sims=5_000, seed=42)
    assert var1[0.01] == var2[0.01]
    assert var1[0.05] == var2[0.05]


def test_different_seeds_give_different_results(sample_returns):
    # Sanity check that seed is actually doing something -- if two
    # different seeds gave identical output, something's wired wrong
    # (e.g. RNG silently not being used).
    w = [1 / 3, 1 / 3, 1 / 3]
    var1 = monte_carlo_var(sample_returns, w, n_sims=5_000, seed=1)
    var2 = monte_carlo_var(sample_returns, w, n_sims=5_000, seed=2)
    assert var1[0.01] != var2[0.01]


@pytest.mark.parametrize("cov_method", ["ewma", "ledoit_wolf", "garch_hybrid"])
def test_converges_to_parametric_var_as_sims_grow(sample_returns, cov_method):
    """
    THE key validation test for this module: Monte Carlo and Parametric
    make the SAME distributional assumption (multivariate normal, same
    Sigma) -- they should converge as n_sims -> infinity, since Monte Carlo
    is just Parametric's closed-form answer approximated via simulation.
    A large n_sims + fixed seed should land within a small tolerance of
    the closed-form number. If this test fails, either the Cholesky
    simulation or the closed-form formula (or both) has a real bug -- this
    is the check that would catch it, since the two methods act as a
    built-in cross-validation of each other.
    """
    w = [1 / 3, 1 / 3, 1 / 3]
    mc_var = monte_carlo_var(
        sample_returns,
        w,
        confidence_levels=(0.01, 0.05),
        cov_method=cov_method,
        n_sims=200_000,
        seed=99,
    )
    param_var = parametric_var(
        sample_returns, w, confidence_levels=(0.01, 0.05), cov_method=cov_method
    )

    # Simulation noise means exact equality is impossible; 5% relative
    # tolerance is generous enough to be stable across seeds/CI runs while
    # still catching a genuinely broken implementation.
    assert mc_var[0.01] == pytest.approx(param_var[0.01], rel=0.05)
    assert mc_var[0.05] == pytest.approx(param_var[0.05], rel=0.05)


def test_converges_to_parametric_es(sample_returns):
    w = [1 / 3, 1 / 3, 1 / 3]
    mc_es = monte_carlo_expected_shortfall(
        sample_returns, w, confidence_levels=(0.01, 0.05), n_sims=200_000, seed=99
    )
    param_es = parametric_expected_shortfall(sample_returns, w, confidence_levels=(0.01, 0.05))

    assert mc_es[0.01] == pytest.approx(param_es[0.01], rel=0.07)
    assert mc_es[0.05] == pytest.approx(param_es[0.05], rel=0.07)


def test_cholesky_reproduces_target_covariance(sample_returns):
    # Directly check the Cholesky mechanism itself: simulated draws'
    # sample covariance should approximate Sigma as n_sims grows -- this
    # isolates the simulation machinery from the VaR/ES math on top of it.
    from risk_engine.var.covariance_selector import get_covariance_matrix

    cov_matrix = get_covariance_matrix(sample_returns, method="ewma")
    L = np.linalg.cholesky(cov_matrix)

    rng = np.random.default_rng(123)
    n_sims = 200_000
    Z = rng.standard_normal(size=(n_sims, 3))
    draws = Z @ L.T

    sample_cov = np.cov(draws, rowvar=False)
    np.testing.assert_allclose(sample_cov, cov_matrix, atol=5e-6)


def test_one_hot_weights_isolate_single_asset(sample_returns):
    # One-hot weights should give a Monte Carlo VaR close to that asset's
    # own Parametric VaR (same single-asset special-case invariant used
    # throughout this project).
    w = [1.0, 0.0, 0.0]
    mc_var = monte_carlo_var(sample_returns, w, confidence_levels=(0.05,), n_sims=200_000, seed=5)
    param_var = parametric_var(sample_returns, w, confidence_levels=(0.05,))
    assert mc_var[0.05] == pytest.approx(param_var[0.05], rel=0.05)
