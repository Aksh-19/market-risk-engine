"""
Tests for src/risk_engine/var/historical.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.var.historical import historical_var, historical_expected_shortfall


@pytest.fixture
def known_returns() -> pd.DataFrame:
    # 100 evenly spaced "returns" from -0.05 to +0.04 for a single asset --
    # deliberately deterministic so exact percentile values are hand-checkable,
    # rather than relying on random data where we'd have to trust the same
    # code we're testing to tell us if the answer is right.
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.linspace(-0.05, 0.04, 100)
    return pd.DataFrame({"A": values}, index=dates)


def test_historical_var_one_hot_matches_manual_percentile(known_returns):
    w = [1.0]
    var = historical_var(known_returns, w, confidence_levels=(0.01, 0.05))

    expected_1pct = -np.percentile(known_returns["A"], 1)
    expected_5pct = -np.percentile(known_returns["A"], 5)

    assert np.isclose(var[0.01], expected_1pct)
    assert np.isclose(var[0.05], expected_5pct)


def test_var_is_positive_for_typical_loss_distribution(known_returns):
    # With a distribution that includes negative returns, VaR (a loss
    # magnitude) should come out positive under our sign convention.
    w = [1.0]
    var = historical_var(known_returns, w)
    assert var[0.01] > 0
    assert var[0.05] > 0


def test_1pct_var_is_more_severe_than_5pct_var(known_returns):
    # Rarer event = larger loss. 1% VaR must be >= 5% VaR always, since the
    # 1st percentile is further into the tail than the 5th.
    w = [1.0]
    var = historical_var(known_returns, w, confidence_levels=(0.01, 0.05))
    assert var[0.01] >= var[0.05]


def test_es_is_at_least_as_large_as_var(known_returns):
    # ES averages the tail BEYOND the VaR threshold, so it must be >= VaR
    # at the same confidence level -- this is the core invariant that
    # distinguishes ES from VaR and should hold for any input.
    w = [1.0]
    var = historical_var(known_returns, w, confidence_levels=(0.01, 0.05))
    es = historical_expected_shortfall(known_returns, w, confidence_levels=(0.01, 0.05))

    assert es[0.01] >= var[0.01]
    assert es[0.05] >= var[0.05]


def test_es_matches_manual_tail_average(known_returns):
    w = [1.0]
    es = historical_expected_shortfall(known_returns, w, confidence_levels=(0.05,))

    threshold = np.percentile(known_returns["A"], 5)
    expected = -known_returns["A"][known_returns["A"] <= threshold].mean()

    assert np.isclose(es[0.05], expected)


def test_portfolio_weighting_changes_var_from_single_asset():
    # Sanity check that VaR actually reflects diversification: a 2-asset
    # equal-weight portfolio of imperfectly correlated assets should have
    # LOWER VaR than either asset held alone at 100% weight.
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(seed=1)
    a = rng.normal(0, 0.02, 200)
    b = rng.normal(0, 0.02, 200)  # independent -- diversification should show up
    df = pd.DataFrame({"A": a, "B": b}, index=dates)

    var_a_alone = historical_var(df, [1.0, 0.0], confidence_levels=(0.05,))[0.05]
    var_equal_weight = historical_var(df, [0.5, 0.5], confidence_levels=(0.05,))[0.05]

    assert var_equal_weight < var_a_alone


def test_zero_variance_asset_gives_zero_var():
    # Degenerate case: a constant return series has zero variance, so VaR
    # and ES should both be (approximately) zero -- no loss is ever
    # possible if returns never move. Good guard against sign-flip bugs.
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({"A": [0.001] * 50}, index=dates)

    var = historical_var(df, [1.0])
    es = historical_expected_shortfall(df, [1.0])

    assert np.isclose(var[0.05], -0.001, atol=1e-9)
    assert np.isclose(es[0.05], -0.001, atol=1e-9)
