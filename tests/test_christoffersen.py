"""
Tests for src/risk_engine/backtest/christoffersen.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.backtest.christoffersen import christoffersen_independence_test


def test_evenly_spaced_breaches_are_not_flagged_as_clustered():
    # Breaches spread out with no adjacency at all -- pi_11 should be ~0
    # (no breach is ever immediately followed by another), pi_01 should be
    # small and roughly stable. This pattern should NOT be flagged as
    # clustering.
    n = 1000
    breaches = np.zeros(n, dtype=bool)
    breaches[::50] = True  # one breach every 50 days, always isolated
    result = christoffersen_independence_test(pd.Series(breaches), alpha_claimed=0.02)
    assert not result.reject_independence


def test_deliberately_clustered_breaches_are_flagged():
    # Construct a sequence where breaches are ARTIFICIALLY bunched: one
    # long consecutive run of breaches, rest calm. pi_11 (breach given
    # breach) should be very high, pi_01 (breach given calm) very low --
    # exactly the clustering signature this test exists to catch.
    n = 1000
    breaches = np.zeros(n, dtype=bool)
    breaches[500:530] = True  # 30 consecutive breach days in one block
    result = christoffersen_independence_test(pd.Series(breaches), alpha_claimed=0.01)
    assert result.pi_11 > result.pi_01  # clustering signature present
    assert result.reject_independence


def test_pi_11_exceeds_pi_01_in_clustered_case():
    n = 500
    breaches = np.zeros(n, dtype=bool)
    breaches[100:120] = True  # 20-day consecutive block
    result = christoffersen_independence_test(pd.Series(breaches), alpha_claimed=0.02)
    assert result.pi_11 > result.pi_01


def test_transition_counts_sum_to_n_minus_1():
    n = 300
    rng = np.random.default_rng(5)
    breaches = pd.Series(rng.random(n) < 0.05)
    result = christoffersen_independence_test(breaches, alpha_claimed=0.05)
    assert result.n00 + result.n01 + result.n10 + result.n11 == n - 1


def test_no_breaches_at_all_does_not_crash():
    breaches = pd.Series([False] * 200)
    result = christoffersen_independence_test(breaches, alpha_claimed=0.01)
    assert np.isfinite(result.lr_independence)
    assert result.n01 == 0
    assert result.n11 == 0


def test_combined_statistic_equals_sum_of_components():
    from risk_engine.backtest.kupiec import kupiec_pof_test

    breaches = pd.Series([True] * 15 + [False] * 485)
    result = christoffersen_independence_test(breaches, alpha_claimed=0.01)
    kupiec_result = kupiec_pof_test(breaches, alpha_claimed=0.01)

    assert result.lr_combined == pytest.approx(kupiec_result.lr_statistic + result.lr_independence)


def test_combined_pvalue_uses_chi_squared_2_df():
    from scipy.stats import chi2

    breaches = pd.Series([True] * 15 + [False] * 485)
    result = christoffersen_independence_test(breaches, alpha_claimed=0.01)
    expected_p = 1 - chi2.cdf(result.lr_combined, df=2)
    assert result.p_value_combined == pytest.approx(expected_p)


def test_real_backtest_scale_data_produces_sane_result():
    # Sanity check at the actual scale of Phase 4's real data (~2015 days).
    rng = np.random.default_rng(7)
    breaches = pd.Series(rng.random(2015) < 0.0184)
    result = christoffersen_independence_test(breaches, alpha_claimed=0.01)
    assert result.n_obs == 2015
    assert 0 <= result.p_value_independence <= 1
    assert 0 <= result.p_value_combined <= 1
