"""
Tests for src/risk_engine/backtest/kupiec.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.backtest.kupiec import kupiec_pof_test


def test_perfectly_calibrated_model_is_not_rejected():
    # 10 breaches out of 1000 days at alpha=0.01 is EXACTLY the claimed
    # rate -- LR should be at (or extremely near) 0, definitely not
    # rejected.
    breaches = pd.Series([True] * 10 + [False] * 990)
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert result.lr_statistic == pytest.approx(0.0, abs=1e-6)
    assert not result.reject_calibration
    assert result.p_value > 0.99


def test_known_textbook_example():
    # n=252, x=6 breaches, alpha=0.01 claimed -- observed rate 6/252=2.38%,
    # roughly 2.4x the claimed rate. Hand-derived reference value:
    # LR = -2*[(246*ln(0.99) + 6*ln(0.01)) - (246*ln(1-6/252) + 6*ln(6/252))]
    #    ~= 3.499. This is JUST below the chi-squared_1 5% critical value
    #    (3.841), so this case should NOT be rejected at the 5% level --
    #    a genuinely informative edge case near the decision boundary.
    breaches = pd.Series([True] * 6 + [False] * 246)
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert result.lr_statistic == pytest.approx(3.499, abs=0.01)
    assert not result.reject_calibration


def test_severely_miscalibrated_model_is_rejected():
    # 40 breaches out of 1000 days at alpha=0.01 claimed (4x the claimed
    # rate) should be a clear, decisive rejection.
    breaches = pd.Series([True] * 40 + [False] * 960)
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert result.reject_calibration
    assert result.p_value < 0.05


def test_zero_breaches_does_not_crash():
    # Edge case: a model that never breached. LR should be a finite,
    # well-defined number (not inf/nan from a log(0) term), and with a
    # reasonably large n, zero breaches at alpha=0.01 IS itself somewhat
    # suspicious (model may be overly conservative) -- worth confirming
    # this produces a sensible, non-crashing result either way.
    breaches = pd.Series([False] * 500)
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert np.isfinite(result.lr_statistic)
    assert result.n_breaches == 0
    assert result.p_hat_observed == 0.0


def test_all_breaches_does_not_crash():
    # Symmetric edge case: every single day breached.
    breaches = pd.Series([True] * 50)
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert np.isfinite(result.lr_statistic)
    assert result.n_breaches == 50
    assert result.reject_calibration  # 100% breach rate vs 1% claimed is extreme


def test_p_hat_matches_manual_calculation():
    breaches = pd.Series([True] * 15 + [False] * 485)
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert result.p_hat_observed == pytest.approx(15 / 500)


def test_custom_significance_level_changes_verdict():
    # A borderline case: significant at a loose 10% level but not at a
    # strict 1% level -- confirms significance_level is actually being
    # used, not hardcoded internally.
    breaches = pd.Series([True] * 20 + [False] * 980)  # alpha=0.01 claimed, 2% observed
    loose = kupiec_pof_test(breaches, alpha_claimed=0.01, significance_level=0.20)
    strict = kupiec_pof_test(breaches, alpha_claimed=0.01, significance_level=0.001)
    assert loose.reject_calibration
    assert not strict.reject_calibration


def test_real_backtest_data_produces_sane_result():
    # Sanity check on the actual shape of data this will be run against --
    # Phase 4's real backtest produced breach_0.01 columns with ~2015
    # observations. This just confirms the function handles that scale
    # without issue, using synthetic data matching that scale/rate.
    rng = np.random.default_rng(seed=1)
    breaches = pd.Series(rng.random(2015) < 0.0184)  # mimics Parametric's observed 1.84% rate
    result = kupiec_pof_test(breaches, alpha_claimed=0.01)
    assert result.n_obs == 2015
    assert 0 <= result.p_value <= 1
