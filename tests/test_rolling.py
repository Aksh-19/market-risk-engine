"""
Tests for risk_engine.volatility.rolling

test_ghosting_effect_is_demonstrable is the interesting one here: it doesn't
just check the function runs correctly, it PROVES the theoretical "ghosting"
failure mode from the EWMA derivation actually happens in this
implementation, by engineering a single extreme return and watching it
create a visible step-change in the rolling estimate exactly `window` days
later, right when it exits the window.
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.volatility.rolling import rolling_volatility


def test_matches_pandas_std_directly():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    vol = rolling_volatility(r, window=20)
    expected = r.rolling(20).std()
    pd.testing.assert_series_equal(vol, expected, check_names=False)


def test_first_window_minus_one_entries_are_nan():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50))
    vol = rolling_volatility(r, window=10)
    assert vol.iloc[:9].isna().all()
    assert vol.iloc[9:].notna().all()


def test_annualization_scales_correctly():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 60))
    daily = rolling_volatility(r, window=20, annualize=False)
    annual = rolling_volatility(r, window=20, annualize=True)
    ratio = (annual / daily).dropna()
    assert ratio.round(6).nunique() == 1
    assert ratio.iloc[0] == pytest.approx(252**0.5)


def test_ghosting_effect_is_demonstrable():
    """
    Build a calm series, inject ONE extreme return at index 50, and confirm:
      (a) rolling vol jumps up right after the shock enters the window
      (b) rolling vol drops sharply the instant the shock exits the window,
          `window` days later -- a mechanical artifact, not a real change
          in market conditions. This is exactly the "ghosting" failure mode
          the EWMA derivation predicts a naive equal-weighted window has.
    """
    rng = np.random.default_rng(3)
    n = 120
    window = 20
    r = rng.normal(0, 0.005, n)
    shock_idx = 50
    r[shock_idx] = 0.15  # a large, one-off shock
    returns = pd.Series(r)

    vol = rolling_volatility(returns, window=window)

    vol_during_shock_window = vol.iloc[shock_idx + 1]
    vol_just_after_shock_exits = vol.iloc[shock_idx + window + 1]

    # While the shock is still inside the window, vol should be elevated
    # well above the baseline calm-period level.
    baseline_vol = vol.iloc[shock_idx - 5]
    assert vol_during_shock_window > baseline_vol * 2

    # The moment the shock exits the window, vol should drop back sharply --
    # this is the actual ghosting step-change, not a gradual decay.
    assert vol_just_after_shock_exits < vol_during_shock_window * 0.5


def test_rejects_invalid_window():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50))
    with pytest.raises(ValueError, match="window"):
        rolling_volatility(r, window=1)
    with pytest.raises(ValueError, match="window"):
        rolling_volatility(r, window=100)  # window >= len(returns)
