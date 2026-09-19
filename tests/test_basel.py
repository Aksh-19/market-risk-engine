"""
Tests for src/risk_engine/backtest/basel.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.backtest.basel import (
    classify_breach_count,
    basel_traffic_light,
    summarize_zones,
    GREEN_MAX,
    YELLOW_MAX,
)


def test_zero_breaches_is_green():
    assert classify_breach_count(0) == "Green"


def test_green_upper_boundary():
    # Exactly 4 breaches must still be Green -- the classic off-by-one
    # risk in threshold code is testing <5 vs <=4; this pins the exact
    # boundary behavior explicitly.
    assert classify_breach_count(GREEN_MAX) == "Green"


def test_yellow_lower_boundary():
    # Exactly 5 breaches must be Yellow, not Green.
    assert classify_breach_count(GREEN_MAX + 1) == "Yellow"


def test_yellow_upper_boundary():
    # Exactly 9 breaches must still be Yellow, not Red.
    assert classify_breach_count(YELLOW_MAX) == "Yellow"


def test_red_lower_boundary():
    # Exactly 10 breaches must be Red.
    assert classify_breach_count(YELLOW_MAX + 1) == "Red"


def test_large_breach_count_is_red():
    assert classify_breach_count(50) == "Red"


def test_traffic_light_produces_correct_number_of_windows():
    # n days, window=250 -> n - 250 + 1 rolling windows.
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    breaches = pd.Series([False] * n, index=dates)
    result = basel_traffic_light(breaches, window=250)
    assert len(result) == n - 250 + 1


def test_all_calm_series_is_always_green():
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    breaches = pd.Series([False] * n, index=dates)
    result = basel_traffic_light(breaches, window=250)
    assert (result["zone"] == "Green").all()
    assert (result["n_breaches"] == 0).all()


def test_window_with_exactly_5_breaches_is_flagged_yellow():
    # Construct a series where exactly one 250-day window contains
    # precisely 5 breaches, rest calm -- confirms the ROLLING count
    # (not just classify_breach_count in isolation) correctly triggers
    # Yellow at the real boundary.
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    breaches = np.zeros(n, dtype=bool)
    breaches[100:105] = True  # 5 breaches, all within any window covering [100,104]
    result = basel_traffic_light(pd.Series(breaches, index=dates), window=250)

    window_covering_breaches = result[(result.index >= 0) & (result["n_breaches"] == 5)]
    assert len(window_covering_breaches) > 0
    assert (window_covering_breaches["zone"] == "Yellow").all()


def test_raises_if_fewer_observations_than_window():
    breaches = pd.Series([False] * 100)
    with pytest.raises(ValueError, match="Need at least"):
        basel_traffic_light(breaches, window=250)


def test_summarize_zones_fractions_sum_to_one():
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    breaches = pd.Series(rng.random(n) < 0.02, index=dates)
    zone_df = basel_traffic_light(breaches, window=250)
    summary = summarize_zones(zone_df)
    assert sum(summary.values()) == pytest.approx(1.0)
    assert set(summary.keys()) == {"Green", "Yellow", "Red"}


def test_summarize_zones_all_green_case():
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    breaches = pd.Series([False] * n, index=dates)
    zone_df = basel_traffic_light(breaches, window=250)
    summary = summarize_zones(zone_df)
    assert summary["Green"] == pytest.approx(1.0)
    assert summary["Yellow"] == pytest.approx(0.0)
    assert summary["Red"] == pytest.approx(0.0)


def test_real_backtest_scale_does_not_crash():
    # Sanity check at Phase 4's actual scale (~2015 out-of-sample days).
    rng = np.random.default_rng(9)
    dates = pd.date_range("2020-01-01", periods=2015, freq="B")
    breaches = pd.Series(rng.random(2015) < 0.0184, index=dates)
    zone_df = basel_traffic_light(breaches, window=250)
    assert len(zone_df) == 2015 - 250 + 1
    summary = summarize_zones(zone_df)
    assert sum(summary.values()) == pytest.approx(1.0)
