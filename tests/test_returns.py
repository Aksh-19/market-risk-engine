"""
Tests for risk_engine.data.returns

WHY THESE SPECIFIC TEST CASES
-------------------------------
Each test targets a concept from the module docstring, not just "does the
function run." That's the difference between testing for coverage and
testing for correctness of the financial logic:
  - test_log_return_matches_closed_form: sanity-checks the actual formula
    against hand computation, not just "returns something."
  - test_time_additivity: proves the property we RELY ON in Phase 3 (sqrt-of
    -time scaling) actually holds for our implementation.
  - test_alignment_drops_unmatched_dates: proves the inner-join calendar
    logic actually protects against the phantom-correlation bug described
    in the docstring, rather than just trusting the pandas call is correct.
  - test_non_positive_price_raises: proves the guardrail against bad ticks
    is live, not just documented.
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.data.returns import (
    build_returns_matrix,
    compute_log_returns,
    compute_simple_returns,
)


def test_log_return_matches_closed_form():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    result = compute_log_returns(prices)
    expected = pd.Series(
        [np.log(110 / 100), np.log(99 / 110)],
        index=prices.index[1:],
    )
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_simple_return_matches_pct_change():
    prices = pd.Series([100.0, 105.0, 94.5], index=pd.bdate_range("2024-01-01", periods=3))
    result = compute_simple_returns(prices)
    assert result.iloc[0] == pytest.approx(0.05)
    assert result.iloc[1] == pytest.approx(-0.10)


def test_time_additivity_of_log_returns():
    """k-day log return should equal the sum of the k daily log returns."""
    prices = pd.Series(
        [100.0, 102.0, 101.0, 105.0, 103.0],
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    daily = compute_log_returns(prices)
    three_day_return = np.log(prices.iloc[3] / prices.iloc[0])  # cumulative day0 -> day3
    assert daily.iloc[:3].sum() == pytest.approx(three_day_return)


def test_non_positive_price_raises():
    prices = pd.Series([100.0, -5.0, 90.0], index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="Non-positive price"):
        compute_log_returns(prices)


def test_alignment_drops_unmatched_dates():
    """
    Asset A trades Mon/Tue/Wed. Asset B trades Mon/Wed/Thu (e.g. different
    holiday calendars). The aligned returns matrix should only keep dates
    both assets share, proving we don't silently fabricate correlation from
    misaligned rows.
    """
    dates_a = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    dates_b = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"])

    price_a = pd.DataFrame({"Adj Close": [100.0, 101.0, 99.0]}, index=dates_a)
    price_b = pd.DataFrame({"Adj Close": [50.0, 51.0, 52.0]}, index=dates_b)

    matrix = build_returns_matrix({"A": price_a, "B": price_b}, method="simple")

    # Only 2024-01-03 has a valid *return* for both (needs a prior day too),
    # and it must be the same date for both assets, not just same row index.
    assert list(matrix.index) == [pd.Timestamp("2024-01-03")]
    assert matrix.loc[pd.Timestamp("2024-01-03"), "A"] == pytest.approx(99.0 / 101.0 - 1)
    assert matrix.loc[pd.Timestamp("2024-01-03"), "B"] == pytest.approx(51.0 / 50.0 - 1)


def test_missing_price_field_raises_keyerror():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price_a = pd.DataFrame({"Close": [100.0, 101.0, 99.0]}, index=dates)
    with pytest.raises(KeyError, match="Adj Close"):
        build_returns_matrix({"A": price_a})
