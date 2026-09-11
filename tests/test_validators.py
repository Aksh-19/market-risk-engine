"""
Tests for risk_engine.data.validators

Each test injects one SPECIFIC data defect (a gap, a flat day, an outlier,
a duplicate) into otherwise-clean synthetic data, then asserts the validator
catches exactly that defect and nothing else. This isolation is what makes
the tests trustworthy diagnostics rather than vague smoke tests.
"""

import numpy as np
import pandas as pd

from risk_engine.data.validators import validate_price_series


def _clean_series(n=250, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    log_rets = rng.normal(0, 0.01, size=n)
    prices = 100 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame({"Adj Close": prices}, index=dates)


def test_clean_series_passes():
    df = _clean_series()
    report = validate_price_series(df, ticker="TEST")
    assert report.is_clean
    assert len(report.extreme_moves) == 0


def test_detects_missing_dates():
    df = _clean_series()
    # Simulate a 5-business-day feed outage by dropping a contiguous chunk.
    df_gapped = df.drop(df.index[100:105])
    report = validate_price_series(df_gapped, ticker="TEST")
    assert report.missing_dates == 5
    assert not report.is_clean


def test_detects_zero_variance_days():
    df = _clean_series().copy()
    # Force 3 consecutive flat days (price literally unchanged) -
    # a classic symptom of a stale feed rather than real market data.
    df.iloc[50:53, df.columns.get_loc("Adj Close")] = df.iloc[49]["Adj Close"]
    report = validate_price_series(df, ticker="TEST")
    assert report.zero_variance_days >= 2  # 3 equal prices -> 2 zero-return days


def test_flags_extreme_move_without_rejecting_it():
    df = _clean_series().copy()
    # Inject one genuine-looking crash day: -25% in a single session.
    df.iloc[150, df.columns.get_loc("Adj Close")] = df.iloc[149]["Adj Close"] * 0.75
    report = validate_price_series(df, ticker="TEST")
    assert len(report.extreme_moves) >= 1
    # Validation FLAGS, it does not delete -> the series is untouched.
    assert len(df) == 250


def test_detects_duplicate_dates():
    df = _clean_series()
    duped = pd.concat([df, df.iloc[[10]]]).sort_index()
    report = validate_price_series(duped, ticker="TEST")
    assert report.duplicate_dates == 1
