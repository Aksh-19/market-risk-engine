"""
Tests for risk_engine.storage.db.RiskDatabase

Each test checks a round-trip: write something in, read it back, and
confirm it matches -- plus the upsert behavior that matters once this
pipeline gets re-run repeatedly (re-fitting shouldn't accumulate duplicate
or stale rows).
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.storage.db import RiskDatabase


@pytest.fixture
def db(tmp_path):
    database = RiskDatabase(tmp_path / "test.db")
    yield database
    database.close()


def test_returns_roundtrip(db):
    dates = pd.bdate_range("2024-01-02", periods=5)
    returns = pd.DataFrame(
        {"SPY": [0.01, -0.02, 0.005, 0.0, 0.012], "AAPL": [0.02, -0.01, 0.03, -0.005, 0.01]},
        index=dates,
    )
    returns.index.name = "date"

    db.write_returns(returns)
    result = db.read_returns()

    assert list(result.columns) == ["AAPL", "SPY"]  # pivot sorts columns alphabetically
    assert len(result) == 5
    assert result.loc[pd.Timestamp("2024-01-02"), "SPY"] == pytest.approx(0.01)
    assert result.loc[pd.Timestamp("2024-01-03"), "AAPL"] == pytest.approx(-0.01)


def test_returns_filter_by_ticker_and_date(db):
    dates = pd.bdate_range("2024-01-02", periods=5)
    returns = pd.DataFrame(
        {"SPY": range(5), "AAPL": range(10, 15), "GLD": range(20, 25)}, index=dates, dtype=float
    )
    returns.index.name = "date"
    db.write_returns(returns)

    result = db.read_returns(tickers=["SPY", "GLD"], start="2024-01-03", end="2024-01-04")
    assert set(result.columns) == {"SPY", "GLD"}
    assert len(result) == 2


def test_volatility_roundtrip_and_nan_dropped(db):
    dates = pd.bdate_range("2024-01-02", periods=5)
    vol = pd.Series([np.nan, np.nan, 0.15, 0.16, 0.14], index=dates)

    db.write_volatility(vol, ticker="SPY", model="ewma")
    result = db.read_volatility(ticker="SPY", model="ewma")

    assert len(result) == 3  # the two leading NaNs should not be persisted
    assert result.iloc[0] == pytest.approx(0.15)


def test_volatility_rewrite_replaces_not_duplicates(db):
    dates = pd.bdate_range("2024-01-02", periods=3)
    vol_v1 = pd.Series([0.10, 0.11, 0.12], index=dates)
    vol_v2 = pd.Series([0.20, 0.21, 0.22], index=dates)  # simulates a re-fit with new numbers

    db.write_volatility(vol_v1, ticker="SPY", model="ewma")
    db.write_volatility(vol_v2, ticker="SPY", model="ewma")
    result = db.read_volatility(ticker="SPY", model="ewma")

    assert len(result) == 3  # not 6 -- old rows must be replaced, not appended
    assert result.iloc[0] == pytest.approx(0.20)


def test_multiple_models_coexist_per_ticker(db):
    dates = pd.bdate_range("2024-01-02", periods=3)
    db.write_volatility(pd.Series([0.10, 0.11, 0.12], index=dates), ticker="SPY", model="ewma")
    db.write_volatility(pd.Series([0.13, 0.14, 0.15], index=dates), ticker="SPY", model="garch")

    ewma_result = db.read_volatility(ticker="SPY", model="ewma")
    garch_result = db.read_volatility(ticker="SPY", model="garch")

    assert ewma_result.iloc[0] == pytest.approx(0.10)
    assert garch_result.iloc[0] == pytest.approx(0.13)


def test_covariance_snapshot_roundtrip_and_symmetry(db):
    tickers = ["SPY", "AAPL", "GLD"]
    cov = np.array([[0.04, 0.01, 0.002], [0.01, 0.09, 0.001], [0.002, 0.001, 0.02]])
    date = pd.Timestamp("2024-06-15")

    db.write_covariance_snapshot(date, cov, tickers, model="ewma")
    result = db.read_covariance_snapshot(date, tickers, model="ewma")

    np.testing.assert_allclose(result, cov)
    # explicitly confirm the reconstruction is symmetric, not just correct
    # on the diagonal/upper triangle
    np.testing.assert_allclose(result, result.T)


def test_covariance_missing_snapshot_raises(db):
    with pytest.raises(KeyError):
        db.read_covariance_snapshot(pd.Timestamp("2099-01-01"), ["SPY", "AAPL"], model="ewma")


def test_garch_params_roundtrip_and_history(db):
    db.write_garch_params(
        "SPY",
        fit_date="2024-01-01",
        omega=1e-6,
        alpha=0.08,
        beta=0.90,
        log_likelihood=8000.0,
        converged=True,
    )
    db.write_garch_params(
        "SPY",
        fit_date="2024-06-01",
        omega=1.2e-6,
        alpha=0.09,
        beta=0.89,
        log_likelihood=8100.0,
        converged=True,
    )
    history = db.read_garch_params("SPY")

    assert len(history) == 2  # two distinct fit dates should both be kept
    assert history.iloc[0]["alpha"] == pytest.approx(0.08)
    assert history.iloc[1]["alpha"] == pytest.approx(0.09)
    assert bool(history.iloc[0]["converged"]) is True
