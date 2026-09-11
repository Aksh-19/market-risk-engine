"""
Tests for risk_engine.data.loader

WHY WE MONKEYPATCH `_download` INSTEAD OF HITTING THE NETWORK
------------------------------------------------------------------
Unit tests that depend on a live network call are flaky by construction:
they fail when Yahoo Finance is slow, rate-limits you, changes its schema,
or when you're on a plane. A unit test should verify YOUR code's logic
(does the cache get written? does a second call skip the network entirely?),
not whether a third-party API is up right now. That's why `_download` is
factored out as its own method in loader.py — it gives us a single seam to
monkeypatch. (An integration test that DOES hit the real API is valuable
too, but it belongs in a separate, explicitly-marked slow test, not in the
suite that runs on every commit.)
"""

from datetime import date

import pandas as pd
import pytest

from risk_engine.data.config import DataConfig
from risk_engine.data.loader import DataLoader


def _fake_ohlcv(n=5, start_price=100.0):
    dates = pd.bdate_range("2024-01-02", periods=n)
    prices = [start_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Adj Close": prices,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


def test_cache_miss_then_hit(tmp_path, monkeypatch):
    config = DataConfig(
        tickers=["FAKE"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        cache_dir=tmp_path,
    )
    loader = DataLoader(config)

    call_count = {"n": 0}

    def fake_download(self, ticker):
        call_count["n"] += 1
        return _fake_ohlcv()

    monkeypatch.setattr(DataLoader, "_download", fake_download)

    # First call: cache miss -> hits our fake "network"
    df1 = loader.fetch_prices("FAKE")
    assert call_count["n"] == 1
    assert config.cache_path_for("FAKE").exists()

    # Second call: cache hit -> must NOT call the network again
    df2 = loader.fetch_prices("FAKE")
    assert call_count["n"] == 1  # unchanged
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)


def test_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    config = DataConfig(
        tickers=["FAKE"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        cache_dir=tmp_path,
    )
    loader = DataLoader(config)
    call_count = {"n": 0}

    def fake_download(self, ticker):
        call_count["n"] += 1
        return _fake_ohlcv(start_price=100.0 + call_count["n"])

    monkeypatch.setattr(DataLoader, "_download", fake_download)

    loader.fetch_prices("FAKE")
    loader.fetch_prices("FAKE", force_refresh=True)
    assert call_count["n"] == 2


def test_load_universe_isolates_failures(tmp_path, monkeypatch):
    """One bad ticker shouldn't take down the whole universe load."""
    config = DataConfig(
        tickers=["GOOD", "BAD"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        cache_dir=tmp_path,
    )
    loader = DataLoader(config)

    def fake_download(self, ticker):
        if ticker == "BAD":
            return pd.DataFrame()  # simulates an invalid/delisted ticker
        return _fake_ohlcv()

    monkeypatch.setattr(DataLoader, "_download", fake_download)

    universe = loader.load_universe()
    assert "GOOD" in universe
    assert "BAD" not in universe


def test_empty_ticker_list_rejected_at_config_time():
    with pytest.raises(Exception):  # pydantic.ValidationError
        DataConfig(tickers=[], start_date=date(2024, 1, 1), end_date=date(2024, 1, 10))


def test_end_before_start_rejected_at_config_time():
    with pytest.raises(Exception):
        DataConfig(tickers=["AAPL"], start_date=date(2024, 6, 1), end_date=date(2024, 1, 1))
