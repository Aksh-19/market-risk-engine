"""
Data acquisition layer: pulls raw prices and caches them to disk.

WHY ADJUSTED CLOSE, NOT CLOSE
-------------------------------
Raw "Close" price jumps on dividend ex-dates and stock splits in a way that
has nothing to do with the market's view of risk — a $2 dividend payout on a
$150 stock creates a ~1.3% "return" that is pure accounting, not price risk.
If you feed raw closes into a VaR model, you will systematically misestimate
volatility around dividend dates and completely blow up on split dates (a
2-for-1 split looks like a -50% return to a naive return calculation).
"Adjusted Close" backs these mechanical effects out, which is why every
serious risk/quant pipeline uses it as the base series. This is a good
example of a data-layer decision silently determining whether your Phase 3
VaR numbers are trustworthy — get it wrong here and no amount of clever GARCH
fitting downstream will save you.

WHY WE CACHE TO PARQUET
-------------------------
Two reasons, both real engineering constraints you'll hit immediately:
  1. Yahoo Finance rate-limits / throttles repeated calls. Re-running your
     GARCH fit 20 times while debugging shouldn't mean 20 network round-trips
     for the same 5 years of AAPL data.
  2. Parquet is a columnar binary format — reading it back is both faster and
     preserves dtypes exactly (a CSV round-trip can silently turn your
     DatetimeIndex into strings). Every serious quant data pipeline caches
     pulled data locally for this reason.

WHY WE DON'T JUST DOWNLOAD EVERYTHING INTO ONE GIANT DATAFRAME BLINDLY
-------------------------------------------------------------------------
Different tickers can have different listing dates, different holiday
calendars (a US stock and a London-listed stock don't share trading days),
and different data-quality issues. We fetch per-ticker, validate per-ticker,
and only ALIGN into a joint frame at the point where an actual calculation
needs it (see returns.py). This keeps failures isolated and debuggable: if
one ticker's pull fails, it doesn't silently poison the whole universe.
"""

from __future__ import annotations

import logging

import pandas as pd

from risk_engine.data.config import DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Fetches and caches historical prices for a configured asset universe."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_prices(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame of OHLCV data for a single ticker, using the local
        parquet cache when available.

        force_refresh=True bypasses the cache — use this when you specifically
        need fresh data (e.g. today's close just posted) rather than as a
        default, since it defeats the whole point of caching.
        """
        cache_path = self.config.cache_path_for(ticker)

        if cache_path.exists() and not force_refresh:
            logger.info("Loading %s from cache: %s", ticker, cache_path)
            return pd.read_parquet(cache_path)

        logger.info(
            "Fetching %s from %s (%s to %s)",
            ticker,
            self.config.source,
            self.config.start_date,
            self.config.end_date,
        )
        df = self._download(ticker)

        if df.empty:
            raise ValueError(
                f"No data returned for {ticker} in range "
                f"{self.config.start_date}–{self.config.end_date}. "
                "Check the ticker is valid and was listed during this window."
            )

        df.to_parquet(cache_path)
        return df

    def load_universe(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch every ticker in the config. Returns {ticker: price_df}."""
        universe: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}

        for ticker in self.config.tickers:
            try:
                universe[ticker] = self.fetch_prices(ticker, force_refresh=force_refresh)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - we want to isolate & report, not crash the whole batch
                failures[ticker] = str(exc)
                logger.warning("Failed to load %s: %s", ticker, exc)

        if failures:
            logger.warning(
                "Loaded %d/%d tickers. Failures: %s",
                len(universe),
                len(self.config.tickers),
                failures,
            )
        if not universe:
            raise RuntimeError(f"Failed to load any ticker. Details: {failures}")

        return universe

    def _download(self, ticker: str) -> pd.DataFrame:
        """Isolated so tests can monkeypatch this instead of hitting the network."""
        import yfinance as yf  # local import: keeps yfinance optional at import-time

        raw = yf.download(
            ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            auto_adjust=False,  # we want BOTH Close and Adj Close visible, see docstring above
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw
