"""
Configuration schema for the data ingestion layer.

WHY THIS FILE EXISTS
---------------------
In a research script, you'd just hardcode tickers and dates at the top of a
notebook. In a production-grade risk engine, the data pull needs to be:
  1. Reproducible   -> anyone (including future-you) can regenerate the exact
                        same dataset from a config file, not from memory of
                        what you typed into a REPL six weeks ago.
  2. Validated       -> a typo'd date range or an empty ticker list should
                        fail LOUDLY at config-parse time, not silently three
                        layers deep inside a VaR calculation with a confusing
                        NaN-propagation error.
  3. Swappable       -> Phase 1 uses Yahoo Finance. Later you might switch to
                        a paid vendor (Bloomberg, Refinitiv, Polygon). If the
                        rest of the codebase only ever talks to `DataConfig`
                        and `DataLoader`, swapping the source touches ONE file.

Pydantic gives us (1) and (2) for free: it's a data-validation library that
raises a clear error the instant you construct an invalid config, instead of
letting bad data silently flow downstream. This is the same pattern FastAPI
uses for request bodies, which is why it was already implicitly in your
dependency tree from Phase 0.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Everything the data layer needs to know to build a dataset."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Asset universe, e.g. ['AAPL', 'MSFT', 'SPY']"
    )
    start_date: date = Field(..., description="Inclusive start of the pull window")
    end_date: date = Field(..., description="Inclusive end of the pull window")
    source: str = Field(default="yfinance", description="Data vendor identifier")
    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Where raw pulls are cached as parquet so we don't hammer the API",
    )
    price_field: str = Field(
        default="Adj Close",
        description="Which price column downstream calcs should key off of. "
        "Adjusted close, not raw close — see loader.py docstring for why.",
    )

    @field_validator("tickers")
    @classmethod
    def _upper_and_dedupe(cls, v: list[str]) -> list[str]:
        # Normalize casing and remove accidental duplicates while preserving
        # order — order matters later for consistent covariance-matrix
        # indexing in the volatility module.
        seen = dict.fromkeys(t.strip().upper() for t in v)
        return list(seen)

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start is not None and v <= start:
            raise ValueError(f"end_date ({v}) must be strictly after start_date ({start})")
        return v

    def cache_path_for(self, ticker: str) -> Path:
        """Deterministic cache filename: one parquet file per ticker per window."""
        return (
            self.cache_dir
            / f"{ticker}_{self.start_date.isoformat()}_{self.end_date.isoformat()}.parquet"
        )
