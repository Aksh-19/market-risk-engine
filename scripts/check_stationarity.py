"""
Stage A diagnostic: run the ADF stationarity test on real price levels vs.
log returns for every asset in the universe.

    python scripts/check_stationarity.py

This is the empirical verification of the assumption this entire project
rests on: prices should test as non-stationary (a "random walk" with no
fixed level to revert to), while log returns should test as stationary
(justifying every volatility/VaR model built on top of them).

We need BOTH price levels and returns for the same tickers, so this script
re-derives price levels from data/cache/*.parquet (Phase 1's raw cache)
rather than only reading the already-computed returns_matrix.parquet.
"""

from __future__ import annotations

import logging

import pandas as pd
import yaml

from risk_engine.data import DataConfig
from risk_engine.diagnostics import stationarity_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str, returns_path: str) -> None:
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    config = DataConfig(**raw_cfg)

    returns = pd.read_parquet(returns_path)

    for ticker in returns.columns:
        cache_path = config.cache_path_for(ticker)
        prices = pd.read_parquet(cache_path)[config.price_field]

        report = stationarity_report(prices, returns[ticker], ticker=ticker)
        print(report)
        print()


if __name__ == "__main__":
    main(
        config_path="configs/universe.yaml",
        returns_path="data/processed/returns_matrix.parquet",
    )
