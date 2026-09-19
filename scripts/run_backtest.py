"""
Orchestrator: run the full out-of-sample rolling backtest for all 4 VaR
methods, save results to parquet (since this is the slow, expensive step --
Kupiec/Christoffersen/Basel, added next, will read these cached results
rather than re-running the backtest every time).

Usage: python3 scripts/run_backtest.py
"""

from __future__ import annotations

import time

import pandas as pd

from risk_engine.backtest import (
    backtest_historical,
    backtest_parametric,
    backtest_monte_carlo,
    backtest_filtered_historical,
)

RETURNS_PATH = "data/processed/returns_matrix.parquet"
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
WINDOW = 500
OUTPUT_DIR = "data/processed/backtests"


def main() -> None:
    returns = pd.read_parquet(RETURNS_PATH)
    print(f"Full history: {len(returns)} days, backtest window={WINDOW}")
    print(f"Out-of-sample days: {len(returns) - WINDOW}\n")

    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = {
        "historical": lambda: backtest_historical(returns, WEIGHTS, window=WINDOW),
        "parametric": lambda: backtest_parametric(returns, WEIGHTS, window=WINDOW),
        "monte_carlo": lambda: backtest_monte_carlo(returns, WEIGHTS, window=WINDOW, n_sims=5000),
        "filtered_historical": lambda: backtest_filtered_historical(
            returns, WEIGHTS, window=WINDOW
        ),
    }

    for name, run_fn in runs.items():
        print(f"Running {name} backtest...")
        start = time.time()
        result = run_fn()
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s — {result.shape[0]} days")

        out_path = f"{OUTPUT_DIR}/{name}.parquet"
        result.to_parquet(out_path)
        print(f"  Saved to {out_path}\n")


if __name__ == "__main__":
    main()
