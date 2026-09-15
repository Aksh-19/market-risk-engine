"""
Orchestrator: compute VaR and Expected Shortfall across all implemented
methods for the equal-weight SPY/AAPL/TLT/GLD portfolio.

Usage: python3 scripts/run_var.py
"""

from __future__ import annotations

import pandas as pd

from risk_engine.var import historical_var, historical_expected_shortfall

RETURNS_PATH = "data/processed/returns_matrix.parquet"
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
CONFIDENCE_LEVELS = (0.01, 0.05)


def main() -> None:
    returns = pd.read_parquet(RETURNS_PATH)

    print(f"Universe: {list(returns.columns)}, weights: {WEIGHTS}")
    print(f"Observations: {len(returns)}\n")

    var = historical_var(returns, WEIGHTS, CONFIDENCE_LEVELS)
    es = historical_expected_shortfall(returns, WEIGHTS, CONFIDENCE_LEVELS)

    print("Historical Simulation VaR / ES")
    print("-" * 40)
    for alpha in CONFIDENCE_LEVELS:
        print(f"  {int(alpha*100)}%  VaR: {var[alpha]:.4%}   ES: {es[alpha]:.4%}")


if __name__ == "__main__":
    main()
