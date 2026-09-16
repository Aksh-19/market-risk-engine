"""
Orchestrator: compute VaR and Expected Shortfall across all implemented
methods for the equal-weight SPY/AAPL/TLT/GLD portfolio.

Usage: python3 scripts/run_var.py
"""

from __future__ import annotations

import pandas as pd
from risk_engine.storage.db import RiskDatabase

from risk_engine.var import (
    historical_var,
    historical_expected_shortfall,
    parametric_var,
    parametric_expected_shortfall,
    monte_carlo_var,
    monte_carlo_expected_shortfall,
    filtered_historical_var,
    filtered_historical_expected_shortfall,
)

RETURNS_PATH = "data/processed/returns_matrix.parquet"
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
CONFIDENCE_LEVELS = (0.01, 0.05)
PARAMETRIC_COV_METHODS = ("ewma", "ledoit_wolf", "garch_hybrid")
MONTE_CARLO_COV_METHODS = ("ewma", "ledoit_wolf")
MC_SIMS = 50_000
MC_SEED = 2024


def main() -> None:
    returns = pd.read_parquet(RETURNS_PATH)

    print(f"Universe: {list(returns.columns)}, weights: {WEIGHTS}")
    print(f"Observations: {len(returns)}\n")

    summary_rows = []

    hist_var = historical_var(returns, WEIGHTS, CONFIDENCE_LEVELS)
    hist_es = historical_expected_shortfall(returns, WEIGHTS, CONFIDENCE_LEVELS)
    print("Historical Simulation VaR / ES")
    print("-" * 40)
    for alpha in CONFIDENCE_LEVELS:
        print(f"  {int(alpha*100)}%  VaR: {hist_var[alpha]:.4%}   ES: {hist_es[alpha]:.4%}")
        summary_rows.append(("Historical Sim", "-", alpha, hist_var[alpha], hist_es[alpha]))

    print("\nParametric (Variance-Covariance) VaR / ES")
    print("-" * 40)
    for cov_method in PARAMETRIC_COV_METHODS:
        var = parametric_var(returns, WEIGHTS, CONFIDENCE_LEVELS, cov_method=cov_method)
        es = parametric_expected_shortfall(
            returns, WEIGHTS, CONFIDENCE_LEVELS, cov_method=cov_method
        )
        print(f"  [{cov_method}]")
        for alpha in CONFIDENCE_LEVELS:
            print(f"    {int(alpha*100)}%  VaR: {var[alpha]:.4%}   ES: {es[alpha]:.4%}")
            summary_rows.append(("Parametric", cov_method, alpha, var[alpha], es[alpha]))

    print(f"\nMonte Carlo VaR / ES ({MC_SIMS:,} simulations, seed={MC_SEED})")
    print("-" * 40)
    for cov_method in MONTE_CARLO_COV_METHODS:
        var = monte_carlo_var(
            returns,
            WEIGHTS,
            CONFIDENCE_LEVELS,
            cov_method=cov_method,
            n_sims=MC_SIMS,
            seed=MC_SEED,
        )
        es = monte_carlo_expected_shortfall(
            returns,
            WEIGHTS,
            CONFIDENCE_LEVELS,
            cov_method=cov_method,
            n_sims=MC_SIMS,
            seed=MC_SEED,
        )
        print(f"  [{cov_method}]")
        for alpha in CONFIDENCE_LEVELS:
            print(f"    {int(alpha*100)}%  VaR: {var[alpha]:.4%}   ES: {es[alpha]:.4%}")
            summary_rows.append(("Monte Carlo", cov_method, alpha, var[alpha], es[alpha]))

    fhs_var = filtered_historical_var(returns, WEIGHTS, CONFIDENCE_LEVELS)
    fhs_es = filtered_historical_expected_shortfall(returns, WEIGHTS, CONFIDENCE_LEVELS)
    print("\nFiltered Historical Simulation VaR / ES (GARCH-conditioned)")
    print("-" * 40)
    for alpha in CONFIDENCE_LEVELS:
        print(f"  {int(alpha*100)}%  VaR: {fhs_var[alpha]:.4%}   ES: {fhs_es[alpha]:.4%}")
        summary_rows.append(
            ("Filtered Historical Sim", "GARCH", alpha, fhs_var[alpha], fhs_es[alpha])
        )

    # --- Comparison table -------------------------------------------------
    summary_df = pd.DataFrame(summary_rows, columns=["method", "cov_source", "alpha", "VaR", "ES"])

    # Persist RAW floats before any display formatting -- write_var_results
    # expects numeric var/es/confidence_level columns, not formatted strings.
    persist_df = summary_df.rename(columns={"alpha": "confidence_level", "VaR": "var", "ES": "es"})
    with RiskDatabase("data/risk_engine.db") as db:
        db.write_var_results(persist_df, run_date=pd.Timestamp.today())

    display_df = summary_df.copy()
    display_df["alpha"] = (display_df["alpha"] * 100).astype(int).astype(str) + "%"
    display_df["VaR"] = (display_df["VaR"] * 100).round(4).astype(str) + "%"
    display_df["ES"] = (display_df["ES"] * 100).round(4).astype(str) + "%"

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON — all methods, all confidence levels")
    print("=" * 60)
    print(display_df.to_string(index=False))
    print("\nResults persisted to data/risk_engine.db (var_results table)")


if __name__ == "__main__":
    main()
