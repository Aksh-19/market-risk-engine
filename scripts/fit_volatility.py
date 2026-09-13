"""
Phase 2 entry point: fit EWMA and GARCH(1,1) volatility models on the
Phase 1 returns matrix, and compare them.

    python scripts/fit_volatility.py

Reads data/processed/returns_matrix.parquet (Phase 1's output contract) and
writes data/processed/volatility_summary.parquet — the per-asset EWMA and
GARCH volatility series Phase 3 will need for VaR calculations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from risk_engine.volatility import (
    covariance_to_correlation,
    ewma_covariance_matrix,
    ewma_volatility,
    fit_garch11,
    garch_volatility_series,
)
from risk_engine.storage import RiskDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(input_path: str, output_path: str, db_path: str) -> None:
    returns = pd.read_parquet(input_path)
    logger.info("Loaded returns matrix: %d dates x %d assets", *returns.shape)

    db = RiskDatabase(db_path)
    ewma_cols = {}
    garch_cols = {}

    print("\n=== Per-asset volatility (annualized) ===")
    for ticker in returns.columns:
        r = returns[ticker]

        vol_ewma = ewma_volatility(r, lambda_=0.94, min_periods=30, annualize=True)
        ewma_cols[f"{ticker}_ewma_vol"] = vol_ewma
        db.write_volatility(vol_ewma, ticker=ticker, model="ewma")

        fitted = fit_garch11(r)
        vol_garch = garch_volatility_series(r, fitted) * (252**0.5)
        garch_cols[f"{ticker}_garch_vol"] = vol_garch
        db.write_volatility(vol_garch, ticker=ticker, model="garch")
        db.write_garch_params(
            ticker,
            fit_date=returns.index[-1],
            omega=fitted.omega,
            alpha=fitted.alpha,
            beta=fitted.beta,
            log_likelihood=fitted.log_likelihood,
            converged=fitted.converged,
        )

        print(f"\n{ticker}:")
        print(f"  Latest EWMA vol (annualized):  {vol_ewma.iloc[-1]:.4f}")
        print(f"  Latest GARCH vol (annualized): {vol_garch.iloc[-1]:.4f}")
        print(f"  {fitted.summary()}")

    print("\n=== EWMA correlation matrix (most recent date) ===")
    cov_series = ewma_covariance_matrix(returns, lambda_=0.94, min_periods=30)
    last_date = list(cov_series.keys())[-1]

    corr = covariance_to_correlation(cov_series[last_date])
    print(f"As of {last_date.date()}:")
    print(pd.DataFrame(corr, index=returns.columns, columns=returns.columns).round(3))

    # Persist the most recent covariance snapshot -- storing every historical
    # date's matrix would be a lot of rows for little benefit at this stage;
    # Phase 3's rolling backtests can decide later whether they need more
    # than "latest" snapshots per model.
    db.write_covariance_snapshot(
        last_date, cov_series[last_date], list(returns.columns), model="ewma"
    )
    db.close()

    summary = pd.DataFrame({**ewma_cols, **garch_cols})
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main(
        input_path="data/processed/returns_matrix.parquet",
        output_path="data/processed/volatility_summary.parquet",
        db_path="data/processed/risk_engine.db",
    )
