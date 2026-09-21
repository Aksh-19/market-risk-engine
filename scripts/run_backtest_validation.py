"""
Orchestrator: run Kupiec, Christoffersen, and Basel traffic-light
validation across all 4 VaR methods' saved backtest results.

Reads the parquet files produced by run_backtest.py -- does NOT re-run the
(slow) rolling backtest itself. Run scripts/run_backtest.py first if those
files don't exist yet or need refreshing.

Usage: python3 scripts/run_backtest_validation.py
"""

from __future__ import annotations

import pandas as pd

from risk_engine.backtest import (
    kupiec_pof_test,
    christoffersen_independence_test,
    basel_traffic_light,
    summarize_zones,
)
from risk_engine.storage.db import RiskDatabase

BACKTEST_DIR = "data/processed/backtests"
METHODS = ["historical", "parametric", "monte_carlo", "filtered_historical"]
CONFIDENCE_LEVELS = (0.01, 0.05)
BASEL_WINDOW = 250
DB_PATH = "data/processed/risk_engine.db"


def main() -> None:
    summary_rows = []

    for method in METHODS:
        df = pd.read_parquet(f"{BACKTEST_DIR}/{method}.parquet")
        print(f"{'=' * 70}")
        print(f"  {method.upper()}")
        print(f"{'=' * 70}")

        for alpha in CONFIDENCE_LEVELS:
            breach_col = f"breach_{alpha}"

            kupiec_result = kupiec_pof_test(df[breach_col], alpha_claimed=alpha)
            christ_result = christoffersen_independence_test(df[breach_col], alpha_claimed=alpha)

            print(f"\n  [{int(alpha*100)}% level]")
            print(f"    {kupiec_result.summary()}")
            print(f"    {christ_result.summary()}")

            summary_rows.append(
                {
                    "method": method,
                    "confidence_level": alpha,
                    "n_obs": kupiec_result.n_obs,
                    "n_breaches": kupiec_result.n_breaches,
                    "breach_rate": kupiec_result.p_hat_observed,
                    "kupiec_lr": kupiec_result.lr_statistic,
                    "kupiec_p_value": kupiec_result.p_value,
                    "kupiec_reject": kupiec_result.reject_calibration,
                    "christoffersen_lr_ind": christ_result.lr_independence,
                    "christoffersen_p_ind": christ_result.p_value_independence,
                    "christoffersen_reject_ind": christ_result.reject_independence,
                    "combined_lr": christ_result.lr_combined,
                    "combined_p_value": christ_result.p_value_combined,
                    "combined_reject": christ_result.reject_combined,
                }
            )

        # Basel is defined specifically for the 1% (99% VaR) level.
        zone_df = basel_traffic_light(df["breach_0.01"], window=BASEL_WINDOW)
        zone_summary = summarize_zones(zone_df)
        current_zone = zone_df.iloc[-1]["zone"]
        current_breaches = int(zone_df.iloc[-1]["n_breaches"])

        print(f"\n  [Basel traffic-light, 1% level, {BASEL_WINDOW}-day rolling windows]")
        print(
            f"    Time in zones — Green: {zone_summary['Green']:.1%}, "
            f"Yellow: {zone_summary['Yellow']:.1%}, Red: {zone_summary['Red']:.1%}"
        )
        print(f"    Most recent window: {current_breaches} breaches -> {current_zone}")
        print()

        summary_rows.append(
            {
                "method": method,
                "confidence_level": "basel_summary",
                "n_obs": len(zone_df),
                "n_breaches": current_breaches,
                "breach_rate": zone_summary["Green"],  # reused column: pct time in Green
                "kupiec_lr": None,
                "kupiec_p_value": None,
                "kupiec_reject": None,
                "christoffersen_lr_ind": None,
                "christoffersen_p_ind": None,
                "christoffersen_reject_ind": None,
                "combined_lr": None,
                "combined_p_value": None,
                "combined_reject": current_zone == "Red",
            }
        )

    # --- Persist + final comparison table --------------------------------
    summary_df = pd.DataFrame(summary_rows)

    with RiskDatabase(DB_PATH) as db:
        db.write_backtest_results(summary_df, run_date=pd.Timestamp.today())

    print("=" * 70)
    print("  FINAL VERDICT — Kupiec + Christoffersen combined test (reject = fail)")
    print("=" * 70)
    verdict_df = summary_df[summary_df["confidence_level"].isin([0.01, 0.05])][
        [
            "method",
            "confidence_level",
            "n_breaches",
            "breach_rate",
            "combined_p_value",
            "combined_reject",
        ]
    ]
    print(verdict_df.to_string(index=False))
    print(f"\nResults persisted to {DB_PATH} (backtest_results table)")


if __name__ == "__main__":
    main()
