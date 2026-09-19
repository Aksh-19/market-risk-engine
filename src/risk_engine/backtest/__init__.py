"""Rolling out-of-sample VaR backtesting and statistical validation."""

from risk_engine.backtest.rolling import (
    backtest_historical,
    backtest_parametric,
    backtest_monte_carlo,
    backtest_filtered_historical,
)
from risk_engine.backtest.kupiec import kupiec_pof_test, KupiecResult
from risk_engine.backtest.christoffersen import (
    christoffersen_independence_test,
    ChristoffersenResult,
)
from risk_engine.backtest.basel import (
    classify_breach_count,
    basel_traffic_light,
    summarize_zones,
)

__all__ = [
    "backtest_historical",
    "backtest_parametric",
    "backtest_monte_carlo",
    "backtest_filtered_historical",
    "kupiec_pof_test",
    "KupiecResult",
    "christoffersen_independence_test",
    "ChristoffersenResult",
    "classify_breach_count",
    "basel_traffic_light",
    "summarize_zones",
]
