"""
Data quality checks.

WHY A SEPARATE VALIDATION MODULE (rather than eyeballing a plot)
---------------------------------------------------------------------
This is the "garbage in, garbage out" firewall. A GARCH model fit on data
with an undetected stock-split artifact will produce a volatility estimate
that's wrong by construction, and nothing about the downstream VaR math will
tell you that — it'll just confidently output a wrong number. In a real risk
function, data validation is a formal, automated gate that runs BEFORE any
model touches the data, precisely because humans don't reliably eyeball
5+ years of daily data across a 20-asset universe. We encode the checks a
risk analyst would manually do, as functions, so they run every time.

The four checks below map directly to the four most common ways market data
silently lies to you:
  - missing days        -> stale/interpolated risk estimates
  - zero-variance days   -> a feed outage disguised as "no price movement"
  - extreme single-day moves -> possible bad ticks OR possible genuine
    tail events (this check doesn't auto-reject, it flags for review —
    a real -20% day, e.g. a March 2020 style shock, is exactly the kind of
    event your EVT/tail-risk work cares about, so we never silently drop it)
  - non-monotonic / duplicate dates -> merge or index bugs upstream
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    ticker: str
    missing_dates: int = 0
    zero_variance_days: int = 0
    extreme_moves: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    duplicate_dates: int = 0

    @property
    def is_clean(self) -> bool:
        return (
            self.missing_dates == 0 and self.zero_variance_days == 0 and self.duplicate_dates == 0
        )

    def summary(self) -> str:
        lines = [f"Validation report for {self.ticker}:"]
        lines.append(f"  missing dates in expected trading calendar: {self.missing_dates}")
        lines.append(f"  zero-variance (flat) days: {self.zero_variance_days}")
        lines.append(f"  duplicate index entries: {self.duplicate_dates}")
        lines.append(f"  extreme moves flagged (|z| > threshold): {len(self.extreme_moves)}")
        if len(self.extreme_moves):
            lines.append(f"    -> {self.extreme_moves.round(4).to_dict()}")
        return "\n".join(lines)


def validate_price_series(
    prices: pd.DataFrame,
    ticker: str,
    price_field: str = "Adj Close",
    extreme_z_threshold: float = 6.0,
) -> ValidationReport:
    """
    Run the standard data-quality checklist on one ticker's price history.

    extreme_z_threshold=6.0 means: flag any return more than 6 standard
    deviations from the series mean. This is deliberately generous — daily
    equity returns are fat-tailed (this is literally the premise of your EVT
    project), so a naive threshold like 3-sigma would flag dozens of
    perfectly genuine trading days and train you to ignore the warnings.
    6-sigma on a Normal would be almost impossible; on real fat-tailed
    returns it still happens a handful of times per decade, which is exactly
    the resolution we want: rare enough to be worth a human look.
    """
    report = ValidationReport(ticker=ticker)

    idx = prices.index
    report.duplicate_dates = int(idx.duplicated().sum())

    # Expected trading calendar: business days between min and max date.
    # This is an approximation (ignores exchange-specific holidays) but is
    # good enough to catch real gaps like a multi-day feed outage.
    expected = pd.bdate_range(idx.min(), idx.max())
    report.missing_dates = len(expected.difference(idx))

    returns = prices[price_field].pct_change().dropna()
    report.zero_variance_days = int((returns == 0).sum())

    z_scores = (returns - returns.mean()) / returns.std()
    report.extreme_moves = returns[z_scores.abs() > extreme_z_threshold]

    return report
