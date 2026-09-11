"""
Return calculations and multi-asset alignment.

WHY LOG RETURNS, NOT SIMPLE RETURNS
--------------------------------------
This is one of the first "why" questions every quant asks, and it's worth
internalizing properly rather than cargo-culting it:

  simple return:  R_t = (P_t - P_{t-1}) / P_{t-1}
  log return:      r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)

Three concrete reasons risk engines standardize on log returns:

1. TIME-ADDITIVITY. A k-day log return is just the SUM of k daily log
   returns: r_(t-k, t) = r_t + r_{t-1} + ... + r_{t-k+1}. This is what lets
   you scale a 1-day VaR to a 10-day VaR by multiplying variance by 10 (the
   sqrt-of-time rule you'll use in Phase 3) — it only works cleanly in log
   space. Simple returns do NOT sum across time; you'd need to compound them
   multiplicatively, which is messier and doesn't decompose the same way.

2. SYMMETRY. A simple return of +50% followed by -50% does NOT bring you
   back to your starting price (100 -> 150 -> 75). Log returns of +x and -x
   are symmetric around zero and treat gains/losses even-handedly, which
   matters when you're fitting a symmetric or near-symmetric distribution
   (Normal, Student-t) to returns in Phase 2 — fitting simple returns biases
   the tails.

3. NUMERICAL STABILITY & approximate normality. For daily returns, ln(1+R) ≈ R
   when R is small, so log returns don't sacrifice much intuition day-to-day,
   but they behave much better under aggregation and are less prone to
   returns below -100% (log returns can't hit -100% at a finite price the
   way a naive simple-return calc edge case could).

The trade-off: log returns are NOT what you'd report to a client as "your
portfolio was up 3.2% this month" — for that, simple/arithmetic returns are
correct and log returns would be a systematic (small) underestimate. Risk
engines use log returns internally for the modeling math; P&L reporting
layers convert back to simple returns for human consumption. We'll keep
both available.

WHY WE ALIGN BEFORE COMPUTING MULTI-ASSET RETURNS
----------------------------------------------------
Two tickers can each have a perfectly clean, gap-free price series
individually and still not share the same calendar (different exchange
holidays, different listing dates, an early-close day for one but not the
other). If you naively join two return series on row position rather than
on date, you introduce phantom returns that don't correspond to real
co-movement — this silently corrupts any correlation/covariance estimate
downstream (which the Monte-Carlo VaR in Phase 3 depends on heavily). We
always join on a shared DatetimeIndex and drop dates where any asset is
missing (an inner join across the universe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}). First observation is dropped (no prior price)."""
    if (prices <= 0).any():
        raise ValueError(
            "Non-positive price found — log returns are undefined for P<=0. "
            "Check for bad ticks or unadjusted corporate-action artifacts."
        )
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.dropna()


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """R_t = P_t / P_{t-1} - 1. Use for P&L / human-facing reporting."""
    simple_ret = prices.pct_change()
    return simple_ret.dropna()


def build_returns_matrix(
    price_frames: dict[str, pd.DataFrame],
    price_field: str = "Adj Close",
    method: str = "log",
) -> pd.DataFrame:
    """
    Turn {ticker: OHLCV DataFrame} into a single aligned returns matrix,
    columns = tickers, index = shared trading dates.

    This is the object every downstream module (EWMA/GARCH vol, historical
    simulation, Monte Carlo covariance) actually consumes — it's the single
    contract point between "data" and "modeling" in this codebase.
    """
    if method not in ("log", "simple"):
        raise ValueError(f"method must be 'log' or 'simple', got {method!r}")

    calc = compute_log_returns if method == "log" else compute_simple_returns

    series = {}
    for ticker, df in price_frames.items():
        if price_field not in df.columns:
            raise KeyError(f"{ticker}: expected column {price_field!r}, got {list(df.columns)}")
        series[ticker] = calc(df[price_field])

    # Inner join on date index -> only dates where EVERY asset has a valid
    # return survive. This is the alignment step discussed in the docstring.
    returns_matrix = pd.concat(series, axis=1, join="inner")
    returns_matrix.columns = list(price_frames.keys())
    return returns_matrix.sort_index()
