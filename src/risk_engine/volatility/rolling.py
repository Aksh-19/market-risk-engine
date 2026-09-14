"""
Rolling (simple, equal-weighted) historical volatility -- the naive
baseline.

WHY THIS BELONGS IN THE PROJECT, EVEN THOUGH IT'S "WORSE" THAN EWMA/GARCH
------------------------------------------------------------------------------
Without a baseline, "GARCH captures volatility clustering well" is an
unfalsifiable claim -- well, compared to WHAT? This module exists purely to
give EWMA and GARCH something concrete to beat. Recall from the EWMA
derivation: equal-weighting every day in a fixed window causes two specific,
nameable failures --

  1. GHOSTING: a single huge return sits in the window with full weight for
     exactly `window` days, then vanishes entirely the instant it rolls out
     -- a visible step-change in estimated volatility that has nothing to do
     with any real change in market conditions, purely a window-length
     artifact.
  2. LAG: a genuinely new volatile regime only shows up gradually, diluted
     by all the calm days still sitting in the window.

Being able to point at your OWN computed rolling-vol series and show the
exact ghosting step-change around, say, day window+1 after a spike, then
show EWMA/GARCH handling the same spike smoothly, is a much stronger project
narrative than asserting it from theory alone.
"""

from __future__ import annotations

import pandas as pd


def rolling_volatility(
    returns: pd.Series,
    window: int = 30,
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.Series:
    """
    Equal-weighted rolling standard deviation of returns.

    window=30 is a common practitioner default -- roughly six weeks of
    trading days, long enough to smooth daily noise, short enough to react
    within a couple months to a real regime change (contrast with EWMA's
    ~11-day half-life, which reacts much faster).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if len(returns) <= window:
        raise ValueError(f"Need more than window={window} observations, got {len(returns)}")

    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * (trading_days**0.5)
    return vol.rename(f"rolling_vol_w{window}")
