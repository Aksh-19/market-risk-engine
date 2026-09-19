"""
Basel Committee traffic-light backtesting zones.

Unlike Kupiec/Christoffersen (statistical hypothesis tests we chose to
run), this is the LITERAL regulatory mechanism banks are required to
self-assess against: count breaches of 99% VaR (the 1% level specifically
-- this is a fixed regulatory standard) over a rolling 250-trading-day
window (~1 year), and classify into Green/Yellow/Red.

THRESHOLDS (fixed by the Basel Committee, not a project choice):
  Green:  0-4 breaches  -- consistent with a correctly calibrated model
  Yellow: 5-9 breaches  -- possibly problematic; increased capital
                           multiplier and regulatory scrutiny
  Red:    10+ breaches  -- strong evidence the model understates risk;
                           mandatory capital multiplier increase

These aren't arbitrary: under a truly correct model, 250 trials at p=0.01
has an EXPECTED breach count of 2.5. Green's upper bound is chosen so a
well-calibrated model lands there with high probability by chance; Red
represents roughly 4x the claimed rate, a point where landing there by
chance under a correct model is vanishingly unlikely.

WHY THIS DIFFERS FROM KUPIEC: Kupiec gives one p-value over the FULL
backtest history. Basel's zones are about ROLLING 250-day windows -- the
regulatory question is "how does this model look right now, reassessed
every year," not "was this model ever right over 8 years." A model can
pass Kupiec in aggregate while still spending real stretches in Yellow or
Red -- which is exactly the kind of thing a single aggregate p-value
cannot show.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

GREEN_MAX = 4
YELLOW_MAX = 9
BASEL_WINDOW = 250


@dataclass
class BaselZoneResult:
    """Basel traffic-light classification for one 250-day window."""

    window_end_date: pd.Timestamp
    n_breaches: int
    zone: str  # "Green", "Yellow", "Red"


def classify_breach_count(n_breaches: int) -> str:
    """Map a raw breach count (out of 250 days) to a Basel zone."""
    if n_breaches <= GREEN_MAX:
        return "Green"
    elif n_breaches <= YELLOW_MAX:
        return "Yellow"
    else:
        return "Red"


def basel_traffic_light(
    breach_series: pd.Series,
    window: int = BASEL_WINDOW,
) -> pd.DataFrame:
    """
    Roll a 250-day window across the FULL out-of-sample breach series,
    classifying each window into Green/Yellow/Red. This is deliberately a
    SEPARATE rolling pass over rolling.py's already-out-of-sample breach
    series -- rolling.py's window (500 days) governs how each day's VaR
    forecast was COMPUTED; this function's window (250 days, fixed by
    Basel) governs how breaches are RETROSPECTIVELY counted and graded.
    The two windows serve different purposes and are not meant to match.

    Returns one row per 250-day window (dates from window-1 through the
    end of breach_series), so the zone can be tracked over time rather
    than collapsed into one summary number.
    """
    breaches = breach_series.to_numpy().astype(int)
    dates = breach_series.index
    n = len(breaches)

    if n < window:
        raise ValueError(f"Need at least {window} observations, got {n}")

    records = []
    for t in range(window - 1, n):
        window_breaches = breaches[t - window + 1 : t + 1]
        count = int(window_breaches.sum())
        zone = classify_breach_count(count)
        records.append({"window_end_date": dates[t], "n_breaches": count, "zone": zone})

    return pd.DataFrame(records)


def summarize_zones(zone_df: pd.DataFrame) -> dict[str, float]:
    """
    Summarize what fraction of all 250-day windows in the backtest period
    landed in each zone -- a single, interpretable snapshot of how often a
    model would have triggered regulatory scrutiny over its full history,
    rather than just its current status.
    """
    counts = zone_df["zone"].value_counts()
    total = len(zone_df)
    return {zone: float(counts.get(zone, 0) / total) for zone in ["Green", "Yellow", "Red"]}
