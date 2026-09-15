"""
Historical Simulation VaR and Expected Shortfall.

Nonparametric: makes zero assumptions about the shape of the return
distribution. VaR is simply the empirical quantile of realized portfolio
returns; ES is the average of everything beyond that quantile. This is the
baseline every other Phase 3 method gets compared against, precisely
because it invents nothing -- whatever the data actually did, this reflects.

SIGN CONVENTION (applies to every VaR/ES method in this project):
Both VaR and ES are reported as POSITIVE numbers representing a magnitude
of loss, even though they're derived from the negative (loss) tail of the
return distribution. "1-day 5% VaR = 1.8%" reads as "on 95% of days, we
don't expect to lose more than 1.8%" -- not as a signed return.

LIMITATION, STATED PLAINLY: this method is entirely backward-looking and
sample-size-limited. At n=2515 observations, 1% VaR is estimated from
roughly the worst ~25 days in the whole sample; going further into the
tail (e.g. 0.1%) would mean estimating a quantile from a mere ~2-3 data
points -- not reliable. This is precisely the gap EVT/GPD tail modeling
exists to address, and part of why we run Historical Sim alongside
Parametric and Monte Carlo rather than trusting any single method alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.var.portfolio import portfolio_returns


def historical_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
) -> dict[float, float]:
    """
    Historical Simulation VaR at each confidence level.

    confidence_levels are expressed as the TAIL probability (0.01 = "1%
    VaR", i.e. the 1st percentile of the loss distribution) -- not as
    "99% confidence," to keep this consistent with how Expected Shortfall
    is naturally expressed as "average loss beyond the alpha-quantile."

    Returns
    -------
    dict mapping each confidence level to its VaR (positive = loss magnitude).
    """
    r_p = portfolio_returns(returns, weights, demean=False)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        quantile_return = np.percentile(r_p, alpha * 100)
        result[alpha] = float(-quantile_return)  # flip sign: loss reported positive
    return result


def historical_expected_shortfall(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
) -> dict[float, float]:
    """
    Historical Simulation Expected Shortfall (CVaR) at each confidence level.

    ES = average of all portfolio returns AT OR BELOW the alpha-quantile
    threshold. Always >= VaR at the same confidence level (it's an average
    over a strictly worse-or-equal set of outcomes) -- this is a good
    sanity invariant to check in tests.
    """
    r_p = portfolio_returns(returns, weights, demean=False)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        threshold = np.percentile(r_p, alpha * 100)
        tail_losses = r_p[r_p <= threshold]
        result[alpha] = float(-tail_losses.mean())  # flip sign: loss reported positive
    return result
