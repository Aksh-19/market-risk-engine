"""
Parametric (Variance-Covariance) VaR and Expected Shortfall.

Assumes portfolio returns are normally distributed: VaR is then just a
z-score times portfolio volatility, with volatility sourced ENTIRELY from
Sigma (via covariance_selector.py) -- never from realized portfolio
returns. This is what makes Parametric fundamentally different from
Historical Simulation, not just a different formula on the same inputs.

Zero-mean convention: mu_p is assumed 0 (standard, conservative daily-VaR
practice -- 1-day drift is negligible relative to volatility). This
collapses VaR_alpha = -mu_p + z_alpha*sigma_p to VaR_alpha = z_alpha*sigma_p.

WHY THIS METHOD IS FAST BUT FRAGILE, GROUNDED IN OUR OWN DATA:
Phase 1's validators flagged 6 SPY days as 6-sigma moves over 2015-2024.
Under a genuine normal distribution, a 6-sigma move should occur roughly
once every ~500 million trading days -- so observing 6 in one decade is
direct, data-driven evidence that the normality assumption underlying this
entire method is false at the tails. Parametric VaR will therefore tend to
UNDERSTATE tail risk relative to Historical Simulation. That's not a bug in
this implementation -- it's the textbook, well-documented limitation of the
method, and precisely why Phase 3 runs three methods side by side instead
of trusting any single one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from risk_engine.var.portfolio import portfolio_volatility
from risk_engine.var.covariance_selector import get_covariance_matrix

import pandas as pd


def parametric_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    cov_method: str = "ewma",
) -> dict[float, float]:
    """
    Parametric VaR: VaR_alpha = z_alpha * sigma_p, with sigma_p = sqrt(w'Sigma*w).

    cov_method selects the Sigma source via covariance_selector.py:
    'ewma' (default, time-varying), 'ledoit_wolf' (full-sample, shrunk),
    or 'garch_hybrid' (GARCH per-asset vol + EWMA correlation).
    """
    cov_matrix = get_covariance_matrix(returns, method=cov_method)
    sigma_p = portfolio_volatility(weights, cov_matrix)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        z = norm.ppf(alpha)  # negative value, e.g. norm.ppf(0.01) = -2.326
        result[alpha] = float(-z * sigma_p)  # flip sign: loss reported positive
    return result


def parametric_expected_shortfall(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    cov_method: str = "ewma",
) -> dict[float, float]:
    """
    Parametric ES under normality: ES_alpha = sigma_p * phi(z_alpha) / alpha,
    where phi is the standard normal PDF. Derived from the normal
    distribution's density -- NOT an empirical tail average, since there is
    no empirical tail here; everything comes from the assumed distribution
    shape plus sigma_p.
    """
    cov_matrix = get_covariance_matrix(returns, method=cov_method)
    sigma_p = portfolio_volatility(weights, cov_matrix)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        z = norm.ppf(alpha)
        es = sigma_p * norm.pdf(z) / alpha
        result[alpha] = float(es)
    return result
