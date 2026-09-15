"""
Shared portfolio-level utilities for VaR computation.

Every VaR method in this project operates on a PORTFOLIO quantity, not a
single asset in isolation -- because that's how VaR is used on a real risk
desk. This module is intentionally narrow: it only handles (a) weights and
(b) the realized portfolio return series r_p = sum_i w_i * r_i. Historical
Simulation VaR uses r_p directly. Parametric and Monte Carlo VaR use it only
for centering purposes (if at all) -- their variance comes from Sigma, via
covariance_selector.py, not from this series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    """
    Validate a weight vector. Enforces weights sum to 1 -- NOT that weights
    are non-negative, since short positions are legitimate and the dot
    product handles them correctly automatically. Don't assume long-only
    when reasoning about results downstream.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_assets,):
        raise ValueError(f"weights must have shape ({n_assets},), got {w.shape}")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"weights must sum to 1.0, got {w.sum():.6f}")
    return w


def portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    demean: bool = False,
) -> pd.Series:
    """
    r_p = sum_i w_i * r_i.

    demean=False by default: Historical Simulation VaR should use RAW
    portfolio returns -- the empirical quantile already reflects whatever
    drift is in the data. Silently demeaning here would change what "1%
    VaR" means for that method without anyone noticing. This flag exists so
    Parametric/Monte Carlo, which pick an explicit mean convention, can
    reuse this function for auxiliary quantities without duplicating the
    weighted-sum logic.
    """
    w = validate_weights(weights, returns.shape[1])
    r_p = returns.to_numpy() @ w
    r_p = pd.Series(r_p, index=returns.index, name="portfolio_return")
    if demean:
        r_p = r_p - r_p.mean()
    return r_p


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    sigma_p^2 = w^T Sigma w.

    This is what makes Parametric/Monte Carlo VaR fundamentally different
    from Historical Sim: it never looks at realized portfolio returns, only
    at Sigma (estimated separately -- EWMA, Ledoit-Wolf, or GARCH-hybrid)
    and the weights. Two portfolios with identical realized histories but a
    different Sigma choice get different Parametric VaR numbers. That's the
    point, not a bug -- but it's exactly why methods can disagree.
    """
    w = validate_weights(weights, cov_matrix.shape[0])
    return float(w @ cov_matrix @ w)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """sigma_p = sqrt(w^T Sigma w) -- feeds directly into the z-score formula."""
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))
