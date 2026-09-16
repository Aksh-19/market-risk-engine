"""
Monte Carlo VaR and Expected Shortfall.

Simulates thousands of correlated asset-return scenarios via Cholesky
decomposition of Sigma, applies portfolio weights to each scenario, then
takes the EMPIRICAL quantile/tail-average of the simulated portfolio P&L --
Parametric's covariance-driven approach, read off via simulation instead of
a closed-form z-score formula.

WHY CHOLESKY: independent standard normal draws Z can't reproduce a
correlated covariance structure on their own. Cholesky factors
Sigma = L @ L.T (L lower-triangular). Then:
    Cov(L @ Z) = L @ Cov(Z) @ L.T = L @ I @ L.T = L @ L.T = Sigma
so L @ Z has EXACTLY the specified covariance, built purely from
independent noise -- the same mechanism used industry-wide for multi-asset
derivative pricing and portfolio stress testing.

RELATIONSHIP TO PARAMETRIC VaR: both assume multivariate normal returns
with the same Sigma. As n_sims -> infinity, Monte Carlo VaR converges to
Parametric's closed-form answer -- it's the SAME calculation via
simulation rather than algebra, which makes that convergence a genuine,
checkable validation (see test suite) rather than a coincidence. For this
project's linear, equal-weight portfolio, Monte Carlo isn't discovering a
distinct "third opinion" on tail risk the way Historical Sim does -- its
real value would show up on portfolios Parametric can't solve in closed
form (options, path-dependent payoffs, non-normal shocks), which this
project doesn't have. Included for completeness and to demonstrate the
technique, with that limitation stated plainly rather than implied away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.var.covariance_selector import get_covariance_matrix
from risk_engine.var.portfolio import validate_weights


def _simulate_portfolio_pnl(
    returns: pd.DataFrame,
    weights: np.ndarray,
    cov_method: str,
    n_sims: int,
    seed: int | None,
) -> np.ndarray:
    """
    Simulate n_sims correlated return scenarios via Cholesky decomposition
    of Sigma, apply portfolio weights. Shared by VaR and ES so a single
    call site only draws once rather than duplicating simulation cost.
    """
    n_assets = returns.shape[1]
    w = validate_weights(weights, n_assets)
    cov_matrix = get_covariance_matrix(returns, method=cov_method)

    L = np.linalg.cholesky(cov_matrix)  # Sigma = L @ L.T

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sims, n_assets))  # independent standard normals
    correlated_draws = Z @ L.T  # shape (n_sims, n_assets); Cov = Sigma

    return correlated_draws @ w  # simulated portfolio P&L, shape (n_sims,)


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    cov_method: str = "ewma",
    n_sims: int = 50_000,
    seed: int | None = None,
) -> dict[float, float]:
    """
    Monte Carlo VaR: empirical quantile of SIMULATED portfolio P&L.

    seed: pass an explicit int for reproducible results (used throughout
    the test suite); leave None for fresh randomness each call.
    """
    pnl = _simulate_portfolio_pnl(returns, weights, cov_method, n_sims, seed)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        quantile_pnl = np.percentile(pnl, alpha * 100)
        result[alpha] = float(-quantile_pnl)
    return result


def monte_carlo_expected_shortfall(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    cov_method: str = "ewma",
    n_sims: int = 50_000,
    seed: int | None = None,
) -> dict[float, float]:
    """
    Monte Carlo ES: mean of simulated P&L at or below the VaR quantile --
    the same tail-averaging idea as Historical Sim's ES, applied to
    simulated rather than realized data.
    """
    pnl = _simulate_portfolio_pnl(returns, weights, cov_method, n_sims, seed)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        threshold = np.percentile(pnl, alpha * 100)
        tail = pnl[pnl <= threshold]
        result[alpha] = float(-tail.mean())
    return result
