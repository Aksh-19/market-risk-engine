"""
Filtered Historical Simulation (FHS) VaR and Expected Shortfall.

Combines GARCH's time-varying volatility with Historical Simulation's
nonparametric tail -- the standard answer to two separate weaknesses seen
so far in this project: plain Historical Sim treats a 2017 calm day and a
March-2020 crash day as equally representative of "tomorrow" (no volatility
conditioning), while Parametric/Monte Carlo condition on today's Sigma but
then assume normality, which Phase 1's own 6-sigma-day findings showed is
false at the tails.

MECHANISM:
1. Standardize: z_{i,t} = r_{i,t} / sigma_{i,t}(GARCH), per asset. These
   standardized residuals are approximately iid -- the volatility
   clustering has been stripped out, leaving pure "shock magnitude,
   independent of regime."
2. Preserve cross-asset correlation by keeping each day's shock vector
   intact across assets -- never shuffle assets independently, since a
   historical day where equities crashed together needs that co-movement
   to survive into the simulated set.
3. Rescale every historical shock by TOMORROW's GARCH volatility forecast:
   r_sim_{i,t} = z_{i,t} * sigma_{i,tomorrow}. This is the actual payoff --
   real historical shock *shapes* (fat tails, skew, crash co-movements),
   scaled to today's volatility regime rather than whatever regime existed
   when the historical shock actually happened.
4. VaR/ES = empirical quantile / tail-average of the weighted, rescaled
   simulated portfolio series -- nonparametric at the final step, but
   volatility-conditioned, unlike plain Historical Sim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.volatility.garch import fit_garch11, forecast_variance
from risk_engine.var.portfolio import validate_weights


def _standardized_residuals_and_forecast(
    returns: pd.DataFrame,
    cached_params: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Fit (or reuse) GARCH(1,1) per asset. Returns:
      Z          -- (T, n_assets) standardized residuals, z_t = r_t / sigma_t
      sigma_next -- (n_assets,) one-step-ahead GARCH volatility forecast.

    cached_params: optional {ticker: (omega, alpha, beta)}. When provided,
    skips MLE re-fitting entirely and just re-runs the (cheap) forward
    recursion with these fixed parameters on the current window -- this is
    the hook rolling.py uses for weekly-refit / daily-reuse backtesting.
    """
    from risk_engine.volatility.garch import (
        GARCH11Result,
        conditional_variance_from_params,
    )

    z_columns = {}
    sigma_next = []

    for col in returns.columns:
        if cached_params is not None:
            omega, alpha, beta = cached_params[col]
            sigma2 = conditional_variance_from_params(returns[col], omega, alpha, beta)
            fitted = GARCH11Result(
                omega=omega,
                alpha=alpha,
                beta=beta,
                log_likelihood=float("nan"),
                converged=True,
                conditional_variance=sigma2,
            )
        else:
            fitted = fit_garch11(returns[col])
            sigma2 = fitted.conditional_variance

        sigma_t = np.sqrt(sigma2)
        z_columns[col] = returns[col].to_numpy() / sigma_t

        var_forecast_1step = forecast_variance(fitted, horizon=1)[0]
        sigma_next.append(np.sqrt(var_forecast_1step))

    Z = pd.DataFrame(z_columns, index=returns.index)
    return Z, np.array(sigma_next)


def _simulate_fhs_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    cached_params: dict[str, tuple[float, float, float]] | None = None,
) -> pd.Series:
    n_assets = returns.shape[1]
    w = validate_weights(weights, n_assets)

    Z, sigma_next = _standardized_residuals_and_forecast(returns, cached_params)

    r_sim = Z.to_numpy() * sigma_next
    r_p_sim = r_sim @ w

    return pd.Series(r_p_sim, index=returns.index, name="fhs_simulated_portfolio_return")


def filtered_historical_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    cached_params: dict[str, tuple[float, float, float]] | None = None,
) -> dict[float, float]:
    """FHS VaR: empirical quantile of the volatility-rescaled simulated series."""
    r_p_sim = _simulate_fhs_portfolio_returns(returns, weights, cached_params)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        quantile_return = np.percentile(r_p_sim, alpha * 100)
        result[alpha] = float(-quantile_return)
    return result


def filtered_historical_expected_shortfall(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    cached_params: dict[str, tuple[float, float, float]] | None = None,
) -> dict[float, float]:
    """FHS ES: tail-average of the volatility-rescaled simulated series."""
    r_p_sim = _simulate_fhs_portfolio_returns(returns, weights, cached_params)

    result: dict[float, float] = {}
    for alpha in confidence_levels:
        threshold = np.percentile(r_p_sim, alpha * 100)
        tail = r_p_sim[r_p_sim <= threshold]
        result[alpha] = float(-tail.mean())
    return result
