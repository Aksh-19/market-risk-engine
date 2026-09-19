"""
Rolling out-of-sample backtest engine.

Core idea: a VaR model is only meaningful if its CLAIMED breach rate
matches its REALIZED breach rate on data it never saw. For each day t,
every method here computes VaR using ONLY data strictly before t (a fixed
500-day rolling window), then checks whether day t's actual portfolio loss
exceeded that forecast. Using full-history VaR (Phase 3's numbers) and
checking it against the same history would be circular -- the model would
be "backtested" against data it was calibrated on. This module produces
the genuinely out-of-sample breach series that Kupiec/Christoffersen need.

WINDOW AND REFIT CHOICES (see project discussion):
- 500-day rolling window (~2 years) -- closer to real industry practice
  than an ever-growing expanding window, and gives EWMA/GARCH-based methods
  a fair test of whether they can actually react to regime changes.
- GARCH refit every 5 trading days (weekly), not daily -- refitting via MLE
  every single day across ~2000 backtest days would be prohibitively slow.
  Parameters are re-estimated weekly; the conditional variance recursion
  (and thus the day-to-day standardized shocks / forecast) still updates
  daily on the shifting window, using conditional_variance_from_params().
  This is standard practice, not a shortcut that compromises validity.

BREACH CONVENTION: VaR is a positive loss magnitude (project-wide sign
convention). A breach on day t means the actual realized portfolio return
was WORSE than the negative of the VaR forecast: actual_return < -VaR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.var.historical import historical_var, historical_expected_shortfall
from risk_engine.var.parametric import parametric_var, parametric_expected_shortfall
from risk_engine.var.monte_carlo import monte_carlo_var, monte_carlo_expected_shortfall
from risk_engine.var.filtered_historical import (
    filtered_historical_var,
    filtered_historical_expected_shortfall,
)
from risk_engine.volatility.garch import fit_garch11


def _run_rolling_engine(
    returns: pd.DataFrame,
    weights: np.ndarray,
    compute_fn,
    window: int,
    confidence_levels: tuple[float, ...],
) -> pd.DataFrame:
    """
    Shared loop: slide a fixed-size window across the data, call compute_fn
    on each window to get (var_dict, es_dict), compare against the actual
    NEXT-day return, and record breaches. compute_fn's internals differ
    completely per method (this mirrors the "shared interface, different
    internals" pattern used throughout Phase 3) -- this function only
    handles the windowing and breach bookkeeping common to all of them.
    """
    w = np.asarray(weights, dtype=float)
    dates = returns.index
    n = len(returns)

    if n <= window:
        raise ValueError(f"Need more than window={window} observations, got {n}")

    records = []
    for t in range(window, n):
        window_returns = returns.iloc[t - window : t]
        actual_return = float(returns.iloc[t].to_numpy() @ w)

        var_dict, es_dict = compute_fn(window_returns, t)

        row = {"date": dates[t], "actual_return": actual_return}
        for alpha in confidence_levels:
            row[f"var_{alpha}"] = var_dict[alpha]
            row[f"es_{alpha}"] = es_dict[alpha]
            row[f"breach_{alpha}"] = actual_return < -var_dict[alpha]
        records.append(row)

    return pd.DataFrame(records).set_index("date")


def backtest_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    window: int = 500,
) -> pd.DataFrame:
    """Rolling backtest of Historical Simulation VaR/ES."""

    def compute_fn(window_returns: pd.DataFrame, _t: int):
        var = historical_var(window_returns, weights, confidence_levels)
        es = historical_expected_shortfall(window_returns, weights, confidence_levels)
        return var, es

    return _run_rolling_engine(returns, weights, compute_fn, window, confidence_levels)


def backtest_parametric(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    window: int = 500,
    cov_method: str = "ewma",
) -> pd.DataFrame:
    """Rolling backtest of Parametric VaR/ES."""

    def compute_fn(window_returns: pd.DataFrame, _t: int):
        var = parametric_var(window_returns, weights, confidence_levels, cov_method)
        es = parametric_expected_shortfall(window_returns, weights, confidence_levels, cov_method)
        return var, es

    return _run_rolling_engine(returns, weights, compute_fn, window, confidence_levels)


def backtest_monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    window: int = 500,
    cov_method: str = "ewma",
    n_sims: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rolling backtest of Monte Carlo VaR/ES.

    n_sims default here (5,000) is deliberately lower than run_var.py's
    single-shot 50,000 -- this function runs a fresh simulation on EVERY
    backtest day (~2000 of them), so the per-call cost matters. 5,000 sims
    is still enough for a stable quantile estimate at the day-to-day level.
    """

    def compute_fn(window_returns: pd.DataFrame, _t: int):
        var = monte_carlo_var(window_returns, weights, confidence_levels, cov_method, n_sims, seed)
        es = monte_carlo_expected_shortfall(
            window_returns, weights, confidence_levels, cov_method, n_sims, seed
        )
        return var, es

    return _run_rolling_engine(returns, weights, compute_fn, window, confidence_levels)


def backtest_filtered_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.01, 0.05),
    window: int = 500,
    refit_frequency: int = 5,
) -> pd.DataFrame:
    """
    Rolling backtest of Filtered Historical Simulation VaR/ES.

    GARCH parameters are refit every `refit_frequency` days (weekly by
    default) via a closure-held cache; on the days in between, the cached
    (omega, alpha, beta) triples are reused and only the conditional
    variance recursion is recomputed on the shifted window -- see
    conditional_variance_from_params() and filtered_historical.py's
    cached_params support.
    """
    cache: dict = {"day_counter": 0, "params": None}

    def compute_fn(window_returns: pd.DataFrame, _t: int):
        cache["day_counter"] += 1
        needs_refit = cache["params"] is None or cache["day_counter"] % refit_frequency == 0

        if needs_refit:
            params = {}
            for col in window_returns.columns:
                fitted = fit_garch11(window_returns[col])
                params[col] = (fitted.omega, fitted.alpha, fitted.beta)
            cache["params"] = params

        var = filtered_historical_var(
            window_returns, weights, confidence_levels, cached_params=cache["params"]
        )
        es = filtered_historical_expected_shortfall(
            window_returns, weights, confidence_levels, cached_params=cache["params"]
        )
        return var, es

    return _run_rolling_engine(returns, weights, compute_fn, window, confidence_levels)
