"""
Single entry point: "give me a portfolio covariance matrix Sigma, sourced
however I ask" -- used by Parametric and Monte Carlo VaR.

  "ewma"         -- time-varying, DEFAULT. Latest snapshot from Phase 2's
                    ewma_covariance_matrix(): "the market's covariance
                    structure as of today," which is what a forward-looking
                    1-day VaR should condition on.

  "ledoit_wolf"  -- full-sample shrunk covariance. Stable, well-conditioned,
                    but NOT time-varying. Included as a comparison point and
                    because shrinkage is the standard answer once asset
                    count grows relative to history.

  "garch_hybrid" -- Sigma = D_garch @ R_ewma @ D_garch. D_garch is a diagonal
                    matrix of each asset's LATEST GARCH-implied volatility
                    (Phase 2 showed SPY/AAPL react ~3x harder to shocks than
                    EWMA assumes; TLT/GLD closely agree with EWMA) -- so this
                    lets each asset's own volatility model drive its
                    variance contribution. R_ewma is EWMA's correlation
                    matrix, since GARCH here is univariate-only and has no
                    native cross-asset covariance. A simplified, static
                    cousin of DCC-GARCH.

All three return a plain (n_assets, n_assets) np.ndarray ordered to match
`returns.columns` -- the only shape portfolio_variance() and Monte Carlo's
Cholesky step need, so the three sources are fully interchangeable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.volatility.ewma import ewma_covariance_matrix, covariance_to_correlation
from risk_engine.volatility.covariance import ledoit_wolf_shrinkage
from risk_engine.volatility.garch import fit_garch11


def get_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "ewma",
    lambda_: float = 0.94,
    min_periods: int = 30,
) -> np.ndarray:
    if method == "ewma":
        return _ewma_latest(returns, lambda_, min_periods)
    elif method == "ledoit_wolf":
        shrunk, _shrinkage = ledoit_wolf_shrinkage(returns)
        return shrunk
    elif method == "garch_hybrid":
        return _garch_hybrid(returns, lambda_, min_periods)
    else:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            "Expected one of: 'ewma', 'ledoit_wolf', 'garch_hybrid'."
        )


def _ewma_latest(returns: pd.DataFrame, lambda_: float, min_periods: int) -> np.ndarray:
    cov_by_date = ewma_covariance_matrix(returns, lambda_=lambda_, min_periods=min_periods)
    return cov_by_date[returns.index[-1]]


def _garch_hybrid(returns: pd.DataFrame, lambda_: float, min_periods: int) -> np.ndarray:
    latest_vols = np.array(
        [np.sqrt(fit_garch11(returns[col]).conditional_variance[-1]) for col in returns.columns]
    )
    D = np.diag(latest_vols)

    ewma_cov = _ewma_latest(returns, lambda_, min_periods)
    R = covariance_to_correlation(ewma_cov)

    return D @ R @ D
