"""
EWMA (Exponentially Weighted Moving Average) volatility and covariance
estimation — the RiskMetrics standard.

See the derivation notes from our theory session for the full math. The
short version implemented here:

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

which unrolls to an infinite recency-weighted sum of past squared returns.
lambda=0.94 is RiskMetrics' daily default (~11 trading day half-life).

WHY WE EXPOSE half-life AS THE PRIMARY KNOB, NOT JUST lambda
------------------------------------------------------------------
lambda=0.94 vs lambda=0.97 doesn't mean much on its own — the numbers look
close but the implied memory is very different (half-life ~11 days vs ~23
days). Half-life is the interpretable version of the same parameter: "how
many days until a shock's influence is cut in half" is a number you can
reason about and defend in an interview; a bare lambda value isn't. We give
you both directions of the conversion so you can specify whichever is more
natural for the situation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def halflife_to_lambda(halflife: float) -> float:
    """
    Convert a half-life (in periods, e.g. trading days) to the corresponding
    EWMA decay factor lambda.

    Derivation: lambda^h = 0.5  =>  lambda = 0.5^(1/h) = exp(ln(0.5)/h)
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    return float(np.exp(np.log(0.5) / halflife))


def lambda_to_halflife(lambda_: float) -> float:
    """Inverse of halflife_to_lambda: h = ln(0.5) / ln(lambda)."""
    if not (0 < lambda_ < 1):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    return float(np.log(0.5) / np.log(lambda_))


def ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    min_periods: int = 30,
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.Series:
    """
    Estimate conditional (time-varying) volatility via EWMA.

    Parameters
    ----------
    returns : daily log returns for a single asset.
    lambda_ : decay factor. 0.94 = RiskMetrics daily default (~11-day half-life).
    min_periods : size of the burn-in window used to seed sigma_0^2 with the
        simple sample variance. See "initialization" note in the derivation:
        because weights decay fast, this seed's influence vanishes within a
        couple months, but the FIRST min_periods estimates are less
        trustworthy and are therefore not emitted at all (NaN) rather than
        silently returned as if they were fully-converged estimates.
    annualize : if True, scale by sqrt(trading_days) to express volatility
        in annualized terms (the convention for quoting "20% vol" style
        numbers) rather than raw daily standard deviation.

    Returns
    -------
    pd.Series of sigma_t (volatility, i.e. already sqrt'd — NOT variance),
    same index as `returns`, with the first `min_periods` entries as NaN.
    """
    if not (0 < lambda_ < 1):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    if len(returns) <= min_periods:
        raise ValueError(
            f"Need more than min_periods={min_periods} observations, got {len(returns)}"
        )

    r = returns.to_numpy()
    n = len(r)
    variance = np.full(n, np.nan)

    # Seed: simple sample variance of the burn-in window. This is a
    # deliberate, documented approximation for the intractable "infinite
    # history" the closed-form recursion technically wants.
    variance[min_periods - 1] = np.var(r[:min_periods])

    for t in range(min_periods, n):
        variance[t] = lambda_ * variance[t - 1] + (1 - lambda_) * r[t - 1] ** 2

    vol = np.sqrt(variance)
    if annualize:
        vol = vol * np.sqrt(trading_days)

    return pd.Series(vol, index=returns.index, name=f"ewma_vol_lambda{lambda_}")


def ewma_covariance_matrix(
    returns: pd.DataFrame,
    lambda_: float = 0.94,
    min_periods: int = 30,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Multi-asset extension: estimate the full time-varying covariance MATRIX
    via the same recursive idea, generalized from scalars to matrices:

        Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_{t-1} r_{t-1}^T

    where r_{t-1} is the (n_assets,) vector of returns and r_{t-1} r_{t-1}^T
    is its outer product — the matrix generalization of "squared return".
    The diagonal of Sigma_t recovers exactly the single-asset ewma_volatility
    result (squared); the off-diagonals are the EWMA covariances, which is
    what a Monte Carlo VaR simulation in Phase 3 needs to draw correlated
    returns rather than treating every asset as independent.

    Returns
    -------
    dict mapping each date (from min_periods onward) to its (n_assets,
    n_assets) covariance matrix. A dict keyed by date, rather than a single
    3D array, keeps it easy to look up "the covariance matrix as of date X"
    which is exactly how Phase 3's rolling backtests will consume this.
    """
    if not (0 < lambda_ < 1):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")

    R = returns.to_numpy()
    n_obs, _n_assets = R.shape

    if n_obs <= min_periods:
        raise ValueError(f"Need more than min_periods={min_periods} observations, got {n_obs}")

    # Seed: covariance matrix of the burn-in window (rowvar=False because
    # our rows are observations, columns are assets). bias=True forces
    # ddof=0 (divide by N, not N-1) to match np.var()'s default used for
    # the single-asset seed in ewma_volatility -- without this the diagonal
    # of this matrix would NOT equal the scalar EWMA variance even though
    # they're supposed to be the same calculation, just generalized to
    # matrix form. This was caught by test_covariance_diagonal_matches_scalar_ewma.
    sigma = np.cov(R[:min_periods], rowvar=False, bias=True)
    # np.cov collapses to a scalar when n_assets == 1; guard for that edge case.
    sigma = np.atleast_2d(sigma)

    result: dict[pd.Timestamp, np.ndarray] = {returns.index[min_periods - 1]: sigma.copy()}

    for t in range(min_periods, n_obs):
        r_prev = R[t - 1].reshape(-1, 1)  # column vector, shape (n_assets, 1)
        outer = r_prev @ r_prev.T  # shape (n_assets, n_assets)
        sigma = lambda_ * sigma + (1 - lambda_) * outer
        result[returns.index[t]] = sigma.copy()

    return result


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """
    Normalize a covariance matrix into a correlation matrix:
    corr_ij = cov_ij / (sigma_i * sigma_j)

    Useful for sanity-checking a covariance matrix visually — correlations
    are bounded in [-1, 1] and immediately readable, whereas raw covariance
    magnitudes depend on each asset's scale and aren't directly comparable
    across, say, a bond ETF and a mega-cap stock.
    """
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    np.fill_diagonal(corr, 1.0)  # guard against floating point drift off exactly 1.0
    return corr
