"""
Augmented Dickey-Fuller (ADF) test for stationarity.

WHY THIS TEST MATTERS FOR THIS PROJECT
------------------------------------------
Every model built so far -- EWMA, GARCH, the returns matrix itself -- rests
on an assumption we've asserted but never formally verified: that RETURNS
are (approximately) stationary while PRICES are not. Stationarity means a
series' statistical properties (mean, variance) don't drift over time in a
structural way. This isn't a technicality -- it's the actual justification
for why we model returns instead of prices at all. A GARCH model fit
directly to price levels would be nonsensical: price levels wander
indefinitely (a "random walk"), so there's no fixed variance for GARCH to
even be modeling.

THE UNIT ROOT / RANDOM WALK CONNECTION
------------------------------------------
A price series behaving like a random walk means: P_t = P_{t-1} + epsilon_t.
Rearranged: P_t - P_{t-1} = epsilon_t, i.e. today's price is yesterday's
price plus pure noise -- there's no "pull back" toward any fixed level.
This is called having a UNIT ROOT. A stationary series, by contrast, DOES
get pulled back toward a fixed mean after a shock (formally: y_t = phi*y_{t-1}
+ epsilon_t with |phi| < 1). The ADF test is precisely a test of "does phi
equal 1 (unit root, non-stationary) or is it less than 1 (stationary)?"

THE REGRESSION (this is what adf_test() actually estimates)
------------------------------------------------------------------
    delta_y_t = alpha + beta * y_{t-1} + sum_{i=1}^{p} gamma_i * delta_y_{t-i} + e_t

  - beta is the parameter of interest. H0: beta = 0 (unit root / non-stationary).
    H1: beta < 0 (stationary -- y_{t-1} pulls delta_y_t back toward zero).
  - The gamma_i lagged-difference terms are the "augmented" part -- they
    soak up any short-run autocorrelation in delta_y so it doesn't
    contaminate the test of beta. p (lag order) is chosen automatically here
    via BIC (Schwarz criterion) over a range suggested by Schwert's (1989)
    rule of thumb, rather than fixed arbitrarily.
  - The test statistic is beta_hat / SE(beta_hat) -- looks like an ordinary
    t-statistic, but it does NOT follow a standard t-distribution under H0
    (because y_{t-1} under a unit root isn't stationary, breaking the usual
    assumptions). It follows the Dickey-Fuller distribution instead, which
    is why critical values below are NOT the familiar +-1.96 you'd expect
    from a normal-based test.

CRITICAL VALUES USED HERE
-----------------------------
We use the standard asymptotic Dickey-Fuller critical values for the
"constant, no trend" case (MacKinnon 1994/2010): approximately -3.43 (1%),
-2.86 (5%), -2.57 (10%). These are ASYMPTOTIC (large-sample) values --
accurate for the ~2,500-observation daily series this project uses, but a
production statistics package (e.g. statsmodels) additionally applies a
finite-sample response-surface correction and reports an exact p-value
rather than a fixed table lookup. We report the fixed-table conclusion here
and note this limitation explicitly rather than fabricate false precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ADFResult:
    test_statistic: float
    lags_used: int
    n_obs: int
    critical_values: dict[str, float]
    is_stationary_5pct: bool

    def summary(self) -> str:
        verdict = "STATIONARY" if self.is_stationary_5pct else "NON-STATIONARY"
        return (
            f"ADF statistic: {self.test_statistic:.4f} (lags={self.lags_used}, n={self.n_obs}) | "
            f"critical values: 1%={self.critical_values['1%']}, "
            f"5%={self.critical_values['5%']}, 10%={self.critical_values['10%']} | "
            f"conclusion (5% level): {verdict}"
        )


# Asymptotic Dickey-Fuller critical values, constant-no-trend case (MacKinnon).
_CRITICAL_VALUES = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def _schwert_max_lag(n: int) -> int:
    """Schwert (1989) rule of thumb for a reasonable maximum lag to consider
    during automatic lag-order selection, scaling slowly with sample size."""
    return int(np.ceil(12 * (n / 100) ** 0.25))


def _fit_adf_regression(y: np.ndarray, lag: int) -> tuple[float, float, int]:
    """
    Fit the ADF regression at a specific lag order and return
    (beta_hat, se_beta_hat, bic) for that lag.
    """
    dy = np.diff(y)
    n_total = len(dy)
    nobs = n_total - lag

    y_lag1 = y[lag : len(y) - 1]  # y_{t-1}, aligned to dy[lag:]
    columns = [np.ones(nobs), y_lag1]
    for i in range(1, lag + 1):
        columns.append(dy[lag - i : n_total - i])  # delta_y_{t-i}
    X = np.column_stack(columns)
    Y = dy[lag:]

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    k = X.shape[1]
    sigma2 = (resid @ resid) / (nobs - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))

    bic = nobs * np.log(sigma2) + k * np.log(nobs)
    return float(beta[1]), float(se[1]), float(bic)


def adf_test(series: pd.Series, max_lag: int | None = None) -> ADFResult:
    """
    Run the Augmented Dickey-Fuller test on a series (price levels OR
    returns -- the same function works on either, that's the point: you run
    it on both and expect opposite conclusions).

    max_lag: if None, uses Schwert's rule of thumb as an upper bound, then
    selects the actual lag order that minimizes BIC across that range --
    an automatic, data-driven choice rather than a fixed guess.
    """
    y = series.dropna().to_numpy()
    n = len(y)
    if max_lag is None:
        max_lag = _schwert_max_lag(n)

    best_lag, best_tstat, best_bic, best_nobs = None, None, None, None
    for lag in range(0, max_lag + 1):
        beta_hat, se_beta_hat, bic = _fit_adf_regression(y, lag)
        tstat = beta_hat / se_beta_hat
        if best_bic is None or bic < best_bic:
            best_lag, best_tstat, best_bic = lag, tstat, bic
            best_nobs = n - 1 - lag

    return ADFResult(
        test_statistic=best_tstat,
        lags_used=best_lag,
        n_obs=best_nobs,
        critical_values=dict(_CRITICAL_VALUES),
        is_stationary_5pct=best_tstat < _CRITICAL_VALUES["5%"],
    )


def stationarity_report(prices: pd.Series, returns: pd.Series, ticker: str) -> str:
    """
    Run the ADF test on both price levels and returns for one asset, and
    print a side-by-side conclusion. Expected (and the entire justification
    for this project's return-based modeling approach): prices come back
    non-stationary, returns come back stationary.
    """
    price_result = adf_test(prices)
    return_result = adf_test(returns)

    lines = [
        f"=== Stationarity report: {ticker} ===",
        f"Price levels: {price_result.summary()}",
        f"Log returns:  {return_result.summary()}",
    ]
    if price_result.is_stationary_5pct:
        lines.append(
            "  NOTE: price levels tested as stationary -- unexpected for a "
            "typical asset price series; worth double-checking the input data."
        )
    if not return_result.is_stationary_5pct:
        lines.append(
            "  WARNING: returns tested as NON-stationary -- this would "
            "undermine the modeling assumptions used throughout this project "
            "and is worth investigating before trusting downstream results."
        )
    return "\n".join(lines)
