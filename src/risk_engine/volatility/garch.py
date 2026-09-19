"""
GARCH(1,1) volatility model, fit via Maximum Likelihood Estimation.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

Unlike EWMA (fixed lambda, no fitting), all three parameters here are
estimated from the data by maximizing the Gaussian log-likelihood:

    ln L = -0.5 * sum_t [ ln(2*pi) + ln(sigma_t^2) + r_t^2 / sigma_t^2 ]

See the derivation notes from our theory discussion for the full reasoning.
Key structural facts encoded in this implementation:

  - Stationarity requires alpha + beta < 1. We enforce this as an explicit
    optimizer constraint, not just a post-hoc check -- an unconstrained
    optimizer can easily wander into alpha+beta >= 1 territory (explosive,
    non-stationary variance) while chasing a slightly higher likelihood on
    a finite sample, which would give you a technically "fitted" model that
    doesn't have a well-defined long-run variance at all.
  - The unconditional (long-run) variance omega / (1 - alpha - beta) is what
    conditional variance reverts to after a shock -- this is the property
    that distinguishes GARCH from EWMA (which has no such reversion target).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GARCH11Result:
    """Fitted GARCH(1,1) parameters plus diagnostics."""

    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    converged: bool
    conditional_variance: np.ndarray = field(repr=False)

    @property
    def persistence(self) -> float:
        """alpha + beta. Must be < 1 for stationarity; closer to 1 means
        shocks decay slowly (long memory in volatility)."""
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """omega / (1 - alpha - beta): the long-run variance level that
        conditional variance mean-reverts to. Undefined (returns inf) if
        the fit is non-stationary."""
        denom = 1 - self.persistence
        return self.omega / denom if denom > 1e-8 else float("inf")

    def summary(self) -> str:
        return (
            f"GARCH(1,1): omega={self.omega:.6g}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, persistence={self.persistence:.4f}, "
            f"long-run vol (annualized)={np.sqrt(self.unconditional_variance*252):.4f}, "
            f"log-likelihood={self.log_likelihood:.2f}, converged={self.converged}"
        )


def _conditional_variance(
    returns: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """
    Run the GARCH(1,1) recursion forward given a parameter triple.

    Seed: sigma_0^2 is set to the unconditional variance implied by these
    very parameters (omega / (1-alpha-beta)) when stationary, falling back
    to the sample variance otherwise. This is standard practice -- it means
    the recursion starts "at rest" rather than at an arbitrary value, which
    matters because during optimization the optimizer tries many candidate
    (omega, alpha, beta) triples and each one needs a principled seed.
    """
    n = len(returns)
    sigma2 = np.empty(n)

    persistence = alpha + beta
    if persistence < 0.999:
        sigma2[0] = omega / (1 - persistence)
    else:
        sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    return sigma2


def _negative_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    sigma2 = _conditional_variance(returns, omega, alpha, beta)
    sigma2 = np.maximum(sigma2, 1e-12)  # guard against log(0) / division by 0
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -ll  # minimize the negative = maximize the likelihood


def fit_garch11(returns: pd.Series, scale: float = 100.0) -> GARCH11Result:
    """
    Fit a GARCH(1,1) model to a return series via MLE.

    WHY `scale` EXISTS -- A REAL NUMERICAL BUG THIS FIX ADDRESSES
    -------------------------------------------------------------------
    Daily log returns are tiny (~0.01), so omega -- which is on the order
    of variance, i.e. return^2 -- ends up around 1e-6, while alpha and beta
    live on the order of 0.01-1.0. That's six orders of magnitude apart in
    the SAME parameter vector. scipy's finite-difference gradient uses one
    step size for every parameter, so at that scale mismatch it effectively
    can't detect how the likelihood changes with omega -- the optimizer can
    (and in early testing here, silently did) terminate after only 4
    function evaluations, report "success", and just hand back its own
    starting guess unchanged. That's not a fit; it's a no-op wearing a
    success flag.

    Fix: multiply returns by `scale` (100, i.e. treat returns as
    percentages) before fitting. alpha and beta are scale-invariant (the
    recursion sigma_t^2 = omega + alpha*r^2 + beta*sigma_{t-1}^2 is
    homogeneous of degree 2 in returns/variance), so only omega needs to be
    divided by scale^2 to convert back. This single change took the
    optimizer from 4 function evaluations (stuck) to 49 (actually
    searching) on the exact same data in testing -- verify this still holds
    if you ever change the starting values or bounds below.

    Starting values: alpha=0.05, beta=0.90 are typical empirical GARCH
    estimates for daily equity returns (persistence ~0.95). omega is
    initialized so the *implied* unconditional variance roughly matches the
    (scaled) sample variance.

    Constraints enforced during optimization (not just checked after):
      - omega > 0        (variance can't be zero or negative)
      - alpha, beta >= 0  (both must be non-negative for sigma_t^2 >= 0 always)
      - alpha + beta < 1  (stationarity -- enforced via inequality constraint)
    """
    r = returns.to_numpy()
    r_scaled = r * scale
    sample_var_scaled = np.var(r_scaled)

    alpha0, beta0 = 0.05, 0.90
    omega0_scaled = sample_var_scaled * (1 - alpha0 - beta0)

    bounds = [(1e-10, None), (0.0, 1.0), (0.0, 1.0)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2])}]

    result = minimize(
        _negative_log_likelihood,
        x0=[omega0_scaled, alpha0, beta0],
        args=(r_scaled,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    omega_scaled, alpha, beta = result.x
    omega = omega_scaled / (scale**2)  # undo the scaling -- alpha, beta need no correction
    sigma2 = _conditional_variance(r, omega, alpha, beta)  # recompute on UNSCALED returns

    # Log-likelihood must also be evaluated on the unscaled series for a
    # meaningful, comparable number (e.g. against test_garch_likelihood_beats_constant_variance_on_clustered_data).
    ll = -_negative_log_likelihood(np.array([omega, alpha, beta]), r)

    return GARCH11Result(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        log_likelihood=float(ll),
        converged=bool(result.success),
        conditional_variance=sigma2,
    )


def conditional_variance_from_params(
    returns: pd.Series, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """
    Public wrapper around the internal GARCH(1,1) recursion, exposed so
    callers can reuse ALREADY-FITTED parameters without re-running MLE
    optimization. This is what makes a "refit weekly, reuse daily" backtest
    schedule computationally tractable -- see rolling.py's
    filtered-historical backtest, which refits every 5 days but still needs
    a fresh conditional variance recursion every single day as the rolling
    window shifts.
    """
    return _conditional_variance(returns.to_numpy(), omega, alpha, beta)


def garch_volatility_series(returns: pd.Series, fitted: GARCH11Result) -> pd.Series:
    """Wrap the fitted conditional variance back into a labeled, sqrt'd
    volatility series aligned to the original return series' index."""
    return pd.Series(np.sqrt(fitted.conditional_variance), index=returns.index, name="garch_vol")


def forecast_variance(fitted: GARCH11Result, horizon: int) -> np.ndarray:
    """
    Forecast conditional variance h steps ahead from the last observed
    variance in the fitted series.

    Derivation: taking expectations of the GARCH recursion forward,
    E[sigma_{t+h}^2] = sigma_inf^2 + (alpha+beta)^h * (sigma_t^2 - sigma_inf^2)

    This is the mean-reversion property in explicit action: as h -> infinity,
    (alpha+beta)^h -> 0 (since persistence < 1), so the forecast converges
    to the unconditional variance regardless of where today's variance
    happens to sit. EWMA has no equivalent formula -- its forecast for any
    horizon is just flat at today's estimate, forever, because it has no
    long-run level to revert to.
    """
    sigma2_inf = fitted.unconditional_variance
    sigma2_last = fitted.conditional_variance[-1]
    persistence = fitted.persistence

    h = np.arange(1, horizon + 1)
    return sigma2_inf + (persistence**h) * (sigma2_last - sigma2_inf)
