"""
Kupiec (1995) Proportion-of-Failures (POF) test.

Tests whether a VaR model's CLAIMED breach rate (the confidence level
alpha) matches its REALIZED breach rate on out-of-sample data, via a
likelihood-ratio test comparing two Bernoulli-trial hypotheses:

  H0: true breach probability = alpha (model is calibrated)
  H1: true breach probability = observed rate x/n (model is not)

LR_POF = -2 * ln[ (1-alpha)^(n-x) * alpha^x  /  (1-p_hat)^(n-x) * p_hat^x ]

which is asymptotically chi-squared distributed with 1 degree of freedom
under H0. A small p-value (conventionally < 0.05) means the observed
breach rate is unlikely under the claimed alpha -- statistically confident
evidence the model is miscalibrated, not just "looks off by eye."

EDGE CASE: x=0 (zero breaches observed) or x=n (every day breached) makes
p_hat land exactly on the boundary (0 or 1), which makes the denominator's
log term blow up (ln(0)). Handled explicitly below rather than letting it
silently produce inf/nan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass
class KupiecResult:
    """Kupiec POF test result for one method/confidence-level combination."""

    n_obs: int
    n_breaches: int
    alpha_claimed: float
    p_hat_observed: float
    lr_statistic: float
    p_value: float
    reject_calibration: bool  # True if p_value < significance_level

    def summary(self) -> str:
        verdict = "MISCALIBRATED" if self.reject_calibration else "calibrated"
        return (
            f"Kupiec POF: {self.n_breaches}/{self.n_obs} breaches "
            f"({self.p_hat_observed:.4%} observed vs {self.alpha_claimed:.2%} claimed), "
            f"LR={self.lr_statistic:.3f}, p={self.p_value:.4f} -> {verdict}"
        )


def kupiec_pof_test(
    breach_series: pd.Series,
    alpha_claimed: float,
    significance_level: float = 0.05,
) -> KupiecResult:
    """
    Run the Kupiec POF test on a single breach series (boolean, one entry
    per out-of-sample day -- exactly the breach_{alpha} columns produced by
    rolling.py's backtest functions).
    """
    n = len(breach_series)
    x = int(breach_series.sum())
    p_hat = x / n

    if x == 0:
        # No breaches at all: LR reduces to -2n*ln(1-alpha) (the p_hat=0
        # term vanishes cleanly since x*ln(p_hat) -> 0 in the limit).
        lr = -2 * n * np.log(1 - alpha_claimed)
    elif x == n:
        # Every day breached: symmetric edge case.
        lr = -2 * n * np.log(alpha_claimed)
    else:
        log_num = (n - x) * np.log(1 - alpha_claimed) + x * np.log(alpha_claimed)
        log_denom = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        lr = -2 * (log_num - log_denom)

    p_value = float(1 - chi2.cdf(lr, df=1))

    return KupiecResult(
        n_obs=n,
        n_breaches=x,
        alpha_claimed=alpha_claimed,
        p_hat_observed=p_hat,
        lr_statistic=float(lr),
        p_value=p_value,
        reject_calibration=p_value < significance_level,
    )
