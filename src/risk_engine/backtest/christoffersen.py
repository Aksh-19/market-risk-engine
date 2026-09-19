"""
Christoffersen (1998) independence test, plus the combined conditional
coverage test.

Kupiec's POF test only checks the COUNT of breaches. It's blind to
CLUSTERING -- a model could breach exactly the claimed rate but have every
breach concentrated in one crisis month, meaning it was reliably fine
day-to-day but went blind exactly when risk spiked. That's the single most
dangerous failure mode for a risk model, and Kupiec cannot detect it.

MECHANISM: model the breach sequence (0/1 per day) as a two-state Markov
chain. Under independence, whether you breached TODAY should not affect
the probability of breaching TOMORROW -- so pi_01 (breach | no breach
yesterday) should equal pi_11 (breach | breach yesterday). If breaches
cluster, pi_11 >> pi_01: a breach today makes tomorrow's breach much more
likely, which is exactly the volatility-clustering signature Phase 2's
EWMA/GARCH work was all about -- except here it shows up as a MODEL
FAILURE rather than a market feature being correctly captured.

LR_ind is chi-squared(1) under independence. The combined test
LR_cc = LR_POF + LR_ind is chi-squared(2), testing BOTH correct coverage
AND independence at once -- the number regulators actually use, since a
model can fail on either dimension separately or both together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from risk_engine.backtest.kupiec import kupiec_pof_test


@dataclass
class ChristoffersenResult:
    """Christoffersen independence test result, plus combined coverage."""

    n_obs: int
    n00: int  # no breach -> no breach
    n01: int  # no breach -> breach
    n10: int  # breach -> no breach
    n11: int  # breach -> breach
    pi_01: float
    pi_11: float
    lr_independence: float
    p_value_independence: float
    reject_independence: bool
    lr_combined: float
    p_value_combined: float
    reject_combined: bool

    def summary(self) -> str:
        clustering = "CLUSTERED" if self.reject_independence else "independent"
        combined = "FAILS combined" if self.reject_combined else "passes combined"
        return (
            f"Christoffersen: pi_01={self.pi_01:.4f}, pi_11={self.pi_11:.4f} "
            f"(n01={self.n01}, n11={self.n11}), "
            f"LR_ind={self.lr_independence:.3f} p={self.p_value_independence:.4f} -> {clustering}; "
            f"LR_cc={self.lr_combined:.3f} p={self.p_value_combined:.4f} -> {combined}"
        )


def _safe_log_term(count: int, prob: float) -> float:
    """count * ln(prob), defined as 0 when count=0 even if prob=0 (the
    limit x*ln(x) -> 0 as x -> 0), avoiding a -inf * 0 = nan crash."""
    if count == 0:
        return 0.0
    return count * np.log(prob)


def christoffersen_independence_test(
    breach_series: pd.Series,
    alpha_claimed: float,
    significance_level: float = 0.05,
) -> ChristoffersenResult:
    """
    Run Christoffersen's independence test (plus the combined conditional
    coverage test) on a breach series. alpha_claimed is needed to also
    compute the Kupiec component for the combined LR_cc statistic.
    """
    breaches = breach_series.to_numpy().astype(int)
    n = len(breaches)

    # Count the four possible yesterday -> today transitions.
    prev = breaches[:-1]
    curr = breaches[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_pooled = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Numerator: single pooled probability (independence assumed).
    log_num = _safe_log_term(n00 + n10, 1 - pi_pooled) + _safe_log_term(n01 + n11, pi_pooled)
    # Denominator: separate pi_01 / pi_11 (dependence allowed).
    log_denom = (
        _safe_log_term(n00, 1 - pi_01)
        + _safe_log_term(n01, pi_01)
        + _safe_log_term(n10, 1 - pi_11)
        + _safe_log_term(n11, pi_11)
    )

    lr_ind = -2 * (log_num - log_denom)
    p_value_ind = float(1 - chi2.cdf(lr_ind, df=1))

    kupiec_result = kupiec_pof_test(breach_series, alpha_claimed, significance_level)
    lr_cc = kupiec_result.lr_statistic + lr_ind
    p_value_cc = float(1 - chi2.cdf(lr_cc, df=2))

    return ChristoffersenResult(
        n_obs=n,
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi_01=pi_01,
        pi_11=pi_11,
        lr_independence=float(lr_ind),
        p_value_independence=p_value_ind,
        reject_independence=p_value_ind < significance_level,
        lr_combined=float(lr_cc),
        p_value_combined=p_value_cc,
        reject_combined=p_value_cc < significance_level,
    )
