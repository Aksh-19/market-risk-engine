"""
Sample covariance and Ledoit-Wolf shrinkage estimation.

WHY PLAIN SAMPLE COVARIANCE IS RISKIER THAN IT LOOKS
---------------------------------------------------------
The sample covariance matrix S = (1/n) X'X (X = demeaned returns) is the
obvious, unbiased-in-expectation estimator -- so why not just use it? The
problem shows up specifically when the number of assets (p) isn't tiny
relative to the number of observations (n). Random matrix theory
(Marchenko-Pastur) tells us that with finite n/p, sample covariance
systematically DISTORTS the eigenvalue spectrum: the largest eigenvalues
are biased upward, the smallest biased downward, relative to the true
underlying covariance. This matters enormously in practice because
portfolio optimization and Monte Carlo VaR (Phase 3) both need to INVERT
the covariance matrix -- and inverting a matrix with artificially
compressed small eigenvalues amplifies estimation noise into wildly unstable
portfolio weights or simulated scenarios. This isn't a hypothetical: it's
the textbook reason "optimal" mean-variance portfolios built on raw sample
covariance are notoriously unstable out-of-sample.

WHAT LEDOIT-WOLF SHRINKAGE DOES ABOUT IT
---------------------------------------------
Ledoit & Wolf (2004) proposed shrinking the sample covariance S toward a
simple, well-conditioned TARGET matrix F = mu*I (mu = average variance
across assets, i.e. "every asset has the same variance, zero correlation")
-- an average of a noisy-but-unbiased estimator (S) and a biased-but-stable
one (F):

    S_shrunk = delta * F + (1 - delta) * S

The key contribution isn't the averaging idea itself (that's old) -- it's
that delta (the shrinkage intensity) has a CLOSED-FORM formula, estimated
directly from the data, that's provably optimal in the sense of minimizing
expected squared Frobenius-norm distance to the (unobservable) true
covariance matrix. No cross-validation or arbitrary tuning parameter needed.

This implementation follows the closed-form given in Ledoit & Wolf (2004)
for a scaled-identity target, and is cross-validated against sklearn's
`ledoit_wolf` (a widely used reference implementation of the same paper) in
the test suite -- the same "validate a from-scratch implementation against
an industry-standard library" pattern used for GARCH vs. the `arch` package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Plain sample covariance matrix (MLE / biased estimator, ddof=0 -- this
    matches the convention used by the Ledoit-Wolf formula below, which was
    derived under this normalization).
    """
    X = returns.to_numpy()
    X = X - X.mean(axis=0)
    n = X.shape[0]
    return (X.T @ X) / n


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Estimate the Ledoit-Wolf shrunk covariance matrix, shrinking toward a
    scaled-identity target F = mu*I.

    Returns
    -------
    (shrunk_covariance, shrinkage_intensity)
        shrinkage_intensity is in [0, 1]: 0 means "trust the sample
        covariance entirely" (no shrinkage applied), 1 means "ignore the
        sample covariance, use the structured target entirely." In
        practice, with a small number of assets relative to a long history
        (our case: 4 assets, ~2500 observations), expect a SMALL shrinkage
        intensity -- there's plenty of data to estimate 4x4 covariance well,
        so shrinkage has little work to do. The technique matters far more
        as the number of assets grows into the dozens/hundreds relative to
        available history.
    """
    X = returns.to_numpy()
    X = X - X.mean(axis=0)
    n, p = X.shape

    S = (X.T @ X) / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)

    # b_bar^2: average squared Frobenius distance between each individual
    # observation's outer product (x_t x_t') and the sample covariance S --
    # this estimates how much sampling noise is in S itself.
    b_bar_sq = 0.0
    for t in range(n):
        x_t = X[t : t + 1].T  # column vector, shape (p, 1)
        outer = x_t @ x_t.T
        b_bar_sq += np.sum((outer - S) ** 2)
    b_bar_sq /= n**2

    d_sq = np.sum((S - F) ** 2)  # squared Frobenius distance from S to the target
    b_sq = min(b_bar_sq, d_sq)
    shrinkage = b_sq / d_sq if d_sq > 1e-15 else 0.0

    shrunk = shrinkage * F + (1 - shrinkage) * S
    return shrunk, float(shrinkage)
