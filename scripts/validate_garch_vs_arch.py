"""
Cross-validate our from-scratch GARCH(1,1) implementation against the
`arch` package (Kevin Sheppard's library -- the de facto standard Python
GARCH implementation) on the real asset universe.

    python scripts/validate_garch_vs_arch.py

This is the GARCH equivalent of what we already did for Ledoit-Wolf vs.
sklearn: independent, external validation on real data, not just internal
tests against our own simulated ground truth.

To match our implementation's assumptions exactly (so any difference is
attributable to the OPTIMIZER/estimation, not to differing model
specifications):
  - mean='Zero'  -- our fit_garch11() never estimates a mean, it operates
    directly on raw returns assuming E[r_t] = 0. arch's default is a
    constant-mean model, which would NOT be an apples-to-apples comparison.
  - vol='GARCH', p=1, q=1  -- plain GARCH(1,1), no asymmetry/leverage terms.
  - dist='normal'  -- matches our Gaussian likelihood assumption.

NOTE: this script was written and reasoned through carefully against arch's
documented API, but could not be executed in the sandbox used to build this
project (no network access to install the `arch` package there). Run it and
report back what you see -- if there's an API mismatch (arch's interface
has changed slightly across versions before), we'll debug it together
rather than assume it's correct on faith.
"""

from __future__ import annotations

import pandas as pd
from arch import arch_model

from risk_engine.volatility import fit_garch11


def main(returns_path: str) -> None:
    returns = pd.read_parquet(returns_path)

    print(f"{'Ticker':<6} {'Param':<8} {'Ours':>12} {'arch':>12} {'Abs diff':>12}")
    print("-" * 54)

    for ticker in returns.columns:
        r = returns[ticker]

        ours = fit_garch11(r)

        # Same fix as our own fit_garch11(): pre-scale returns by 100 before
        # fitting (arch's own DataScaleWarning independently flags the exact
        # numerical fragility we hit and fixed ourselves earlier -- two
        # separate codebases agreeing this is a real issue, not one we
        # invented). rescale=False stops arch from doing its own internal
        # auto-rescaling on top of ours, so we know exactly what scale we're
        # working in and can convert back unambiguously.
        SCALE = 100.0
        r_scaled = r * SCALE
        am = arch_model(r_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        arch_fit = am.fit(disp="off")

        arch_omega = arch_fit.params["omega"] / (
            SCALE**2
        )  # alpha, beta are scale-invariant (see ewma/garch.py docstrings)
        arch_alpha = arch_fit.params["alpha[1]"]
        arch_beta = arch_fit.params["beta[1]"]

        # arch internally works in whatever scale the input is in but
        # reports omega back in THAT scale -- since we pass raw (unscaled)
        # returns here, this should be directly comparable to our omega
        # without any manual rescaling on our end.
        rows = [
            ("omega", ours.omega, arch_omega),
            ("alpha", ours.alpha, arch_alpha),
            ("beta", ours.beta, arch_beta),
        ]
        for param, ours_val, arch_val in rows:
            diff = abs(ours_val - arch_val)
            print(f"{ticker:<6} {param:<8} {ours_val:>12.6g} {arch_val:>12.6g} {diff:>12.2e}")

        # Persistence (alpha+beta) is usually much better identified than
        # alpha and beta individually -- the likelihood surface often has a
        # flat "ridge" along combinations with similar alpha+beta, so two
        # different optimizers can land at different points on that ridge
        # while still agreeing closely on the number that actually drives
        # forecasts (see forecast_variance()'s mean-reversion formula).
        our_persist = ours.alpha + ours.beta
        arch_persist = arch_alpha + arch_beta
        print(
            f"{ticker:<6} {'persist':<8} {our_persist:>12.6g} {arch_persist:>12.6g} "
            f"{abs(our_persist - arch_persist):>12.2e}"
        )
        print()


if __name__ == "__main__":
    main(returns_path="data/processed/returns_matrix.parquet")
