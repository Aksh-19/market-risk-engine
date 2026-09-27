"""
Data access layer for the Streamlit dashboard.

Currently imports risk_engine directly (no deployed API dependency, by
design — see Phase 6 scope notes). If/when the FastAPI service is
deployed (Phase 7), swap this module's internals to call it over HTTP
instead; app.py should never need to change, since it only calls the
functions defined here, never risk_engine directly.
"""

import numpy as np
import pandas as pd
import streamlit as st
from risk_engine.var import (
    get_covariance_matrix,
    historical_var,
    historical_expected_shortfall,
    parametric_var,
    parametric_expected_shortfall,
    monte_carlo_var,
    monte_carlo_expected_shortfall,
    filtered_historical_var,
    filtered_historical_expected_shortfall,
)
from risk_engine.storage.db import RiskDatabase
from scipy.stats import norm


RETURNS_PATH = "data/processed/returns_matrix.parquet"
DB_PATH = "data/processed/risk_engine.db"


@st.cache_data
def load_returns() -> pd.DataFrame:
    """Cached so the parquet is read once per session, not on every widget interaction."""
    return pd.read_parquet(RETURNS_PATH)


def compute_all_methods(
    returns: pd.DataFrame, weights: dict[str, float], alpha: float = 0.01
) -> pd.DataFrame:
    """Returns a tidy DataFrame: one row per method, VaR/ES columns.
    alpha here is engine convention (tail probability), not 0.99-style."""
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0).to_numpy()
    alphas = (alpha,)

    rows = []
    for label, var_fn, es_fn, kwargs in [
        ("Historical Sim", historical_var, historical_expected_shortfall, {}),
        (
            "Parametric (EWMA)",
            parametric_var,
            parametric_expected_shortfall,
            {"cov_method": "ewma"},
        ),
        (
            "Monte Carlo (EWMA)",
            monte_carlo_var,
            monte_carlo_expected_shortfall,
            {"cov_method": "ewma", "seed": 42},
        ),
        (
            "Filtered Historical Sim",
            filtered_historical_var,
            filtered_historical_expected_shortfall,
            {},
        ),
    ]:
        var = var_fn(returns, w, confidence_levels=alphas, **kwargs)[alpha]
        es = es_fn(returns, w, confidence_levels=alphas, **kwargs)[alpha]
        rows.append({"Method": label, "VaR": var, "ES": es, "ES/VaR": es / var})
    return pd.DataFrame(rows)


@st.cache_data
def load_backtest_results() -> pd.DataFrame:
    """Cached — this is Phase 4's static, already-computed validation output,
    not something that changes with the dashboard's live weight sliders."""
    with RiskDatabase(DB_PATH) as db:
        return db.read_backtest_results()


def compute_attribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
    confidence: float = 0.99,
    cov_method: str = "ewma",
) -> pd.DataFrame:
    """Component VaR via Euler decomposition — same math as
    api/services/attribution_service.py, reimplemented here rather than
    imported from the api package to keep dashboard/ independent of the
    api/ package (no FastAPI dependency needed just to run the dashboard)."""
    tickers = list(returns.columns)
    w = pd.Series(weights).reindex(tickers).fillna(0.0).to_numpy()

    cov = get_covariance_matrix(returns, method=cov_method)
    sigma_p = float(np.sqrt(w @ cov @ w))
    alpha = round(1 - confidence, 10)
    z = -norm.ppf(alpha)
    portfolio_var = sigma_p * z

    marginal = (cov @ w) / sigma_p
    component_var = w * marginal * z

    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": w,
            "Component VaR": component_var,
            "% of Portfolio VaR": component_var / portfolio_var,
        }
    )
