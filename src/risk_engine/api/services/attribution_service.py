import numpy as np
import pandas as pd
from scipy.stats import norm
from risk_engine.api.errors import UnknownTickerError
from risk_engine.api.schemas import AssetContribution, AttributionRequest, AttributionResponse
from risk_engine.var import get_covariance_matrix


def compute_attribution(returns: pd.DataFrame, req: AttributionRequest) -> AttributionResponse:
    unknown = sorted(set(req.weights) - set(returns.columns))
    if unknown:
        raise UnknownTickerError(unknown)

    tickers = list(returns.columns)
    w = pd.Series(req.weights).reindex(tickers).fillna(0.0).to_numpy()

    cov = get_covariance_matrix(returns, method=req.cov_source)
    sigma_p = float(np.sqrt(w @ cov @ w))

    alpha = round(1 - req.confidence_level, 10)
    z = -norm.ppf(alpha)  # positive: e.g. z ~ 2.326 at alpha=0.01
    portfolio_var = sigma_p * z

    marginal = (cov @ w) / sigma_p  # ∂σ_p/∂w_i, one per asset
    component_var = w * marginal * z  # Euler split, sums exactly to portfolio_var

    contributions = [
        AssetContribution(
            ticker=t,
            weight=float(w[i]),
            marginal_contribution=float(marginal[i]),
            component_var=float(component_var[i]),
            pct_of_var=float(component_var[i] / portfolio_var),
        )
        for i, t in enumerate(tickers)
        if w[i] != 0.0  # skip assets the caller didn't include
    ]

    return AttributionResponse(
        confidence_level=req.confidence_level,
        portfolio_var=portfolio_var,
        contributions=contributions,
        sum_check=float(sum(c.component_var for c in contributions)),
    )
