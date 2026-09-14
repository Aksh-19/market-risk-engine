"""
Public API of the volatility modeling layer.

Downstream modules (VaR, backtesting) should import from
risk_engine.volatility, not reach into submodules directly.
"""

from risk_engine.volatility.covariance import ledoit_wolf_shrinkage, sample_covariance
from risk_engine.volatility.ewma import (
    covariance_to_correlation,
    ewma_covariance_matrix,
    ewma_volatility,
    halflife_to_lambda,
    lambda_to_halflife,
)
from risk_engine.volatility.garch import (
    GARCH11Result,
    fit_garch11,
    forecast_variance,
    garch_volatility_series,
)
from risk_engine.volatility.rolling import rolling_volatility

__all__ = [
    "GARCH11Result",
    "covariance_to_correlation",
    "ewma_covariance_matrix",
    "ewma_volatility",
    "fit_garch11",
    "forecast_variance",
    "garch_volatility_series",
    "halflife_to_lambda",
    "ledoit_wolf_shrinkage",
    "lambda_to_halflife",
    "rolling_volatility",
    "sample_covariance",
]
