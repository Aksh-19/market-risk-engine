"""
Public API of the volatility modeling layer.

Downstream modules (VaR, backtesting) should import from
risk_engine.volatility, not reach into submodules directly.
"""

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

__all__ = [
    "ewma_volatility",
    "ewma_covariance_matrix",
    "covariance_to_correlation",
    "halflife_to_lambda",
    "lambda_to_halflife",
    "GARCH11Result",
    "fit_garch11",
    "forecast_variance",
    "garch_volatility_series",
]
