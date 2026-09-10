"""
risk_engine
-----------
A market risk modelling toolkit: data ingestion, volatility estimation,
Value-at-Risk / Expected Shortfall calculation, and backtesting.

This __init__.py currently just exposes a version string. As we build out
phases 1-4, this is where we'll expose the public API of the package, e.g.:

    from risk_engine.var import historical_var, parametric_var

so that other code (the FastAPI app, the dashboard, notebooks) can import
clean functions instead of reaching into internal modules.
"""

__version__ = "0.1.0"
