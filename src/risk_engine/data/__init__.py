"""
Public API of the data ingestion layer.

Downstream modules (volatility, VaR, backtesting) should import from
`risk_engine.data`, not reach into submodules directly — this keeps the
internal file layout free to change without breaking Phase 2/3 code.
"""

from risk_engine.data.config import DataConfig
from risk_engine.data.loader import DataLoader
from risk_engine.data.returns import (
    build_returns_matrix,
    compute_log_returns,
    compute_simple_returns,
)
from risk_engine.data.validators import ValidationReport, validate_price_series

__all__ = [
    "DataConfig",
    "DataLoader",
    "build_returns_matrix",
    "compute_log_returns",
    "compute_simple_returns",
    "ValidationReport",
    "validate_price_series",
]
