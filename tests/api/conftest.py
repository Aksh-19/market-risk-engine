import numpy as np
import pandas as pd
import pytest
from risk_engine.storage.db import RiskDatabase


@pytest.fixture
def tmp_returns_parquet(tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=(500, 4)),
        index=dates,
        columns=["SPY", "AAPL", "TLT", "GLD"],
    )
    path = tmp_path / "returns_matrix.parquet"
    returns.to_parquet(path)
    return path


@pytest.fixture
def tmp_risk_db(tmp_path):
    db_path = tmp_path / "risk_engine.db"
    db = RiskDatabase(db_path)
    results = pd.DataFrame(
        [
            {
                "method": "Historical Sim",
                "cov_source": "-",
                "confidence_level": 0.01,
                "var": 0.0199,
                "es": 0.0280,
            },
            {
                "method": "Parametric",
                "cov_source": "ewma",
                "confidence_level": 0.01,
                "var": 0.0162,
                "es": 0.0185,
            },
        ]
    )
    db.write_var_results(results, run_date="2026-01-01", portfolio="equal_weight_4asset")
    db.close()
    return db_path
