import numpy as np
import pandas as pd
import pytest
from risk_engine.storage.db import RiskDatabase
from fastapi.testclient import TestClient
from risk_engine.api.main import create_app


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

    var_rows = pd.DataFrame(
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
    db.write_var_results(var_rows, run_date="2026-01-01", portfolio="equal_weight_4asset")

    backtest_rows = pd.DataFrame(
        [
            {
                "method": "historical",
                "confidence_level": 0.01,
                "n_obs": 100,
                "n_breaches": 1,
                "breach_rate": 0.01,
                "kupiec_lr": 0.5,
                "kupiec_p_value": 0.5,
                "kupiec_reject": False,
                "christoffersen_lr_ind": 0.3,
                "christoffersen_p_ind": 0.6,
                "christoffersen_reject_ind": False,
                "combined_lr": 0.8,
                "combined_p_value": 0.7,
                "combined_reject": False,
            },
            {
                "method": "historical",
                "confidence_level": "basel_summary",
                "n_obs": 90,
                "n_breaches": 2,
                "breach_rate": 0.9,
                "kupiec_lr": None,
                "kupiec_p_value": None,
                "kupiec_reject": None,
                "christoffersen_lr_ind": None,
                "christoffersen_p_ind": None,
                "christoffersen_reject_ind": None,
                "combined_lr": None,
                "combined_p_value": None,
                "combined_reject": False,
            },
        ]
    )
    db.write_backtest_results(backtest_rows, run_date="2026-01-01")

    db.close()
    return db_path


@pytest.fixture
def client(tmp_returns_parquet, tmp_risk_db, monkeypatch):
    monkeypatch.setenv("RISK_RETURNS_PATH", str(tmp_returns_parquet))
    monkeypatch.setenv("RISK_DB_PATH", str(tmp_risk_db))
    monkeypatch.setenv("RISK_API_KEY", "test-key")
    with TestClient(create_app(), headers={"X-API-Key": "test-key"}) as c:
        yield c
