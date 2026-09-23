import numpy as np
import pytest
from fastapi.testclient import TestClient
from risk_engine.api.main import create_app
from risk_engine.var import historical_var


@pytest.fixture
def client(tmp_returns_parquet, monkeypatch):
    monkeypatch.setenv("RISK_RETURNS_PATH", str(tmp_returns_parquet))
    with TestClient(create_app()) as c:
        yield c


def test_api_matches_engine(client):
    w = {"SPY": 0.25, "AAPL": 0.25, "TLT": 0.25, "GLD": 0.25}
    body = client.post(
        "/v1/var",
        json={
            "weights": w,
            "confidence_level": 0.99,
            "methods": ["historical"],
        },
    ).json()

    weights_arr = np.array([w[t] for t in client.app.state.returns.columns])
    direct = historical_var(client.app.state.returns, weights_arr, confidence_levels=(0.01,))[0.01]

    assert body["results"][0]["var"] == pytest.approx(direct, abs=1e-12)


def test_rejects_bad_weights(client):
    r = client.post("/v1/var", json={"weights": {"SPY": 0.5}, "confidence_level": 0.99})
    assert r.status_code == 422


def test_unknown_ticker(client):
    r = client.post("/v1/var", json={"weights": {"NOPE": 1.0}, "confidence_level": 0.99})
    assert r.status_code == 422
    assert r.json()["tickers"] == ["NOPE"]
