import numpy as np
import pytest
from risk_engine.var import historical_var
from fastapi.testclient import TestClient


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


def test_history_filters_and_maps_correctly(client):
    r = client.get(
        "/v1/var/history",
        params={
            "portfolio": "equal_weight_4asset",
            "method": "historical",
            "confidence_level": 0.99,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["results"][0]["cov_source"] is None
    assert body["results"][0]["var_value"] == pytest.approx(0.0199)


def test_history_empty_when_no_match(client):
    r = client.get("/v1/var/history", params={"method": "monte_carlo", "confidence_level": 0.99})
    assert r.status_code == 200
    assert r.json() == {"count": 0, "results": []}


def test_attribution_sums_to_portfolio_var(client):
    r = client.post(
        "/v1/var/attribution",
        json={
            "weights": {"SPY": 0.25, "AAPL": 0.25, "TLT": 0.25, "GLD": 0.25},
            "confidence_level": 0.99,
            "cov_source": "ewma",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sum_check"] == pytest.approx(body["portfolio_var"], abs=1e-9)
    assert sum(c["weight"] for c in body["contributions"]) == pytest.approx(1.0)


def test_attribution_unknown_ticker(client):
    r = client.post(
        "/v1/var/attribution", json={"weights": {"NOPE": 1.0}, "confidence_level": 0.99}
    )
    assert r.status_code == 422


def test_missing_api_key_rejected(tmp_returns_parquet, tmp_risk_db, monkeypatch):
    monkeypatch.setenv("RISK_RETURNS_PATH", str(tmp_returns_parquet))
    monkeypatch.setenv("RISK_DB_PATH", str(tmp_risk_db))
    monkeypatch.setenv("RISK_API_KEY", "test-key")
    from risk_engine.api.main import create_app

    with TestClient(create_app()) as no_key_client:  # deliberately no header
        r = no_key_client.post("/v1/var", json={"weights": {"SPY": 1.0}, "confidence_level": 0.99})
    assert r.status_code == 401
