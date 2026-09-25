import pytest

# client fixture is NOT redefined here — pytest auto-discovers fixtures
# from conftest.py for every test file in the same directory, so this
# file just uses `client` as a parameter like test_var_endpoint.py does.


def test_backtest_returns_quantile_and_basel(client):
    r = client.get("/v1/backtests/historical")
    assert r.status_code == 200
    body = r.json()
    assert len(body["quantile_results"]) == 1
    assert body["basel_summary"]["pct_time_green"] == pytest.approx(0.9)


def test_backtest_unknown_method_returns_422_for_invalid_literal(client):
    r = client.get("/v1/backtests/nonexistent_method")
    assert r.status_code == 422


def test_backtest_valid_method_no_data_404(client):
    r = client.get("/v1/backtests/parametric")
    assert r.status_code == 404
