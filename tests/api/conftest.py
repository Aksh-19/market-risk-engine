import numpy as np
import pandas as pd
import pytest


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
