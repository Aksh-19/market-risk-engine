from risk_engine.api.schemas import VaRHistoryEntry, VaRHistoryResponse
from risk_engine.storage.db import RiskDatabase

# DB stores run_var.py's human-readable labels; API uses canonical enum values.
_METHOD_DB_TO_API = {
    "Historical Sim": "historical",
    "Parametric": "parametric",
    "Monte Carlo": "monte_carlo",
    "Filtered Historical Sim": "filtered_historical",
}
_METHOD_API_TO_DB = {v: k for k, v in _METHOD_DB_TO_API.items()}

_COV_DB_TO_API = {
    "-": None,  # Historical Sim has no covariance source
    "ewma": "ewma",
    "ledoit_wolf": "ledoit_wolf",
    "GARCH": "garch_hybrid",
}
_COV_API_TO_DB = {v: k for k, v in _COV_DB_TO_API.items() if v is not None}


def get_var_history(
    db: RiskDatabase,
    portfolio: str = "equal_weight_4asset",
    method: str | None = None,
    cov_source: str | None = None,
    confidence_level: float | None = None,
    start: str | None = None,
    end: str | None = None,
) -> VaRHistoryResponse:
    df = db.read_var_results(portfolio=portfolio)

    if method:
        df = df[df["method"] == _METHOD_API_TO_DB[method]]
    if cov_source:
        df = df[df["cov_source"] == _COV_API_TO_DB[cov_source]]
    if confidence_level is not None:
        alpha = round(1 - confidence_level, 10)
        df = df[df["confidence_level"] == alpha]
    if start:
        df = df[df["run_date"] >= start]
    if end:
        df = df[df["run_date"] <= end]

    entries = [
        VaRHistoryEntry(
            run_date=row.run_date,
            portfolio=row.portfolio,
            method=_METHOD_DB_TO_API[row.method],
            cov_source=_COV_DB_TO_API[row.cov_source],
            confidence_level=round(1 - row.confidence_level, 10),
            var_value=row.var_value,
            es_value=row.es_value,
        )
        for row in df.itertuples()
    ]
    return VaRHistoryResponse(count=len(entries), results=entries)
