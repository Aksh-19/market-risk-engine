from fastapi import APIRouter, Depends, Request
import pandas as pd
from risk_engine.api.schemas import VaRRequest, VaRResponse, Method, CovSource
from risk_engine.api.services.var_service import compute_var
from risk_engine.api.schemas import VaRHistoryResponse, BacktestResponse
from risk_engine.api.services.history_service import get_var_history
from risk_engine.api.services.backtest_service import get_backtest_results
from risk_engine.storage.db import RiskDatabase
from collections.abc import Iterator
from risk_engine.api.schemas import AttributionRequest, AttributionResponse
from risk_engine.api.services.attribution_service import compute_attribution


router = APIRouter(prefix="/v1", tags=["var"])


def get_returns(request: Request) -> pd.DataFrame:
    return request.app.state.returns


@router.post("/var", response_model=VaRResponse)
def post_var(req: VaRRequest, returns: pd.DataFrame = Depends(get_returns)):
    return compute_var(returns, req)


def get_db(request: Request) -> Iterator[RiskDatabase]:
    db = RiskDatabase(request.app.state.settings.db_path)
    try:
        yield db
    finally:
        db.close()


@router.get("/var/history", response_model=VaRHistoryResponse)
def var_history(
    portfolio: str = "equal_weight_4asset",
    method: Method | None = None,
    cov_source: CovSource | None = None,
    confidence_level: float | None = None,
    start: str | None = None,
    end: str | None = None,
    db: RiskDatabase = Depends(get_db),
):
    return get_var_history(db, portfolio, method, cov_source, confidence_level, start, end)


@router.get("/backtests/{method}", response_model=BacktestResponse)
def backtests(method: Method, db: RiskDatabase = Depends(get_db)):
    return get_backtest_results(db, method)


@router.post("/var/attribution", response_model=AttributionResponse)
def post_attribution(req: AttributionRequest, returns: pd.DataFrame = Depends(get_returns)):
    return compute_attribution(returns, req)
