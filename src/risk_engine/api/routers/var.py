from fastapi import APIRouter, Depends, Request
import pandas as pd
from risk_engine.api.schemas import VaRRequest, VaRResponse
from risk_engine.api.services.var_service import compute_var

router = APIRouter(prefix="/v1", tags=["var"])


def get_returns(request: Request) -> pd.DataFrame:
    return request.app.state.returns


@router.post("/var", response_model=VaRResponse)
def post_var(req: VaRRequest, returns: pd.DataFrame = Depends(get_returns)):
    return compute_var(returns, req)
