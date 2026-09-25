from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from risk_engine import __version__
from risk_engine.api.config import Settings
from risk_engine.api.errors import UnknownTickerError, NoBacktestError
from risk_engine.api.routers import health, var


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    returns = pd.read_parquet(settings.returns_path)
    if returns.empty:
        raise RuntimeError("returns matrix is empty")
    app.state.settings, app.state.returns = settings, returns
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="market-risk-engine", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(var.router)

    @app.exception_handler(UnknownTickerError)
    async def _unknown_ticker(_: Request, exc: UnknownTickerError):
        return JSONResponse(
            status_code=422,
            content={"error": "unknown_ticker", "detail": str(exc), "tickers": exc.tickers},
        )

    @app.exception_handler(NoBacktestError)
    async def _no_backtest(_: Request, exc: NoBacktestError):
        return JSONResponse(
            status_code=404,
            content={"error": "no_backtest", "detail": str(exc), "method": exc.method},
        )

    return app


app = create_app()
