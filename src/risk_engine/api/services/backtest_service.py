from risk_engine.api.errors import NoBacktestError
from risk_engine.api.schemas import BacktestQuantileResult, BaselSummary, BacktestResponse
from risk_engine.storage.db import RiskDatabase


def get_backtest_results(db: RiskDatabase, method: str) -> BacktestResponse:
    df = db.read_backtest_results()
    df = df[df["method"] == method]
    if df.empty:
        raise NoBacktestError(method)

    basel_row = df[df["confidence_level"] == "basel_summary"]
    quantile_df = df[df["confidence_level"] != "basel_summary"]

    quantile_results = [
        BacktestQuantileResult(
            confidence_level=float(row.confidence_level),
            n_obs=row.n_obs,
            n_breaches=row.n_breaches,
            breach_rate=row.breach_rate,
            kupiec_lr=row.kupiec_lr,
            kupiec_p_value=row.kupiec_p_value,
            kupiec_reject=bool(row.kupiec_reject),
            christoffersen_lr_ind=row.christoffersen_lr_ind,
            christoffersen_p_ind=row.christoffersen_p_ind,
            christoffersen_reject_ind=bool(row.christoffersen_reject_ind),
            combined_lr=row.combined_lr,
            combined_p_value=row.combined_p_value,
            combined_reject=bool(row.combined_reject),
        )
        for row in quantile_df.itertuples()
    ]

    basel_summary = None
    if not basel_row.empty:
        r = basel_row.iloc[0]
        basel_summary = BaselSummary(
            n_windows=int(r.n_obs),
            current_breaches=int(r.n_breaches),
            pct_time_green=float(r.breach_rate),
            current_zone_is_red=bool(r.combined_reject),
        )

    return BacktestResponse(
        method=method,
        run_date=(
            quantile_df.iloc[0].run_date if not quantile_df.empty else basel_row.iloc[0].run_date
        ),
        quantile_results=quantile_results,
        basel_summary=basel_summary,
    )
