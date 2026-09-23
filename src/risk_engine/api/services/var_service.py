import time
import numpy as np
import pandas as pd
from risk_engine import __version__
from risk_engine.api.errors import UnknownTickerError
from risk_engine.api.schemas import Provenance, VaRRequest, VaRResponse, VaRResult
from risk_engine.var import (
    historical_var,
    historical_expected_shortfall,
    parametric_var,
    parametric_expected_shortfall,
    monte_carlo_var,
    monte_carlo_expected_shortfall,
    filtered_historical_var,
    filtered_historical_expected_shortfall,
)

# Each entry: (var_fn, es_fn, extra_kwargs_builder)
# extra_kwargs_builder(req) -> dict of method-specific kwargs
METHODS = {
    "historical": (
        historical_var,
        historical_expected_shortfall,
        lambda req: {},
    ),
    "parametric": (
        parametric_var,
        parametric_expected_shortfall,
        lambda req: {"cov_method": req.cov_source},
    ),
    "monte_carlo": (
        monte_carlo_var,
        monte_carlo_expected_shortfall,
        lambda req: {"cov_method": req.cov_source, "seed": req.seed},
    ),
    "filtered_historical": (
        filtered_historical_var,
        filtered_historical_expected_shortfall,
        lambda req: {},
    ),
}


def _weight_vector(returns: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    unknown = sorted(set(weights) - set(returns.columns))
    if unknown:
        raise UnknownTickerError(unknown)
    return pd.Series(weights).reindex(returns.columns).fillna(0.0).to_numpy()


def compute_var(returns: pd.DataFrame, req: VaRRequest) -> VaRResponse:
    w = _weight_vector(returns, req.weights)
    alpha = round(1 - req.confidence_level, 10)
    alphas = (alpha,)  # engine expects a tuple even for one level

    t0 = time.perf_counter()
    results = []
    for m in req.methods:
        var_fn, es_fn, kwargs_fn = METHODS[m]
        kwargs = kwargs_fn(req)
        var = var_fn(returns, w, confidence_levels=alphas, **kwargs)[alpha]
        es = es_fn(returns, w, confidence_levels=alphas, **kwargs)[alpha]
        results.append(VaRResult(method=m, var=var, es=es, es_var_ratio=es / var))

    return VaRResponse(
        confidence_level=req.confidence_level,
        results=results,
        provenance=Provenance(
            data_as_of=returns.index[-1].date(),
            n_observations=len(returns),
            engine_version=__version__,
            cov_source=req.cov_source,
            seed=req.seed,
            compute_ms=(time.perf_counter() - t0) * 1000,
        ),
    )
