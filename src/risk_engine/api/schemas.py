from datetime import date
from typing import Annotated, Literal, get_args
from pydantic import BaseModel, Field, model_validator

Method = Literal["historical", "parametric", "monte_carlo", "filtered_historical"]
CovSource = Literal["ewma", "ledoit_wolf", "garch_hybrid"]
Weight = Annotated[float, Field(allow_inf_nan=False)]


class VaRRequest(BaseModel):
    weights: dict[str, Weight]
    confidence_level: float = Field(0.99, gt=0.5, lt=1.0)
    methods: list[Method] = Field(default_factory=lambda: list(get_args(Method)))
    cov_source: CovSource = "ewma"
    seed: int | None = None

    @model_validator(mode="after")
    def _weights_sum_to_one(self):
        if not self.weights:
            raise ValueError("weights must not be empty")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1 (got {total:.6f})")
        return self


class VaRResult(BaseModel):
    method: Method
    var: float = Field(description="Positive decimal loss fraction, e.g. 0.0199")
    es: float
    es_var_ratio: float


class Provenance(BaseModel):
    data_as_of: date
    n_observations: int
    engine_version: str
    cov_source: CovSource
    seed: int | None
    compute_ms: float


class VaRResponse(BaseModel):
    confidence_level: float
    horizon_days: int = 1
    results: list[VaRResult]
    provenance: Provenance


class VaRHistoryEntry(BaseModel):
    run_date: date
    portfolio: str
    method: Method
    cov_source: CovSource | None
    confidence_level: float  # returned in 0.99-style, flipped back for display
    var_value: float
    es_value: float


class VaRHistoryResponse(BaseModel):
    count: int
    results: list[VaRHistoryEntry]


class BacktestQuantileResult(BaseModel):
    confidence_level: float
    n_obs: int
    n_breaches: int
    breach_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    kupiec_reject: bool
    christoffersen_lr_ind: float
    christoffersen_p_ind: float
    christoffersen_reject_ind: bool
    combined_lr: float
    combined_p_value: float
    combined_reject: bool


class BaselSummary(BaseModel):
    n_windows: int
    current_breaches: int
    pct_time_green: float
    current_zone_is_red: bool


class BacktestResponse(BaseModel):
    method: Method
    run_date: date
    quantile_results: list[BacktestQuantileResult]
    basel_summary: BaselSummary | None
