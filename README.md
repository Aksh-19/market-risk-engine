# Market Risk Engine

A market risk modelling toolkit built from first principles: volatility
estimation, Value-at-Risk / Expected Shortfall calculation across multiple
methodologies, statistical backtesting, a FastAPI service layer, and a
Streamlit dashboard.

## Why this project exists

Most "risk modelling" portfolio projects stop at a single VaR number in a
Jupyter notebook. This one implements and *compares* multiple VaR
methodologies, statistically validates them with the same backtests banks
are regulatorily required to run (Kupiec, Christoffersen, Basel
traffic-light), and serves the results through a real API and dashboard.

## Roadmap

- [x] Phase 0 — Project scaffold, CI, testing setup
- [x] Phase 1 — Data ingestion & cleaning pipeline
- [x] Phase 2 — Volatility models (EWMA, GARCH(1,1), covariance estimation)
- [ ] Phase 3 — VaR & Expected Shortfall (historical, parametric, Monte Carlo, Cornish-Fisher)
- [ ] Phase 4 — Backtesting (Kupiec, Christoffersen, Basel traffic-light)
- [ ] Phase 5 — FastAPI service layer
- [ ] Phase 6 — Streamlit dashboard
- [ ] Phase 7 — Docker + deployment + CI polish

**Phase 1 details:** `src/risk_engine/data/` — Pydantic-validated config
(`DataConfig`), a caching data loader (`DataLoader`) pulling from Yahoo
Finance, data-quality validation (missing dates, zero-variance days,
duplicates, statistically extreme moves — flagged, never silently dropped),
and log-return calculation with calendar-aligned multi-asset joins.
`scripts/run_ingestion.py` runs the full pipeline end to end, producing
`data/processed/returns_matrix.parquet` as the input to Phase 2. Default
universe: SPY, AAPL, TLT, GLD, 2015–2025.

**Phase 2 details:** `src/risk_engine/volatility/` — EWMA volatility and
covariance estimation (`ewma.py`, RiskMetrics-style, with half-life<->lambda
conversion) and GARCH(1,1) fit via maximum likelihood (`garch.py`), including
variance forecasting and mean-reversion to a long-run unconditional
variance — the key property EWMA structurally lacks. `scripts/fit_volatility.py`
runs both models across the universe and writes
`data/processed/volatility_summary.parquet` for Phase 3.

**Stage A (Phase 1 extension) — persistence & stationarity:**
`src/risk_engine/storage/db.py` — a SQLite persistence layer (`RiskDatabase`)
storing returns, volatility estimates, covariance snapshots, and GARCH fit
history in queryable long/tidy tables, complementing the parquet raw-data
cache. `src/risk_engine/diagnostics/stationarity.py` — the Augmented
Dickey-Fuller test implemented from scratch via OLS with automatic
BIC-based lag selection, empirically confirming on real data that price
levels are non-stationary while log returns are stationary — the
assumption this entire project's return-based modeling rests on. Validated
against simulated random-walk / stationary-AR(1) ground truth; measured
false-positive rate of 5.0% across 100 simulated random walks against a
5% significance threshold.

**Stage B (Phase 2 extension) — baselines & robust covariance:**
`src/risk_engine/volatility/rolling.py` — a naive equal-weighted rolling
volatility baseline, giving EWMA/GARCH something concrete to beat; its test
suite numerically demonstrates the "ghosting" effect (a single shock causes
a ~6x volatility spike that mechanically vanishes exactly `window` days
later, regardless of actual market conditions). `src/risk_engine/volatility/covariance.py`
— plain sample covariance and Ledoit-Wolf (2004) shrinkage implemented from
scratch, cross-validated against scikit-learn's reference implementation to
within floating-point precision (~1e-16) across multiple sample sizes and
asset counts. The custom GARCH(1,1) implementation was additionally
cross-validated against the `arch` package (`scripts/validate_garch_vs_arch.py`)
on real data — parameters agree within ~0.1–0.2% after fixing a shared
numerical scaling issue independently flagged by both implementations.

55 tests total across all modules, all passing in CI.

## Local setup

```bash
# Clone your repo, then from its root:
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # installs the project + dev tools
pre-commit install               # activates the auto-format/lint git hook

pytest                           # should pass with the smoke test
```

## Project structure

```
src/risk_engine/     the actual package — importable code lives here
tests/                pytest test suite, mirrors the src/ structure
configs/              YAML configs (asset universe, date ranges) — no hardcoded params in code
scripts/              thin orchestration entry points (e.g. run_ingestion.py)
data/cache/           gitignored — raw price pulls, regenerable via scripts/run_ingestion.py
data/processed/       gitignored — pipeline outputs (returns_matrix.parquet, risk_engine.db, etc.)
.github/workflows/    CI: lint + test on every push
pyproject.toml        dependencies + tool config (single source of truth)
```