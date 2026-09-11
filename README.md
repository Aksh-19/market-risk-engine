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
- [ ] Phase 2 — Volatility models (EWMA, GARCH(1,1), covariance estimation)
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
`data/processed/returns_matrix.parquet` as the input to Phase 2. 17 tests
cover the return math, alignment logic, and caching behavior. Default
universe: SPY, AAPL, TLT, GLD, 2015–2025.

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
data/processed/       gitignored — pipeline outputs (returns_matrix.parquet etc.)
.github/workflows/    CI: lint + test on every push
pyproject.toml        dependencies + tool config (single source of truth)
```
