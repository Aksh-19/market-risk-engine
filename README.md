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
- [ ] Phase 1 — Data ingestion & cleaning pipeline
- [ ] Phase 2 — Volatility models (EWMA, GARCH(1,1), covariance estimation)
- [ ] Phase 3 — VaR & Expected Shortfall (historical, parametric, Monte Carlo, Cornish-Fisher)
- [ ] Phase 4 — Backtesting (Kupiec, Christoffersen, Basel traffic-light)
- [ ] Phase 5 — FastAPI service layer
- [ ] Phase 6 — Streamlit dashboard
- [ ] Phase 7 — Docker + deployment + CI polish

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
.github/workflows/    CI: lint + test on every push
pyproject.toml        dependencies + tool config (single source of truth)
```
