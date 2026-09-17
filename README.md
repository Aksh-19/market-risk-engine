# Market Risk Engine

A market risk modelling toolkit built from first principles: volatility
estimation, Value-at-Risk / Expected Shortfall calculation across four
methodologies, statistical backtesting, a FastAPI service layer, and a
Streamlit dashboard.

## Why this project exists

Most "risk modelling" portfolio projects stop at a single VaR number in a
Jupyter notebook. This one implements and *compares* four VaR
methodologies — including Filtered Historical Simulation, which combines
GARCH volatility conditioning with nonparametric tails rather than
assuming normality — statistically validates them with the same backtests
banks are regulatorily required to run (Kupiec, Christoffersen, Basel
traffic-light), and serves the results through a real API and dashboard.

## Roadmap

- [x] Phase 0 — Project scaffold, CI, testing setup
- [x] Phase 1 — Data ingestion & cleaning pipeline
- [x] Phase 2 — Volatility models (EWMA, GARCH(1,1), covariance estimation)
- [x] Phase 3 — VaR & Expected Shortfall (Historical Simulation, Parametric,
      Monte Carlo, Filtered Historical Simulation)
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
storing returns, volatility estimates, covariance snapshots, GARCH fit
history, and VaR/ES results in queryable long/tidy tables, complementing the
parquet raw-data cache. `src/risk_engine/diagnostics/stationarity.py` — the
Augmented Dickey-Fuller test implemented from scratch via OLS with
automatic BIC-based lag selection, empirically confirming on real data that
price levels are non-stationary while log returns are stationary — the
assumption this entire project's return-based modeling rests on. Validated
against simulated random-walk / stationary-AR(1) ground truth; measured
false-positive rate of 5.0% across 100 simulated random walks against a 5%
significance threshold.

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

**Phase 3 details:** `src/risk_engine/var/` — a portfolio-first design where
every method shares the `(returns_matrix, weights)` interface and
single-asset VaR falls out as a one-hot weight special case, never a
separate code path.

- `portfolio.py` — weight validation, `r_p = Σ wᵢrᵢ`, and `w'Σw` portfolio
  variance/volatility — the shared building blocks every method composes.
- `covariance_selector.py` — one entry point returning Σ from any of three
  sources: EWMA (default, time-varying), Ledoit-Wolf (full-sample, shrunk),
  or a GARCH-hybrid `D_garch @ R_ewma @ D_garch` construction — GARCH's
  faster per-asset shock reactivity substituted onto EWMA's correlation
  structure, a simplified cousin of DCC-GARCH.
- `historical.py` — nonparametric VaR/ES via empirical quantiles; makes no
  distributional assumption, so it fully reflects the fat tails Phase 1's
  validators already flagged (six 6σ SPY days in March 2020 — statistically
  near-impossible under normality).
- `parametric.py` — closed-form VaR/ES via `z_α · σ_p`, run across all three
  Σ sources, letting the covariance choice alone move the reported risk
  number.
- `monte_carlo.py` — Cholesky-decomposes Σ to simulate correlated return
  scenarios; validated to converge with Parametric's closed-form answer to
  within ~1% across all confidence levels and Σ sources — a genuine
  cross-check between two independently implemented methods, not just a
  design intention.
- `filtered_historical.py` — Filtered Historical Simulation: standardizes
  historical returns by GARCH conditional volatility (`z_t = r_t/σ_t`),
  preserves cross-asset correlation by keeping each day's shock vector
  intact, then rescales by *today's* GARCH forecast before taking the
  empirical quantile. Combines real historical tail shape with current
  volatility conditioning — closer to how sophisticated risk desks actually
  compute VaR than either pure Historical Sim or pure Parametric alone.
- `scripts/run_var.py` — orchestrates all four methods across both
  confidence levels, prints a full comparison table, and persists results
  to `data/processed/risk_engine.db` (`var_results` table, keyed by
  run date/portfolio/method/covariance source/confidence level — supports
  re-running without duplicating rows, and accumulates across run dates so
  VaR drift over time is queryable later).

**Real results (equal-weight SPY/AAPL/TLT/GLD, 2015–2024, 2,515 trading days):**

| Method | 1% VaR | 1% ES | ES/VaR ratio |
|---|---|---|---|
| Historical Simulation | 1.99% | 2.80% | 1.41 |
| Parametric (EWMA) | 1.62% | 1.85% | 1.15 |
| Monte Carlo (EWMA) | 1.61% | 1.85% | 1.15 |
| Filtered Historical Simulation | 1.68% | 2.12% | 1.26 |

The ES/VaR ratio is the key diagnostic: Historical Simulation's fat empirical
tail (1.41) versus Parametric's thin normal-distribution tail (1.15) is
direct, data-driven evidence that the normality assumption understates tail
risk — consistent with Phase 1's extreme-move findings. Filtered Historical
Simulation lands between the two (1.26), inheriting Historical Sim's real
tail shape while conditioning its threshold on current GARCH volatility,
exactly as the method's theory predicts.

129 tests total across all modules, all passing in CI.

## Local setup

```bash
# Clone your repo, then from its root:
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # installs the project + dev tools
pre-commit install               # activates the auto-format/lint git hook

pytest                           # should pass with the smoke test
```

## Running the pipeline

```bash
python3 scripts/run_ingestion.py   # Phase 1: fetch, validate, build returns matrix
python3 scripts/fit_volatility.py  # Phase 2: fit EWMA + GARCH, persist to SQLite
python3 scripts/run_var.py         # Phase 3: compute VaR/ES across 4 methods, persist results
```

## Project structure

```
src/risk_engine/     the actual package — importable code lives here
  data/               Phase 1: config, loader, returns, validators
  volatility/         Phase 2: EWMA, GARCH, rolling baseline, covariance/shrinkage
  var/                Phase 3: portfolio utils, covariance selector, 4 VaR/ES methods
  diagnostics/        ADF stationarity testing
  storage/            SQLite persistence layer (RiskDatabase)
tests/                pytest test suite, mirrors the src/ structure (129 tests)
configs/              YAML configs (asset universe, date ranges) — no hardcoded params in code
scripts/              thin orchestration entry points (run_ingestion.py, fit_volatility.py, run_var.py)
data/cache/           gitignored — raw price pulls, regenerable via scripts/run_ingestion.py
data/processed/       gitignored — pipeline outputs (returns_matrix.parquet, risk_engine.db, etc.)
.github/workflows/    CI: lint + test on every push
pyproject.toml        dependencies + tool config (single source of truth)
```