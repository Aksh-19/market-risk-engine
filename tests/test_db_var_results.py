"""
Tests for the var_results table in src/risk_engine/storage/db.py
(Phase 3 persistence: write_var_results / read_var_results).
"""

from __future__ import annotations

import pandas as pd
import pytest

from risk_engine.storage.db import RiskDatabase


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test_risk.db"
    database = RiskDatabase(db_path)
    yield database
    database.close()


@pytest.fixture
def sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "Historical Sim",
                "cov_source": "-",
                "confidence_level": 0.01,
                "var": 0.0199,
                "es": 0.0280,
            },
            {
                "method": "Historical Sim",
                "cov_source": "-",
                "confidence_level": 0.05,
                "var": 0.0115,
                "es": 0.0174,
            },
            {
                "method": "Parametric",
                "cov_source": "ewma",
                "confidence_level": 0.01,
                "var": 0.0162,
                "es": 0.0185,
            },
            {
                "method": "Filtered Historical Sim",
                "cov_source": "GARCH",
                "confidence_level": 0.01,
                "var": 0.0168,
                "es": 0.0212,
            },
        ]
    )


def test_write_then_read_roundtrip(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_var_results(sample_results, run_date=run_date)

    out = db.read_var_results()
    assert len(out) == len(sample_results)
    assert set(out["method"]) == set(sample_results["method"])


def test_values_survive_roundtrip_exactly(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_var_results(sample_results, run_date=run_date)

    out = db.read_var_results()
    row = out[(out["method"] == "Parametric") & (out["cov_source"] == "ewma")].iloc[0]
    assert row["var_value"] == pytest.approx(0.0162)
    assert row["es_value"] == pytest.approx(0.0185)


def test_rerun_replaces_not_duplicates(db, sample_results):
    # Same invariant Phase 2 tested for write_volatility() --
    # write_var_results() twice for the same run_date/portfolio should
    # overwrite, not accumulate duplicate rows. A silent duplication bug
    # here would corrupt every downstream query (e.g. Phase 4 pulling
    # "today's VaR numbers" and getting doubled rows).
    run_date = pd.Timestamp("2025-01-15")
    db.write_var_results(sample_results, run_date=run_date)
    db.write_var_results(sample_results, run_date=run_date)

    out = db.read_var_results()
    assert len(out) == len(sample_results)


def test_different_run_dates_coexist(db, sample_results):
    # Unlike volatility (overwritten on every re-fit), VaR results are
    # explicitly meant to accumulate ACROSS run_dates -- that's what makes
    # "track how VaR estimates drift over time" possible later.
    db.write_var_results(sample_results, run_date=pd.Timestamp("2025-01-15"))
    db.write_var_results(sample_results, run_date=pd.Timestamp("2025-01-16"))

    out = db.read_var_results()
    assert len(out) == 2 * len(sample_results)
    assert set(out["run_date"]) == {"2025-01-15", "2025-01-16"}


def test_filter_by_run_date(db, sample_results):
    db.write_var_results(sample_results, run_date=pd.Timestamp("2025-01-15"))
    db.write_var_results(sample_results, run_date=pd.Timestamp("2025-01-16"))

    out = db.read_var_results(run_date=pd.Timestamp("2025-01-15"))
    assert len(out) == len(sample_results)
    assert (out["run_date"] == "2025-01-15").all()


def test_different_portfolios_are_isolated(db, sample_results):
    db.write_var_results(
        sample_results, run_date=pd.Timestamp("2025-01-15"), portfolio="equal_weight_4asset"
    )
    db.write_var_results(
        sample_results, run_date=pd.Timestamp("2025-01-15"), portfolio="custom_60_40"
    )

    out_equal = db.read_var_results(portfolio="equal_weight_4asset")
    out_custom = db.read_var_results(portfolio="custom_60_40")
    assert len(out_equal) == len(sample_results)
    assert len(out_custom) == len(sample_results)


def test_empty_query_returns_empty_dataframe(db):
    out = db.read_var_results(portfolio="nonexistent_portfolio")
    assert out.empty
