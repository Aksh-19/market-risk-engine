"""
Tests for the backtest_results table in src/risk_engine/storage/db.py
(Phase 4 persistence: write_backtest_results / read_backtest_results).
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
                "method": "historical",
                "confidence_level": 0.01,
                "n_obs": 2015,
                "n_breaches": 24,
                "breach_rate": 0.0119,
                "kupiec_lr": 0.700,
                "kupiec_p_value": 0.4027,
                "kupiec_reject": False,
                "christoffersen_lr_ind": 4.607,
                "christoffersen_p_ind": 0.0318,
                "christoffersen_reject_ind": True,
                "combined_lr": 5.307,
                "combined_p_value": 0.0704,
                "combined_reject": False,
            },
            {
                "method": "historical",
                "confidence_level": "basel_summary",
                "n_obs": 1766,
                "n_breaches": 2,
                "breach_rate": 0.707,
                "kupiec_lr": None,
                "kupiec_p_value": None,
                "kupiec_reject": None,
                "christoffersen_lr_ind": None,
                "christoffersen_p_ind": None,
                "christoffersen_reject_ind": None,
                "combined_lr": None,
                "combined_p_value": None,
                "combined_reject": False,
            },
        ]
    )


def test_write_then_read_roundtrip(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_backtest_results(sample_results, run_date=run_date)

    out = db.read_backtest_results()
    assert len(out) == len(sample_results)


def test_numeric_values_survive_roundtrip(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_backtest_results(sample_results, run_date=run_date)

    out = db.read_backtest_results()
    row = out[out["confidence_level"] == "0.01"].iloc[0]
    assert row["kupiec_lr"] == pytest.approx(0.700)
    assert row["combined_p_value"] == pytest.approx(0.0704)


def test_boolean_flags_survive_as_integers(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_backtest_results(sample_results, run_date=run_date)

    out = db.read_backtest_results()
    row = out[out["confidence_level"] == "0.01"].iloc[0]
    assert row["kupiec_reject"] == 0  # False -> 0
    assert row["christoffersen_reject_ind"] == 1  # True -> 1


def test_none_values_in_basel_row_survive_as_null(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_backtest_results(sample_results, run_date=run_date)

    out = db.read_backtest_results()
    basel_row = out[out["confidence_level"] == "basel_summary"].iloc[0]
    assert pd.isna(basel_row["kupiec_lr"])
    assert pd.isna(basel_row["kupiec_reject"])


def test_string_confidence_level_and_numeric_coexist(db, sample_results):
    # The core reason confidence_level is TEXT not REAL: "basel_summary"
    # and "0.01" must both be storable in the same column without error.
    run_date = pd.Timestamp("2025-01-15")
    db.write_backtest_results(sample_results, run_date=run_date)

    out = db.read_backtest_results()
    assert set(out["confidence_level"]) == {"0.01", "basel_summary"}


def test_rerun_replaces_not_duplicates(db, sample_results):
    run_date = pd.Timestamp("2025-01-15")
    db.write_backtest_results(sample_results, run_date=run_date)
    db.write_backtest_results(sample_results, run_date=run_date)

    out = db.read_backtest_results()
    assert len(out) == len(sample_results)


def test_different_run_dates_coexist(db, sample_results):
    db.write_backtest_results(sample_results, run_date=pd.Timestamp("2025-01-15"))
    db.write_backtest_results(sample_results, run_date=pd.Timestamp("2025-01-16"))

    out = db.read_backtest_results()
    assert len(out) == 2 * len(sample_results)


def test_filter_by_run_date(db, sample_results):
    db.write_backtest_results(sample_results, run_date=pd.Timestamp("2025-01-15"))
    db.write_backtest_results(sample_results, run_date=pd.Timestamp("2025-01-16"))

    out = db.read_backtest_results(run_date=pd.Timestamp("2025-01-15"))
    assert len(out) == len(sample_results)
    assert (out["run_date"] == "2025-01-15").all()


def test_empty_query_returns_empty_dataframe(db):
    out = db.read_backtest_results(run_date=pd.Timestamp("2099-01-01"))
    assert out.empty
