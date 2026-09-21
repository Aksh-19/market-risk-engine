"""
SQLite persistence for processed risk data (returns, volatility estimates,
covariance snapshots).

WHY A DATABASE, GIVEN PARQUET ALREADY WORKS
-----------------------------------------------
Parquet (used in the data/cache/ layer) is excellent for "read the whole
file into pandas" access patterns -- which is exactly what raw price
caching needs. But it's fundamentally a flat file: to answer "which dates
did AAPL's GARCH volatility exceed 40%?" you have to load the entire file
into memory first and filter in pandas. As this project accumulates more
derived data -- volatility estimates from THREE different models, rolling
covariance matrices, eventually regime labels from Phase C -- the ability
to query directly (WHERE ticker='AAPL' AND model='garch' AND vol > 0.4)
without loading everything becomes genuinely useful, and is how this data
would actually be served if this became the backend for a real dashboard
or API (which, not coincidentally, is exactly Phase 5-6 of this project).

WHY "LONG" (TIDY) FORMAT, NOT WIDE
---------------------------------------
Your in-memory pandas DataFrames are "wide": one column per ticker. But
relational databases are built around "long"/tidy tables: one row per
individual observation, e.g. (date, ticker, value). This isn't a stylistic
choice -- a wide table would need a new database COLUMN every time you add
a ticker, and covariance matrices don't even have a natural wide form
without an awkward ticker-pair-per-column scheme. Long format is the
standard relational pattern precisely because it doesn't care how many
tickers or models you have; you just add more ROWS. Converting between wide
(for pandas/numpy math) and long (for storage) at the boundary is the
normal, expected friction of this design -- see the pivot in read_returns().

SCHEMA
--------
returns(date, ticker, log_return)                          -- one row per asset-day
volatility(date, ticker, model, value)                      -- model in {'rolling','ewma','garch'}
covariance(date, model, ticker_i, ticker_j, value)           -- upper triangle only (i<=j); symmetric, so storing both halves is redundant
garch_params(ticker, fit_date, omega, alpha, beta, log_likelihood, converged)  -- one row per fit, so you can track re-fits over time
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd


class RiskDatabase:
    """Thin wrapper around a SQLite file for the processed data layer."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._initialize_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        # IF NOT EXISTS makes this safe to call every time the app starts --
        # no separate "migration" step needed for a project at this scale.
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS returns (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                log_return REAL NOT NULL,
                PRIMARY KEY (date, ticker)
            );

            CREATE TABLE IF NOT EXISTS volatility (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                model TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (date, ticker, model)
            );

            CREATE TABLE IF NOT EXISTS covariance (
                date TEXT NOT NULL,
                model TEXT NOT NULL,
                ticker_i TEXT NOT NULL,
                ticker_j TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (date, model, ticker_i, ticker_j)
            );

            CREATE TABLE IF NOT EXISTS garch_params (
                ticker TEXT NOT NULL,
                fit_date TEXT NOT NULL,
                omega REAL NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                log_likelihood REAL NOT NULL,
                converged INTEGER NOT NULL,
                PRIMARY KEY (ticker, fit_date)
            );

            CREATE TABLE IF NOT EXISTS var_results (
                run_date TEXT NOT NULL,
                portfolio TEXT NOT NULL,
                method TEXT NOT NULL,
                cov_source TEXT NOT NULL,
                confidence_level REAL NOT NULL,
                var_value REAL NOT NULL,
                es_value REAL NOT NULL,
                PRIMARY KEY (run_date, portfolio, method, cov_source, confidence_level)
            );

            CREATE TABLE IF NOT EXISTS backtest_results (
                run_date TEXT NOT NULL,
                method TEXT NOT NULL,
                confidence_level TEXT NOT NULL,
                n_obs INTEGER,
                n_breaches INTEGER,
                breach_rate REAL,
                kupiec_lr REAL,
                kupiec_p_value REAL,
                kupiec_reject INTEGER,
                christoffersen_lr_ind REAL,
                christoffersen_p_ind REAL,
                christoffersen_reject_ind INTEGER,
                combined_lr REAL,
                combined_p_value REAL,
                combined_reject INTEGER,
                PRIMARY KEY (run_date, method, confidence_level)
            );

            CREATE INDEX IF NOT EXISTS idx_returns_ticker ON returns(ticker);
            CREATE INDEX IF NOT EXISTS idx_volatility_ticker_model ON volatility(ticker, model);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def write_returns(self, returns: pd.DataFrame) -> None:
        """Store a wide returns DataFrame (columns=tickers) in long form."""
        long = returns.reset_index().melt(
            id_vars=returns.index.name or "index", var_name="ticker", value_name="log_return"
        )
        long.columns = ["date", "ticker", "log_return"]
        long["date"] = long["date"].astype(str)
        long.to_sql("returns", self._conn, if_exists="replace", index=False)
        self._conn.commit()

    def read_returns(
        self, tickers: list[str] | None = None, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Read back returns as a wide DataFrame (columns=tickers), the
        shape everything else in the codebase (EWMA, GARCH) expects."""
        query = "SELECT date, ticker, log_return FROM returns WHERE 1=1"
        params: list = []
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        long = pd.read_sql(query, self._conn, params=params, parse_dates=["date"])
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="date", columns="ticker", values="log_return")
        wide.index.name = "date"
        return wide.sort_index()

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def write_volatility(self, vol: pd.Series, ticker: str, model: str) -> None:
        """Store one asset's volatility series under a named model
        ('rolling', 'ewma', 'garch') so multiple models can coexist per ticker."""
        df = vol.rename("value").reset_index()
        df.columns = ["date", "value"]
        df["date"] = df["date"].astype(str)
        df["ticker"] = ticker
        df["model"] = model
        df = df.dropna(subset=["value"])  # don't persist the NaN burn-in rows

        # Upsert-by-replace: delete any existing rows for this ticker+model
        # combination first, so re-running a fit doesn't leave stale rows
        # from a previous run mixed in with fresh ones.
        self._conn.execute("DELETE FROM volatility WHERE ticker = ? AND model = ?", (ticker, model))
        df[["date", "ticker", "model", "value"]].to_sql(
            "volatility", self._conn, if_exists="append", index=False
        )
        self._conn.commit()

    def read_volatility(self, ticker: str, model: str) -> pd.Series:
        query = "SELECT date, value FROM volatility WHERE ticker = ? AND model = ? ORDER BY date"
        df = pd.read_sql(query, self._conn, params=[ticker, model], parse_dates=["date"])
        return df.set_index("date")["value"].rename(f"{ticker}_{model}")

    # ------------------------------------------------------------------
    # Covariance snapshots
    # ------------------------------------------------------------------

    def write_covariance_snapshot(
        self, date: pd.Timestamp, cov: np.ndarray, tickers: list[str], model: str
    ) -> None:
        """Store one date's covariance matrix, upper triangle only (i<=j) --
        the matrix is symmetric so the lower triangle is fully redundant."""
        date_str = str(pd.Timestamp(date).date())
        rows = []
        for i, ti in enumerate(tickers):
            for j in range(i, len(tickers)):
                tj = tickers[j]
                rows.append((date_str, model, ti, tj, float(cov[i, j])))

        self._conn.execute("DELETE FROM covariance WHERE date = ? AND model = ?", (date_str, model))
        self._conn.executemany(
            "INSERT INTO covariance (date, model, ticker_i, ticker_j, value) VALUES (?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def read_covariance_snapshot(
        self, date: pd.Timestamp, tickers: list[str], model: str
    ) -> np.ndarray:
        """Reconstruct the full symmetric matrix from the stored upper triangle."""
        date_str = str(pd.Timestamp(date).date())
        query = "SELECT ticker_i, ticker_j, value FROM covariance " "WHERE date = ? AND model = ?"
        df = pd.read_sql(query, self._conn, params=[date_str, model])
        if df.empty:
            raise KeyError(f"No covariance snapshot for {date_str} / model={model}")

        n = len(tickers)
        idx = {t: k for k, t in enumerate(tickers)}
        matrix = np.full((n, n), np.nan)
        for _, row in df.iterrows():
            i, j, v = idx[row["ticker_i"]], idx[row["ticker_j"]], row["value"]
            matrix[i, j] = v
            matrix[j, i] = v  # mirror the stored upper triangle
        return matrix

    # ------------------------------------------------------------------
    # GARCH fit parameters
    # ------------------------------------------------------------------

    def write_garch_params(
        self,
        ticker: str,
        fit_date: pd.Timestamp,
        omega: float,
        alpha: float,
        beta: float,
        log_likelihood: float,
        converged: bool,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO garch_params
            (ticker, fit_date, omega, alpha, beta, log_likelihood, converged)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                str(pd.Timestamp(fit_date).date()),
                omega,
                alpha,
                beta,
                log_likelihood,
                int(converged),
            ),
        )
        self._conn.commit()

    def read_garch_params(self, ticker: str) -> pd.DataFrame:
        """All historical fits for a ticker, letting you track how estimated
        parameters have drifted across re-fits over time."""
        query = "SELECT * FROM garch_params WHERE ticker = ? ORDER BY fit_date"
        return pd.read_sql(query, self._conn, params=[ticker], parse_dates=["fit_date"])

    # ------------------------------------------------------------------
    # VaR / ES results (Phase 3)
    # ------------------------------------------------------------------

    def write_var_results(
        self,
        results: pd.DataFrame,
        run_date: pd.Timestamp,
        portfolio: str = "equal_weight_4asset",
    ) -> None:
        """
        Persist one full run_var.py comparison table.

        Expects a long-format DataFrame with columns:
        method, cov_source, confidence_level (float, e.g. 0.01), var, es.
        This is exactly the shape of the `summary_rows` list built in
        scripts/run_var.py, before the display-string formatting is applied
        -- store raw floats, format only at print/read time.

        Upsert-by-replace: deletes any existing rows for this run_date +
        portfolio first, matching the same "re-running overwrites, never
        accumulates duplicates" convention used by write_volatility() and
        write_covariance_snapshot() in Phase 2.
        """
        run_date_str = str(pd.Timestamp(run_date).date())

        df = results.copy()
        df["run_date"] = run_date_str
        df["portfolio"] = portfolio
        df = df.rename(columns={"var": "var_value", "es": "es_value"})

        self._conn.execute(
            "DELETE FROM var_results WHERE run_date = ? AND portfolio = ?",
            (run_date_str, portfolio),
        )
        df[
            [
                "run_date",
                "portfolio",
                "method",
                "cov_source",
                "confidence_level",
                "var_value",
                "es_value",
            ]
        ].to_sql("var_results", self._conn, if_exists="append", index=False)
        self._conn.commit()

    def read_var_results(
        self, portfolio: str = "equal_weight_4asset", run_date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """
        Read back stored VaR/ES results. Without run_date, returns every
        historical run for this portfolio -- useful later for tracking how
        VaR estimates have drifted as new data arrives (the same "track
        re-fits over time" idea read_garch_params() already supports).
        """
        query = "SELECT * FROM var_results WHERE portfolio = ?"
        params: list = [portfolio]
        if run_date is not None:
            query += " AND run_date = ?"
            params.append(str(pd.Timestamp(run_date).date()))
        query += " ORDER BY run_date, method, cov_source, confidence_level"
        return pd.read_sql(query, self._conn, params=params)

    # ------------------------------------------------------------------
    # Backtest validation results (Phase 4)
    # ------------------------------------------------------------------

    def write_backtest_results(self, results: pd.DataFrame, run_date: pd.Timestamp) -> None:
        """
        Persist one full run_backtest_validation.py run: Kupiec,
        Christoffersen, and Basel summary rows for every method/confidence
        level combination.

        Boolean reject flags are stored as 0/1 (SQLite has no native
        boolean type) -- cast explicitly here rather than relying on
        pandas/sqlite3's implicit conversion, which can be inconsistent
        across pandas versions for columns containing None alongside bools.

        Upsert-by-replace: deletes any existing rows for this run_date
        first, matching the exact convention write_var_results() already
        established -- re-running overwrites that day's run, never
        duplicates it.
        """
        run_date_str = str(pd.Timestamp(run_date).date())

        df = results.copy()
        df["run_date"] = run_date_str
        df["confidence_level"] = df["confidence_level"].astype(str)

        bool_cols = ["kupiec_reject", "christoffersen_reject_ind", "combined_reject"]
        for col in bool_cols:
            df[col] = df[col].map(lambda v: None if pd.isna(v) else int(bool(v)))

        self._conn.execute("DELETE FROM backtest_results WHERE run_date = ?", (run_date_str,))

        columns = [
            "run_date",
            "method",
            "confidence_level",
            "n_obs",
            "n_breaches",
            "breach_rate",
            "kupiec_lr",
            "kupiec_p_value",
            "kupiec_reject",
            "christoffersen_lr_ind",
            "christoffersen_p_ind",
            "christoffersen_reject_ind",
            "combined_lr",
            "combined_p_value",
            "combined_reject",
        ]
        df[columns].to_sql("backtest_results", self._conn, if_exists="append", index=False)
        self._conn.commit()

    def read_backtest_results(self, run_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        Read back stored backtest validation results. Without run_date,
        returns every historical run -- same "track drift over time" idea
        as read_var_results() and read_garch_params().
        """
        query = "SELECT * FROM backtest_results WHERE 1=1"
        params: list = []
        if run_date is not None:
            query += " AND run_date = ?"
            params.append(str(pd.Timestamp(run_date).date()))
        query += " ORDER BY run_date, method, confidence_level"
        return pd.read_sql(query, self._conn, params=params)
