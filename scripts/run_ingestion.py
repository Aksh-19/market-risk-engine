"""
Phase 1 entry point: run the full ingestion pipeline end to end.

    python scripts/run_ingestion.py --config configs/universe.yaml

This script is intentionally thin — it's an orchestration layer that calls
into `risk_engine.data`, not a place where new logic should live. If you
find yourself writing real logic here, it belongs in a module instead. This
mirrors how production pipelines are structured: a small, readable "driver"
script and a well-tested library underneath it.

WHAT THIS PRODUCES
--------------------
`data/processed/returns_matrix.parquet` — the aligned, validated, log-return
matrix that Phase 2 (volatility modeling) will load as its starting point.
This file is the formal contract between Phase 1 and Phase 2: Phase 2 code
should never need to know Yahoo Finance exists.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from risk_engine.data import DataConfig, DataLoader, build_returns_matrix, validate_price_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str, output_path: str) -> None:
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    config = DataConfig(**raw_cfg)

    loader = DataLoader(config)
    price_frames = loader.load_universe()

    logger.info("=== Data validation ===")
    all_clean = True
    for ticker, df in price_frames.items():
        report = validate_price_series(df, ticker=ticker, price_field=config.price_field)
        print(report.summary())
        if not report.is_clean:
            all_clean = False
    if not all_clean:
        logger.warning(
            "One or more series had data-quality issues (see above). "
            "Proceeding anyway since these are diagnostic flags, not hard "
            "failures — but review them before trusting downstream VaR output."
        )

    logger.info("=== Building aligned returns matrix ===")
    returns_matrix = build_returns_matrix(
        price_frames, price_field=config.price_field, method="log"
    )
    logger.info(
        "Returns matrix: %d dates x %d assets (%s to %s)",
        returns_matrix.shape[0],
        returns_matrix.shape[1],
        returns_matrix.index.min().date(),
        returns_matrix.index.max().date(),
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    returns_matrix.to_parquet(out_path)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: data ingestion pipeline")
    parser.add_argument("--config", default="configs/universe.yaml")
    parser.add_argument("--output", default="data/processed/returns_matrix.parquet")
    args = parser.parse_args()
    main(args.config, args.output)
