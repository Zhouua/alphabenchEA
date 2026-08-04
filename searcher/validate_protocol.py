#!/usr/bin/env python3
"""Validate the aligned AlphaBench-EA protocol without reading test returns."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from searcher.config.config import load_config_from_yaml


EXPECTED = {
    "market": "csi300",
    "benchmark": "SH000300",
    "train": ("2010-01-01", "2019-11-30"),
    "valid": ("2020-01-01", "2021-11-30"),
    "test": ("2022-01-01", "2025-12-31"),
    "label": "open_to_open_10d",
    "target": "Ref($open,-11)/Ref($open,-1)-1",
    "fields": {"open", "high", "low", "close", "volume", "vwap"},
}


def _compact(expression: str) -> str:
    return "".join(expression.split())


def validate_config(config) -> None:
    actual = {
        "market": config.backtesting.market.lower(),
        "benchmark": config.backtesting.benchmark.upper(),
        "train": (config.backtesting.search_start, config.backtesting.search_end),
        "valid": (config.verification.val_start, config.verification.val_end),
        "test": (config.verification.test_start, config.verification.test_end),
        "label": config.backtesting.label,
        "target": _compact(config.backtesting.target_expression),
        "fields": set(config.backtesting.fields),
    }
    for key, expected in EXPECTED.items():
        if actual[key] != expected:
            raise ValueError(f"{key} mismatch: expected {expected!r}, got {actual[key]!r}")
    if config.backtesting.forward_n != 1:
        raise ValueError("forward_n must be 1; the exact 10-day return is already the label")
    if config.backtesting.use_cache:
        raise ValueError("use_cache must be false for contention-free EA batch evaluation")
    if (config.ruler.estimator.lower(), config.ruler.alpha, config.ruler.fit_intercept) != (
        "ridge", 10.0, False,
    ):
        raise ValueError("ruler must be Qlib LinearModel ridge/alpha=10/fit_intercept=false")
    strategy = (
        config.backtesting.top_k, config.backtesting.n_drop,
        config.backtesting.deal_price, config.backtesting.account,
        config.backtesting.open_cost, config.backtesting.close_cost,
        config.backtesting.min_cost,
    )
    if strategy != (50, 5, "open", 100_000_000.0, 0.0005, 0.0015, 5.0):
        raise ValueError(f"strategy mismatch: {strategy!r}")
    if config.verification.test_policy != "public_only":
        raise ValueError("verification.test_policy must be public_only")


def validate_train_valid_data(config) -> None:
    """Read a tiny train/valid sample only; never query test prices or labels."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/alphabench-matplotlib")
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    import pandas as pd

    qlib.init(
        provider_uri=os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"),
        region=REG_CN,
        kernels=1,
    )
    fields = [f"${name}" for name in config.backtesting.fields]
    instruments = D.instruments(config.backtesting.market)
    for label, date in (
        ("train", config.backtesting.search_start),
        ("valid", config.verification.val_start),
    ):
        sample_end = (pd.Timestamp(date) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        frame = D.features(instruments, fields, start_time=date, end_time=sample_end, freq="day")
        if frame.empty:
            raise RuntimeError(f"no {label} data at/after {date}")
        missing = [field for field in fields if field not in frame or frame[field].notna().sum() == 0]
        if missing:
            raise RuntimeError(f"{label} data missing fields: {missing}")
        print(f"{label}: {len(frame)} rows, six fields available")
    bench = D.features(
        [config.backtesting.benchmark], ["$close"],
        start_time=config.verification.val_start,
        end_time=(pd.Timestamp(config.verification.val_start) + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        freq="day",
    )
    if bench.empty:
        raise RuntimeError(f"benchmark unavailable: {config.backtesting.benchmark}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="example/search/configs/alphabench_ea_alphamining.yaml",
    )
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    config = load_config_from_yaml(args.config)
    validate_config(config)
    print("configuration: aligned")
    if not args.skip_data:
        validate_train_valid_data(config)
        print("train/validation data: ready (test returns were not read)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
