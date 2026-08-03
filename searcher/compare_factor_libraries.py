#!/usr/bin/env python3
"""Public, fixed-protocol comparison of Qlib factor registries.

This program is deliberately separate from the EA search pipeline.  It fits the
same Qlib Ridge ruler to every supplied factor set and runs one native
TopkDropout backtest.  Test access requires an explicit ``--public-test`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from searcher.config.config import load_config_from_yaml

_ANNUAL = 252
_EPS = 1e-10


def load_registry(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Load strings or AlphaBench/AlphaMining registry records."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    names, expressions, seen = [], [], set()
    for index, record in enumerate(records, 1):
        if isinstance(record, str):
            name, expression = f"factor{index}", record
        else:
            name = str(record.get("name") or f"factor{record.get('id', index)}")
            expression = record.get("qlib_expr", record.get("expression", record.get("expr")))
        expression = str(expression or "").strip()
        if not expression or expression in seen:
            continue
        seen.add(expression)
        names.append(name)
        expressions.append(expression)
    if not expressions:
        raise ValueError(f"no Qlib expressions found in {path}")
    if len(set(names)) != len(names):
        names = [f"factor{index}" for index in range(1, len(names) + 1)]
    return names, expressions


def _dataset_config(names, expressions, config, segment: str) -> dict:
    segment_end = (
        config.verification.val_end if segment == "valid" else config.verification.test_end
    )
    segments = {
        "train": (config.backtesting.search_start, config.backtesting.search_end),
        "valid": (config.verification.val_start, config.verification.val_end),
        "test": (config.verification.test_start, config.verification.test_end),
    }
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "instruments": config.backtesting.market,
                    "start_time": config.backtesting.search_start,
                    "end_time": segment_end,
                    "data_loader": {
                        "class": "QlibDataLoader",
                        "module_path": "qlib.data.dataset.loader",
                        "kwargs": {"config": {
                            "feature": (expressions, names),
                            "label": ([config.backtesting.target_expression], ["LABEL0"]),
                        }},
                    },
                    "infer_processors": [
                        {"class": "ProcessInf"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
                        {"class": "Fillna", "kwargs": {
                            "fields_group": "feature", "fill_value": 0,
                        }},
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                    "drop_raw": True,
                },
            },
            "segments": segments,
        },
    }


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _stats(returns: pd.Series) -> Dict[str, float]:
    values = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return {"ar": 0.0, "ir": 0.0, "mdd": 0.0, "cr": 0.0}
    annual_return = _finite(values.mean() * _ANNUAL)
    std = values.std()
    information_ratio = _finite(values.mean() / std * math.sqrt(_ANNUAL)) if std > _EPS else 0.0
    cumulative = values.cumsum()
    max_drawdown = _finite((cumulative - cumulative.cummax()).min())
    calmar = _finite(annual_return / abs(max_drawdown)) if max_drawdown else 0.0
    return {"ar": annual_return, "ir": information_ratio, "mdd": max_drawdown, "cr": calmar}


def _evaluate_library(name: str, registry: Path, config, segment: str):
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D
    from qlib.utils import init_instance_by_config

    factor_names, expressions = load_registry(registry)
    dataset = init_instance_by_config(
        _dataset_config(factor_names, expressions, config, segment)
    )
    model = init_instance_by_config({
        "class": "LinearModel",
        "module_path": "qlib.contrib.model.linear",
        "kwargs": {
            "estimator": config.ruler.estimator,
            "alpha": config.ruler.alpha,
            "fit_intercept": config.ruler.fit_intercept,
        },
    })
    model.fit(dataset)
    prediction = model.predict(dataset, segment=segment)
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.iloc[:, 0]

    start, end = (
        (config.verification.val_start, config.verification.val_end)
        if segment == "valid"
        else (config.verification.test_start, config.verification.test_end)
    )
    calendar = D.calendar(freq="day")
    if pd.Timestamp(end) >= calendar[-1]:
        end = pd.Timestamp(calendar[-2]).strftime("%Y-%m-%d")

    strategy = TopkDropoutStrategy(
        signal=prediction,
        topk=config.backtesting.top_k,
        n_drop=config.backtesting.n_drop,
    )
    portfolio_metrics, _ = qlib_backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        account=config.backtesting.account,
        benchmark=config.backtesting.benchmark,
        exchange_kwargs={
            "freq": "day",
            "codes": config.backtesting.market,
            **config.backtesting.get_exchange_kwargs(),
        },
    )
    report, _positions = portfolio_metrics["1day"]
    series = pd.DataFrame({
        "gross": report["return"],
        "net": report["return"] - report["cost"],
        "bench": report["bench"],
        "turnover": report["turnover"],
    }).dropna()
    gross, net = _stats(series["gross"]), _stats(series["net"])
    excess_gross = _stats(series["gross"] - series["bench"])
    excess_net = _stats(series["net"] - series["bench"])
    summary = {
        "library": name,
        "registry": str(registry),
        "factor_count": len(expressions),
        "segment": segment,
        "ar_gross": gross["ar"],
        "ar_net": net["ar"],
        "aer_gross": excess_gross["ar"],
        "aer_net": excess_net["ar"],
        "sharpe_net": net["ir"],
        "ir_net": excess_net["ir"],
        "mdd_net": net["mdd"],
        "mdd_excess_net": excess_net["mdd"],
        "calmar_excess_net": excess_net["cr"],
        "turnover_daily": _finite(series["turnover"].mean()),
    }
    curves = pd.DataFrame({
        f"{name}_cum_net": series["net"].cumsum(),
        f"{name}_cum_excess_net": (series["net"] - series["bench"]).cumsum(),
    })
    return summary, curves


def _write_outputs(rows: List[dict], curves: List[pd.DataFrame], output_dir: Path, config, segment: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    protocol = {
        "segment": segment,
        "period": (
            [config.verification.val_start, config.verification.val_end]
            if segment == "valid"
            else [config.verification.test_start, config.verification.test_end]
        ),
        "train": [config.backtesting.search_start, config.backtesting.search_end],
        "market": config.backtesting.market,
        "benchmark": config.backtesting.benchmark,
        "target_expression": config.backtesting.target_expression,
        "ruler": {
            "class": "qlib.contrib.model.linear.LinearModel",
            "estimator": config.ruler.estimator,
            "alpha": config.ruler.alpha,
            "fit_intercept": config.ruler.fit_intercept,
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "topk": config.backtesting.top_k,
            "n_drop": config.backtesting.n_drop,
            "deal_price": config.backtesting.deal_price,
            "account": config.backtesting.account,
            "open_cost": config.backtesting.open_cost,
            "close_cost": config.backtesting.close_cost,
            "min_cost": config.backtesting.min_cost,
            "limit_threshold": config.backtesting.limit_threshold,
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve_frame = pd.concat(curves, axis=1).sort_index()
    curve_frame.to_csv(output_dir / "cumulative_returns.csv")
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for row in rows:
            library = row["library"]
            axes[0].plot(curve_frame.index, curve_frame[f"{library}_cum_net"], label=library)
            axes[1].plot(
                curve_frame.index,
                curve_frame[f"{library}_cum_excess_net"],
                label=library,
            )
        axes[0].set_title("Cumulative net return (arithmetic)")
        axes[1].set_title("Cumulative net excess return vs benchmark")
        for axis in axes:
            axis.axhline(0, color="black", linewidth=0.7)
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "comparison.png", dpi=160)
        plt.close(figure)
    except Exception as exc:
        print(f"warning: comparison plot was not generated: {exc}")

    display = [
        ("Factors", "factor_count"), ("Net AR", "ar_net"),
        ("Net AER", "aer_net"), ("Net IR", "ir_net"),
        ("Net MDD", "mdd_net"), ("Daily turnover", "turnover_daily"),
    ]
    lines = ["# Factor-library comparison", "", "| Metric | " + " | ".join(r["library"] for r in rows) + " |",
             "|---|" + "---:|" * len(rows)]
    for label, key in display:
        values = []
        for row in rows:
            value = row[key]
            values.append(str(value) if key == "factor_count" else f"{value:.4f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(["", "All libraries use the same Qlib Ridge ruler and TopkDropout protocol."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="example/search/configs/alphabench_ea_alphamining.yaml",
    )
    parser.add_argument(
        "--library", action="append", default=[], metavar="NAME=REGISTRY",
        help="repeat for each factor library",
    )
    parser.add_argument("--segment", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--public-test", action="store_true",
        help="explicitly authorize the held-out public test evaluation",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    if args.segment == "test" and not args.public_test:
        parser.error("test is held out; rerun the public evaluator with --public-test")
    if config.ruler.estimator.lower() != "ridge":
        parser.error("aligned comparison requires ruler.estimator=ridge")

    libraries = []
    if args.library:
        for item in args.library:
            if "=" not in item:
                parser.error(f"invalid --library {item!r}; expected NAME=REGISTRY")
            name, path = item.split("=", 1)
            libraries.append((name, Path(path)))
    else:
        libraries = [
            (
                "AlphaBench-EA",
                Path(config.savedir) / config.export.directory / "registry.json",
            ),
            ("AlphaMining", _ROOT.parent / "alpha_mining" / "factors" / "registry.json"),
        ]
    for name, path in libraries:
        if not path.exists():
            parser.error(f"registry for {name} not found: {path}")

    # Avoid matplotlib writing to a non-workspace home during Qlib imports.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/alphabench-matplotlib")
    import qlib
    from qlib.constant import REG_CN
    from ffo.utils.qlib_custom_ops import CUSTOM_OPS

    qlib.init(
        provider_uri=os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"),
        region=REG_CN,
        kernels=max(1, min(8, os.cpu_count() or 1)),
        custom_ops=CUSTOM_OPS,
    )
    rows, curves = [], []
    for name, path in libraries:
        row, curve = _evaluate_library(name, path, config, args.segment)
        rows.append(row)
        curves.append(curve)
        print(f"{name}: factors={row['factor_count']} net IR={row['ir_net']:.4f}")

    output_dir = Path(args.output_dir or f"runs/library_comparison/{args.segment}")
    _write_outputs(rows, curves, output_dir, config, args.segment)
    print(f"comparison written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
