"""Export AlphaBench search results as a portable Qlib factor library."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def _unique_factors(factors: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for factor in factors:
        expression = str(factor.get("expression", "")).strip()
        if not expression or expression in seen:
            continue
        seen.add(expression)
        result.append(factor)
    return result


def export_factor_library(factors, config, run_dir: Union[str, Path]) -> Dict[str, str]:
    """Write an AlphaMining-compatible registry plus audit-friendly tables.

    The compatibility contract is intentionally small: every registry record has
    ``id`` and ``qlib_expr``. Extra fields retain AlphaBench train/validation
    evidence without adding any test result.
    """
    export_dir = Path(run_dir) / config.export.directory
    export_dir.mkdir(parents=True, exist_ok=True)
    unique = _unique_factors(factors)

    records = []
    for index, factor in enumerate(unique, 1):
        train_metrics = factor.get("search_metrics", factor.get("metrics", {})) or {}
        valid_metrics = factor.get("val_metrics", {}) or {}
        records.append(
            {
                "id": index,
                "name": factor.get("name") or f"factor{index}",
                "qlib_expr": factor.get("expression", ""),
                "source": config.export.source,
                "provenance": factor.get("provenance", ""),
                "rationale": factor.get("reason", ""),
                "metrics_train": train_metrics,
                "metrics_valid": valid_metrics,
            }
        )

    registry_path = export_dir / "registry.json"
    registry_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = export_dir / "factors.csv"
    columns = [
        "id", "name", "qlib_expr", "train_ic", "train_rank_ic",
        "train_icir", "train_rank_icir", "valid_ic", "valid_rank_ic",
        "valid_icir", "valid_rank_icir", "provenance", "rationale",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            train = rec["metrics_train"]
            valid = rec["metrics_valid"]
            writer.writerow(
                {
                    "id": rec["id"],
                    "name": rec["name"],
                    "qlib_expr": rec["qlib_expr"],
                    "train_ic": train.get("ic"),
                    "train_rank_ic": train.get("rank_ic"),
                    "train_icir": train.get("icir"),
                    "train_rank_icir": train.get("rank_icir"),
                    "valid_ic": valid.get("ic"),
                    "valid_rank_ic": valid.get("rank_ic"),
                    "valid_icir": valid.get("icir"),
                    "valid_rank_icir": valid.get("rank_icir"),
                    "provenance": rec["provenance"],
                    "rationale": rec["rationale"],
                }
            )

    manifest = {
        "format": "alphabench.factor_library.v1",
        "source": config.export.source,
        "factor_count": len(records),
        "registry": registry_path.name,
        "test_evaluated": False,
        "fields": config.backtesting.fields,
        "market": config.backtesting.market,
        "benchmark": config.backtesting.benchmark,
        "segments": {
            "train": [config.backtesting.search_start, config.backtesting.search_end],
            "valid": [config.verification.val_start, config.verification.val_end],
            "test": [config.verification.test_start, config.verification.test_end],
        },
        "target": {
            "name": config.backtesting.label,
            "expression": config.backtesting.target_expression,
        },
        "ruler": asdict(config.ruler),
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
        "test_policy": config.verification.test_policy,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme_path = export_dir / "README.md"
    readme_path.write_text(
        "# AlphaBench-EA factor library\n\n"
        "`registry.json` is directly consumable by AlphaMining's `--registry` option. "
        "It contains train/validation evidence only; test has not been evaluated.\n",
        encoding="utf-8",
    )
    return {
        "directory": str(export_dir),
        "registry": str(registry_path),
        "manifest": str(manifest_path),
        "csv": str(csv_path),
    }
