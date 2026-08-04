import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ffo.utils.factor_store import FactorStore
from ffo.utils.labels import LABEL_MAP
from searcher.config.config import load_config_from_yaml
from searcher.factor_library import export_factor_library
from searcher.validate_protocol import validate_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "example/search/configs/alphabench_ea_alphamining.yaml"


def test_aligned_config_is_exact():
    config = load_config_from_yaml(str(CONFIG))
    validate_config(config)
    assert LABEL_MAP["open_to_open_10d"] == "Ref($open, -11)/Ref($open, -1) - 1"
    assert config.backtesting.forward_n == 1
    assert config.backtesting.use_cache is False
    assert config.verification.test_policy == "public_only"


def test_factor_library_is_alphamining_compatible_and_test_free(tmp_path):
    config = load_config_from_yaml(str(CONFIG))
    factors = [
        {
            "name": "f1",
            "expression": "Mean($close, 5)",
            "metrics": {"rank_ic": 0.03},
            "val_metrics": {"rank_ic": 0.02},
        },
        {
            "name": "duplicate",
            "expression": "Mean($close, 5)",
            "metrics": {"rank_ic": 0.01},
        },
    ]
    paths = export_factor_library(factors, config, tmp_path)
    registry = json.loads(Path(paths["registry"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert registry == [
        {
            "id": 1,
            "name": "f1",
            "qlib_expr": "Mean($close, 5)",
            "source": "alphabench_ea",
            "provenance": "",
            "rationale": "",
            "metrics_train": {"rank_ic": 0.03},
            "metrics_valid": {"rank_ic": 0.02},
        }
    ]
    assert manifest["test_evaluated"] is False
    assert manifest["target"]["expression"] == "Ref($open, -11) / Ref($open, -1) - 1"
    assert manifest["ruler"] == {
        "estimator": "ridge", "alpha": 10.0, "fit_intercept": False,
    }


def test_factor_store_serializes_concurrent_writes(tmp_path):
    store = FactorStore(str(tmp_path / "cache"))

    def write_one(index):
        factor_hash = f"factor-{index}"
        store.register_expression(factor_hash, f"Mean($close, {index + 2})")
        store.put_daily_ic(
            factor_hash,
            "csi300",
            "open_to_open_10d",
            [
                {
                    "date": "2020-01-02",
                    "ic": 0.01,
                    "rank_ic": 0.02,
                    "turnover": 0.1,
                }
            ],
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_one, range(32)))

    stats = store.get_store_stats()
    assert stats["expressions"] == 32
    assert stats["daily_ic_entries"] == 32
