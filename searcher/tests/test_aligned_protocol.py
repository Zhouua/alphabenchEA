import json
from pathlib import Path

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
