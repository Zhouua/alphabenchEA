"""
Configuration dataclasses for the AlphaBench searcher platform.

YAML format (search_config.yaml)
─────────────────────────────────
searching:
  algo:
    name: ea
    param:
      rounds: 10
      N: 30
      mutation_rate: 0.4
      crossover_rate: 0.6
      pool_size: 30

  model:
    name: deepseek-chat
    base_url: https://api.deepseek.com/v1
    key: ${DEEPSEEK_API_KEY}
    temperature: 0.7

backtesting:
  ffo_server: "127.0.0.1:19777"
  market: csi300
  benchmark: SH000300
  search_start: "2016-01-01"
  search_end: "2021-01-01"
  top_k: 30
  n_drop: 1
  fast: true

verification:
  enabled: true
  auto_verify: true
  search_start: "2016-01-01"
  search_end: "2021-01-01"
  val_start: "2021-01-01"
  val_end: "2022-01-01"
  test_start: "2022-01-01"
  test_end: "2025-01-01"
  verification_forward_n: 1

savedir: "./results"
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AlgoConfig:
    """
    Algorithm selection and parameters.

    Attributes:
        name:       Algorithm name — "ea".
        param:      Algorithm-specific parameter dict (forwarded verbatim to the algo).
        seed_file:  Optional path to a seed factor file (warm start).
        seed_top_k: Max seeds to pass to LLM context per round (for controller-based algos).
    """
    name: str = "ea"
    param: Dict[str, Any] = field(default_factory=dict)
    seed_file: str = ""
    seed_top_k: int = 50


@dataclass
class ModelConfig:
    """
    LLM model configuration.

    Attributes:
        name:        Model identifier (e.g. "deepseek-chat", "gpt-4o").
        base_url:    API base URL.  Leave empty for default OpenAI-compatible endpoint.
        key:         API key.  Use "${ENV_VAR}" syntax to read from environment.
        temperature: Sampling temperature (0.0–2.0).
    """
    name: str = "deepseek-chat"
    base_url: str = ""
    key: str = ""
    temperature: float = 0.7

    def resolve_key(self) -> str:
        """Resolve the API key, expanding environment variable references."""
        key = self.key
        if key.startswith("${") and key.endswith("}"):
            env_var = key[2:-1]
            resolved = os.getenv(env_var, "")
            if not resolved:
                raise ValueError(
                    f"Environment variable '{env_var}' is not set. "
                    f"Set it with: export {env_var}=<your-api-key>"
                )
            return resolved
        return key


@dataclass
class BacktestConfig:
    """
    Backtesting configuration — all evaluation goes through the FFO server.

    Attributes:
        ffo_server:    FFO API server address ("host:port").
        market:        Market universe identifier (e.g. "csi300", "csi500").
        benchmark:     Benchmark index for portfolio comparison (e.g. "SH000300").
        search_start:  Search period start date ("YYYY-MM-DD").
        search_end:    Search period end date ("YYYY-MM-DD").
        top_k:         Long-only portfolio size (top-K factors/stocks selected).
                       Only used when fast=False.
        n_drop:        Number of positions dropped per rebalance.
        fast:          Fast mode — compute IC metrics only (True) or full portfolio
                       backtest (False).  Fast mode is ~5-10x quicker.
        n_jobs:        Parallel evaluation workers.
        use_cache:     Persist FFO metrics in SQLite. Disable for large EA
                       batches to avoid unnecessary cache-write contention.
        timeout:          Per-factor evaluation timeout in seconds.
        accept_threshold: Minimum RankIC a factor must achieve to be accepted
                          into the search pool.  Use 0.0 to accept all factors
                          with non-negative RankIC; use a negative value (e.g.
                          -1.0) to effectively disable filtering.
    """
    ffo_server: str = "127.0.0.1:19777"
    market: str = "csi300"
    benchmark: str = "SH000300"
    search_start: str = "2016-01-01"
    search_end: str = "2021-01-01"
    fields: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume", "vwap"]
    )
    label: str = "close_return"
    target_expression: str = "Ref($close, -1)/$close - 1"
    forward_n: int = 1
    top_k: int = 30
    n_drop: int = 1
    account: float = 100_000_000.0
    deal_price: str = "close"
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    limit_threshold: Optional[float] = 0.095
    fast: bool = True
    n_jobs: int = 4
    timeout: int = 120
    use_cache: bool = True
    accept_threshold: float = 0.0

    def get_api_url(self) -> str:
        """Return the full HTTP URL for the FFO API server."""
        server = self.ffo_server
        if not server.startswith("http"):
            server = f"http://{server}"
        return server

    def get_exchange_kwargs(self) -> Dict[str, Any]:
        """Return the Qlib exchange settings used by full portfolio backtests."""
        return {
            "deal_price": self.deal_price,
            "open_cost": self.open_cost,
            "close_cost": self.close_cost,
            "min_cost": self.min_cost,
            "limit_threshold": self.limit_threshold,
        }


@dataclass
class RulerConfig:
    """Fixed Qlib model used to compare complete factor libraries."""

    estimator: str = "ridge"
    alpha: float = 10.0
    fit_intercept: bool = False


@dataclass
class ExportConfig:
    """Factor-library export settings."""

    enabled: bool = False
    directory: str = "factor_library"
    source: str = "alphabench_ea"


@dataclass
class VerificationConfig:
    """
    Verification / validation configuration for out-of-sample evaluation.

    During searching, factors are evaluated on both the search period and
    the validation period. Only search-period metrics are used for algorithm
    decisions (no data leakage). Val-period metrics are saved for analysis only.

    After searching completes, a separate script evaluates the final portfolio
    on the test period.

    Attributes:
        enabled:       Whether to run validation evaluation during search.
        auto_verify:   Automatically run validation after each round.
        search_start:  Search period start (same as backtesting for reference).
        search_end:    Search period end.
        val_start:     Validation period start date.
        val_end:       Validation period end date.
        test_start:    Test period start date (used by test script only).
        test_end:      Test period end date (used by test script only).
        verification_forward_n: Number of forward periods for verification.
    """
    enabled: bool = True
    auto_verify: bool = True
    search_start: str = "2016-01-01"
    search_end: str = "2021-01-01"
    val_start: str = "2021-01-01"
    val_end: str = "2022-01-01"
    test_start: str = "2022-01-01"
    test_end: str = "2025-01-01"
    verification_forward_n: int = 1
    test_policy: str = "manual"


@dataclass
class SearchingConfig:
    """
    Search process configuration.

    Attributes:
        algo:  Algorithm selection and parameters.
        model: LLM model to use for factor generation.
    """
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class FullConfig:
    """
    Complete configuration for a search run.

    Attributes:
        searching:    Search algorithm and model settings.
        backtesting:  FFO backtesting settings.
        verification: Verification / validation settings.
        savedir:      Directory to save results.
    """
    searching: SearchingConfig = field(default_factory=SearchingConfig)
    backtesting: BacktestConfig = field(default_factory=BacktestConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    savedir: str = "./results"


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_config_from_yaml(yaml_path: str) -> FullConfig:
    """
    Load a FullConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed FullConfig object.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return load_config_from_dict(data or {})


def load_config_from_dict(data: Dict[str, Any]) -> FullConfig:
    """
    Build a FullConfig from a raw dictionary (e.g. parsed YAML).

    Args:
        data: Configuration dictionary.

    Returns:
        FullConfig object.
    """
    # ── searching ─────────────────────────────────────────────────────
    search_data = data.get("searching", {})

    algo_data = search_data.get("algo", {})
    algo_config = AlgoConfig(
        name=algo_data.get("name", "ea"),
        param=algo_data.get("param", {}),
        seed_file=algo_data.get("seed_file", ""),
        seed_top_k=algo_data.get("seed_top_k", 50),
    )

    model_data = search_data.get("model", {})
    model_config = ModelConfig(
        name=model_data.get("name", "deepseek-chat"),
        base_url=model_data.get("base_url", ""),
        key=model_data.get("key", ""),
        temperature=float(model_data.get("temperature", 0.7)),
    )

    searching_config = SearchingConfig(algo=algo_config, model=model_config)

    # ── backtesting ───────────────────────────────────────────────────
    bt_data = data.get("backtesting", {})
    backtest_config = BacktestConfig(
        ffo_server=bt_data.get("ffo_server", "127.0.0.1:19777"),
        market=bt_data.get("market", "csi300"),
        benchmark=bt_data.get("benchmark", "SH000300"),
        search_start=bt_data.get("search_start", bt_data.get("period_start", "2016-01-01")),
        search_end=bt_data.get("search_end", bt_data.get("period_end", "2021-01-01")),
        fields=list(bt_data.get("fields", ["open", "high", "low", "close", "volume", "vwap"])),
        label=bt_data.get("label", "close_return"),
        target_expression=bt_data.get(
            "target_expression", "Ref($close, -1)/$close - 1"
        ),
        forward_n=int(bt_data.get("forward_n", 1)),
        top_k=int(bt_data.get("top_k", 30)),
        n_drop=int(bt_data.get("n_drop", 1)),
        account=float(bt_data.get("account", 100_000_000)),
        deal_price=bt_data.get("deal_price", "close"),
        open_cost=float(bt_data.get("open_cost", 0.0005)),
        close_cost=float(bt_data.get("close_cost", 0.0015)),
        min_cost=float(bt_data.get("min_cost", 5)),
        limit_threshold=(
            None if bt_data.get("limit_threshold", 0.095) is None
            else float(bt_data.get("limit_threshold", 0.095))
        ),
        fast=bool(bt_data.get("fast", True)),
        n_jobs=int(bt_data.get("n_jobs", 4)),
        timeout=int(bt_data.get("timeout", 120)),
        use_cache=bool(bt_data.get("use_cache", True)),
        accept_threshold=float(bt_data.get("accept_threshold", 0.0)),
    )

    # ── verification ──────────────────────────────────────────────────
    ver_data = data.get("verification", {})
    verification_config = VerificationConfig(
        enabled=bool(ver_data.get("enabled", True)),
        auto_verify=bool(ver_data.get("auto_verify", True)),
        search_start=ver_data.get("search_start", backtest_config.search_start),
        search_end=ver_data.get("search_end", backtest_config.search_end),
        val_start=ver_data.get("val_start", "2021-01-01"),
        val_end=ver_data.get("val_end", "2022-01-01"),
        test_start=ver_data.get("test_start", "2022-01-01"),
        test_end=ver_data.get("test_end", "2025-01-01"),
        verification_forward_n=int(ver_data.get("verification_forward_n", 1)),
        test_policy=ver_data.get("test_policy", "manual"),
    )

    ruler_data = data.get("ruler", {})
    ruler_config = RulerConfig(
        estimator=str(ruler_data.get("estimator", "ridge")),
        alpha=float(ruler_data.get("alpha", 10.0)),
        fit_intercept=bool(ruler_data.get("fit_intercept", False)),
    )

    export_data = data.get("export", {})
    export_config = ExportConfig(
        enabled=bool(export_data.get("enabled", False)),
        directory=str(export_data.get("directory", "factor_library")),
        source=str(export_data.get("source", "alphabench_ea")),
    )

    savedir = data.get("savedir", "./results")

    return FullConfig(
        searching=searching_config,
        backtesting=backtest_config,
        verification=verification_config,
        ruler=ruler_config,
        export=export_config,
        savedir=savedir,
    )
