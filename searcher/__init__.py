# AlphaBench-EA factor discovery platform.
#
# Public API
# ──────────
#   SearchPipeline      end-to-end pipeline (seeds → baseline → algo → results)
#   Backtester          FFO-backed factor evaluator
#   create_algo         instantiate the EA search algorithm
#   register_algo       register a custom algo
#   list_algos          list all registered algo names
#   load_config_from_yaml / load_config_from_dict
#
from .pipeline import SearchPipeline
from .backtester import Backtester
from .algo import (
    BaseAlgo,
    EAAlgo,
    EA_Searcher,
    create_algo,
    register_algo,
    list_algos,
)
from .config.config import (
    FullConfig,
    SearchingConfig,
    BacktestConfig,
    AlgoConfig,
    ModelConfig,
    load_config_from_yaml,
    load_config_from_dict,
)

__all__ = [
    # Pipeline
    "SearchPipeline",
    "Backtester",
    # Algo registry
    "create_algo",
    "register_algo",
    "list_algos",
    # Algo classes
    "BaseAlgo",
    "EAAlgo",
    "EA_Searcher",
    # Config
    "FullConfig",
    "SearchingConfig",
    "BacktestConfig",
    "AlgoConfig",
    "ModelConfig",
    "load_config_from_yaml",
    "load_config_from_dict",
]
