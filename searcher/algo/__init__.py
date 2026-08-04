"""Evolutionary factor-search registry used by AlphaBench-EA."""

from .base import BaseAlgo
from .ea import EAAlgo, EA_Searcher
from typing import Any, Callable, Dict

# Registry maps algo name → class
_REGISTRY: Dict[str, type] = {
    "ea":  EAAlgo,
}


def register_algo(name: str, cls: type) -> None:
    """Register a custom algo class under a given name."""
    _REGISTRY[name.lower()] = cls


def list_algos():
    """Return names of all registered algorithms."""
    return list(_REGISTRY.keys())


def create_algo(
    name: str,
    config: Dict[str, Any],
    evaluate_fn: Callable,
    batch_evaluate_fn: Callable,
    search_fn: Callable,
    batch_evaluate_fn_dict: Callable = None,
    logger=None,
) -> BaseAlgo:
    """
    Instantiate a search algorithm by name.

    Args:
        name:                  Algorithm name ("ea" or a custom registration).
        config:                Algorithm-specific params dict (from YAML algo.param).
        evaluate_fn:           fn(expression: str) -> Dict  (single factor eval via FFO).
        batch_evaluate_fn:     fn(factors: List[Dict]) -> List[Dict]  (list result).
        search_fn:             LLM search callable (e.g. call_qlib_search).
        batch_evaluate_fn_dict: fn(factors: List[Dict]) -> Dict[str, Dict]  (dict by name).
                               If None, derived automatically from batch_evaluate_fn.
        logger:                Optional SearchLogger instance for structured output.

    Returns:
        Configured BaseAlgo instance.

    Raises:
        ValueError: If the algo name is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown algo '{name}'. Available: {list(_REGISTRY.keys())}. "
            f"Register custom algos with register_algo()."
        )
    cls = _REGISTRY[key]
    return cls(
        evaluate_fn=evaluate_fn,
        batch_evaluate_fn=batch_evaluate_fn,
        search_fn=search_fn,
        config=config,
        batch_evaluate_fn_dict=batch_evaluate_fn_dict,
        logger=logger,
    )


__all__ = [
    "BaseAlgo",
    "EAAlgo",
    "EA_Searcher",
    "create_algo",
    "register_algo",
    "list_algos",
]
