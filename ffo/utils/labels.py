"""Canonical named Qlib target expressions used by FFO."""

LABEL_MAP = {
    "close_return": "Ref($close, -1)/$close - 1",
    "close_return_lag": "Ref($close, -2)/Ref($close, -1) - 1",
    "open_to_open_10d": "Ref($open, -11)/Ref($open, -1) - 1",
    "close": "Ref($close, -1)",
}
