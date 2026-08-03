"""FFO-local Qlib operators used by AlphaBench factor expressions."""

from __future__ import annotations

import numpy as np
from qlib.data.ops import ElemOperator, NpElemOperator


class Sqrt(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "sqrt")


class Exp(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "exp")


class Square(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "square")


class Sin(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "sin")


class Cos(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "cos")


class Tan(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "tan")


class Tanh(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "tanh")


class Reciprocal(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "reciprocal")


class Clip(ElemOperator):
    def __init__(self, feature, a_min=None, a_max=None):
        if a_min is None and a_max is None:
            raise ValueError("Clip requires at least one bound")
        self.feature = feature
        self.a_min = a_min
        self.a_max = a_max

    def _load_internal(self, instrument, start_index, end_index, *args):
        values = self.feature.load(instrument, start_index, end_index, *args).astype(np.float32)
        return np.clip(values, self.a_min, self.a_max)


CUSTOM_OPS = [Sqrt, Exp, Square, Sin, Cos, Tan, Tanh, Reciprocal, Clip]
