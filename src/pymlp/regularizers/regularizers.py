
from abc import ABC, abstractmethod

from pymlp.typing import *


class Regularizer(ABC):
    @abstractmethod
    def compute(self, weights: NDArray) -> Float64:
        pass

    @abstractmethod
    def derivative(self, weights: NDArray) -> NDArray:
        pass

    def _assert_shape(self, weights: NDArray) -> None:
        if weights.shape[0] == 0:
            raise ValueError("Weights array is empty")

    def _assert_non_negative(self, penalty: Float64) -> None:
        if penalty < 0:
            raise ValueError("Penalty is negative")


class L1Norm(Regularizer):
    penalty: Float64

    def __init__(self, penalty: Float64) -> None:
        self._assert_non_negative(penalty)
        self.penalty = penalty

    def compute(self, weights: NDArray) -> Float64:
        self._assert_shape(weights)
        return self.penalty * np.sum(np.abs(weights))

    def derivative(self, weights: NDArray) -> NDArray:
        self._assert_shape(weights)
        return self.penalty * np.sign(weights)


class L2Norm(Regularizer):
    penalty: Float64

    def __init__(self, penalty: Float64) -> None:
        self._assert_non_negative(penalty)
        self.penalty = penalty

    def compute(self, weights: NDArray) -> Float64:
        self._assert_shape(weights)
        return self.penalty / 2 * np.sum(weights ** 2)

    def derivative(self, weights: NDArray) -> NDArray:
        self._assert_shape(weights)
        return self.penalty * weights


class ElasticNet(Regularizer):
    l1_regularizer: L1Norm
    l2_regularizer: L2Norm

    def __init__(self, l1_penalty: Float64, l2_penalty: Float64) -> None:
        self._assert_non_negative(l1_penalty)
        self._assert_non_negative(l2_penalty)
        self.l1_regularizer = L1Norm(l1_penalty)
        self.l2_regularizer = L2Norm(l2_penalty)

    def compute(self, weights: NDArray) -> Float64:
        self._assert_shape(weights)
        return self.l1_regularizer.compute(weights) + self.l2_regularizer.compute(weights)

    def derivative(self, weights: NDArray) -> NDArray:
        self._assert_shape(weights)
        return self.l1_regularizer.derivative(weights) + self.l2_regularizer.derivative(weights)
