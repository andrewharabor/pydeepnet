from __future__ import annotations

from pymlp.typing import *


class CostFunc(ABC):
    @abstractmethod
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        pass

    @abstractmethod
    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        pass


class MeanSquaredError(CostFunc):
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        return np.sum((predicted - expected) ** 2) / (2 * predicted.shape[0])

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        return (predicted - expected) / predicted.shape[0]


class CrossEntropy(CostFunc):
    offset: Float64 = Float64(1e-9)  # to avoid logarithm of or division by zero

    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        return -np.sum(expected * np.log(predicted + self.offset))

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        return -expected / (predicted + self.offset)


class MeanAbsoluteError(CostFunc):
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        return np.sum(np.abs(predicted - expected)) / predicted.shape[0]

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        return np.sign(predicted - expected) / predicted.shape[0]


class LogCosh(CostFunc):
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        return np.sum(np.log(np.cosh(predicted - expected))) / predicted.shape[0]

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        return np.tanh(predicted - expected) / predicted.shape[0]
