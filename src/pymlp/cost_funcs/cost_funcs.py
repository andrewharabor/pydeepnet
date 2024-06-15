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
    error: Float64 = Float64(1e-9)

    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        return -np.sum(expected * np.log(predicted + self.error))

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        return -expected / (predicted + self.error)
