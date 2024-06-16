
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
    offset: Float64 # to avoid logarithm of or division by zero

    def __init__(self, offset: Float64=Float64(1e-9)):
        self.offset = offset

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


class Huber(CostFunc):
    threshold: Float64  # threshold for quadratic and linear parts

    def __init__(self, threshold: Float64=Float64(1.0)):
        self.threshold = threshold

    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        diff: NDArray = np.abs(predicted - expected)
        return np.sum(np.where(diff <= self.threshold, 0.5 * (diff ** 2), self.threshold * (diff - 0.5 * self.threshold))) / predicted.shape[0]

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        diff: NDArray = predicted - expected
        return np.where(np.abs(diff) <= self.threshold, diff, self.threshold * np.sign(diff)) / predicted.shape[0]
