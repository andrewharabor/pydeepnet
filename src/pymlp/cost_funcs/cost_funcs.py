
from pymlp.typing import *


class CostFunc(ABC):
    @abstractmethod
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        pass

    @abstractmethod
    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        pass

    @staticmethod
    def _assert_shapes(predicted: NDArray, expected: NDArray) -> None:
        if predicted.shape[0] == 0:
            raise ValueError("`predicted` array is empty")
        if expected.shape[0] == 0:
            raise ValueError("`expected` array is empty")
        if predicted.shape != expected.shape:
            raise ValueError("`predicted` and `expected` arrays have different shapes")


class CrossEntropy(CostFunc):
    offset: Float64 = Float64(1e-9)

    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        CrossEntropy._assert_shapes(predicted, expected)
        return -np.sum(expected * np.log(predicted + self.offset))

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        CrossEntropy._assert_shapes(predicted, expected)
        return -expected / (predicted + self.offset)


class Huber(CostFunc):
    threshold: Float64  # threshold for quadratic and linear parts

    def __init__(self, threshold: Float64) -> None:
        self.threshold = threshold

    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        Huber._assert_shapes(predicted, expected)
        diff: NDArray = np.abs(predicted - expected)
        return np.sum(np.where(diff <= self.threshold, 0.5 * (diff ** 2), self.threshold * (diff - 0.5 * self.threshold))) / predicted.shape[0]

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        Huber._assert_shapes(predicted, expected)
        diff: NDArray = predicted - expected
        return np.where(np.abs(diff) <= self.threshold, diff, self.threshold * np.sign(diff)) / predicted.shape[0]


class LogCosh(CostFunc):
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        LogCosh._assert_shapes(predicted, expected)
        return np.sum(np.log(np.cosh(predicted - expected))) / predicted.shape[0]

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        LogCosh._assert_shapes(predicted, expected)
        return np.tanh(predicted - expected) / predicted.shape[0]


class MeanAbsoluteError(CostFunc):
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        MeanAbsoluteError._assert_shapes(predicted, expected)
        return np.sum(np.abs(predicted - expected)) / predicted.shape[0]

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        MeanAbsoluteError._assert_shapes(predicted, expected)
        return np.sign(predicted - expected) / predicted.shape[0]


class MeanSquaredError(CostFunc):
    def compute(self, predicted: NDArray, expected: NDArray) -> Float64:
        MeanSquaredError._assert_shapes(predicted, expected)
        return np.sum((predicted - expected) ** 2) / (2 * predicted.shape[0])

    def derivative(self, predicted: NDArray, expected: NDArray) -> NDArray:
        MeanSquaredError._assert_shapes(predicted, expected)
        return (predicted - expected) / predicted.shape[0]
