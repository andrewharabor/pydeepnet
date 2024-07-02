
from pynet._typing import *


class CostFunc(ABC):
    @abstractmethod
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        pass

    @abstractmethod
    def derivative(self, predictions: NDArray, targets: NDArray) -> NDArray:
        pass

    def _assert_shapes(self, predictions: NDArray, targets: NDArray) -> None:
        if predictions.shape[0] == 0:
            raise ValueError("Predictions array is empty")
        if targets.shape[0] == 0:
            raise ValueError("Targets array is empty")
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets arrays have different shapes")


class CrossEntropy(CostFunc):
    offset: Float64 = Float64(1e-9)  # to avoid logarithm of or division by zero

    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        self._assert_probabilities(predictions)
        self._assert_one_hot(targets)
        return -np.sum(targets * np.log(predictions + self.offset))

    def derivative(self, predictions: NDArray, targets: NDArray) -> NDArray:
        self._assert_shapes(predictions, targets)
        self._assert_probabilities(predictions)
        self._assert_one_hot(targets)
        return -targets / (predictions + self.offset)

    def _assert_probabilities(self, predictions: NDArray) -> None:
        if np.any(predictions < 0) or np.any(predictions > 1):
            raise ValueError("Predictions are not valid probabilities")
        if not np.isclose(np.sum(predictions), 1):
            raise ValueError("Predictions do not make up a valid probability distribution")

    def _assert_one_hot(self, targets: NDArray) -> None:
        if not np.all(np.isin(targets, [0, 1])):
            raise ValueError("Targets are not one-hot encoded")
        if np.sum(targets) != 1:
            raise ValueError("Targets do not make up a valid one-hot encoding")

class Huber(CostFunc):
    threshold: Float64  # threshold for quadratic and linear parts

    def __init__(self, threshold: Float64) -> None:
        if threshold < 0:
            raise ValueError("Threshold is negative")
        self.threshold = threshold

    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        diff: NDArray = np.abs(predictions - targets)
        return np.sum(np.where(diff <= self.threshold, 0.5 * (diff ** 2), self.threshold * (diff - 0.5 * self.threshold))) / predictions.shape[0]

    def derivative(self, predictions: NDArray, targets: NDArray) -> NDArray:
        self._assert_shapes(predictions, targets)
        diff: NDArray = predictions - targets
        return np.where(np.abs(diff) <= self.threshold, diff, self.threshold * np.sign(diff)) / predictions.shape[0]


class LogCosh(CostFunc):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return np.sum(np.log(np.cosh(predictions - targets))) / predictions.shape[0]

    def derivative(self, predictions: NDArray, targets: NDArray) -> NDArray:
        self._assert_shapes(predictions, targets)
        return np.tanh(predictions - targets) / predictions.shape[0]


class MeanAbsoluteError(CostFunc):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return np.sum(np.abs(predictions - targets)) / predictions.shape[0]

    def derivative(self, predictions: NDArray, targets: NDArray) -> NDArray:
        self._assert_shapes(predictions, targets)
        return np.sign(predictions - targets) / predictions.shape[0]


class MeanSquaredError(CostFunc):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return np.sum((predictions - targets) ** 2) / (2 * predictions.shape[0])

    def derivative(self, predictions: NDArray, targets: NDArray) -> NDArray:
        self._assert_shapes(predictions, targets)
        return (predictions - targets) / predictions.shape[0]
