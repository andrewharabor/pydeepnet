
from pydeepnet._typing import *


class ErrorMetric(ABC):
    @abstractmethod
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        pass

    def _assert_shapes(self, predictions: NDArray, targets: NDArray) -> None:
        if predictions.shape[0] == 0:
            raise ValueError("Predictions array is empty")
        if targets.shape[0] == 0:
            raise ValueError("Targets array is empty")
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets arrays have different shapes")


class MeanAbsolutePercentageError(ErrorMetric):
    offset: Float64 = Float64(1e-9)

    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return 100 * np.sum(np.abs((predictions - targets) / (targets + self.offset))) / np.prod(predictions.shape)


class PercentageAccuracy(ErrorMetric):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        self._assert_probabilities(predictions)
        self._assert_one_hot(targets)
        return 100 * np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) / predictions.shape[0]

    def _assert_probabilities(self, predictions: NDArray) -> None:
        if np.any(predictions < 0) or np.any(predictions > 1):
            raise ValueError("Predictions are not valid probabilities")
        if not np.allclose(np.sum(predictions, axis=1), 1):
            raise ValueError("Predictions do not make up a valid probability distribution")

    def _assert_one_hot(self, targets: NDArray) -> None:
        if not np.all(np.isin(targets, [0, 1])):
            raise ValueError("Targets are not one-hot encoded")
        if np.all(np.sum(targets, axis=1) != 1):
            raise ValueError("Targets do not make up a valid one-hot encoding")
