
from pymlp.typing import *


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
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return npla.norm((predictions - targets) / targets, ord=1) / predictions.shape[0]


class PercentCorrect(ErrorMetric):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) / predictions.shape[0]
