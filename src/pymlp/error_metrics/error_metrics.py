
from pymlp.typing import *


class ErrorMetric(ABC):
    @abstractmethod
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        pass

    def _assert_shapes(self, predictions: NDArray, targets: NDArray) -> None:
        if predictions.shape[0] == 0:
            raise ValueError("`predictions` array is empty")
        if targets.shape[0] == 0:
            raise ValueError("`targets` array is empty")
        if predictions.shape != targets.shape:
            raise ValueError("`predictions` and `targets` arrays have different shapes")


class MeanAbsolutePercentageError(ErrorMetric):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return npla.norm((predictions - targets) / targets, ord=1) / predictions.shape[0]


class PercentCorrect(ErrorMetric):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        self._assert_shapes(predictions, targets)
        return np.sum(np.round(predictions) == targets) / predictions.shape[0]
