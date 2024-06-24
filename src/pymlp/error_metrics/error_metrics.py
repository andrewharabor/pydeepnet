
from pymlp.typing import *


class ErrorMetric(ABC):
    @abstractmethod
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        pass


class MeanAbsolutePercentageError(ErrorMetric):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        return npla.norm((predictions - targets) / targets, ord=1) / predictions.shape[0]


class PercentCorrect(ErrorMetric):
    def compute(self, predictions: NDArray, targets: NDArray) -> Float64:
        return np.sum(np.round(predictions) == targets) / predictions.shape[0]
