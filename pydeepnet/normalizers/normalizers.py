
from pydeepnet._typing import *


class Normalizer(ABC):
    _adapted: bool = False

    @abstractmethod
    def fit(self, inputs: NDArray) -> None:
        pass

    @abstractmethod
    def transform(self, inputs: NDArray) -> NDArray:
        pass

    @abstractmethod
    def undo(self, inputs: NDArray) -> NDArray:
        pass

    def _assert_shape(self, inputs: NDArray) -> None:
        if inputs.shape[0] == 0:
            raise ValueError("Inputs array is empty")

    def _assert_fit(self) -> None:
        if not self._adapted:
            raise RuntimeError("Normalizer has not been fit to input data through `fit()`")


class DecimalScaling(Normalizer):
    scale: NDArray

    def fit(self, inputs: NDArray) -> None:
        self._assert_shape(inputs)
        self._adapted = True
        self.scale = np.ceil(np.log10(np.maximum(np.max(np.abs(inputs)), 1.0)))
        self.scale = np.where(self.scale < 0, 0, self.scale)  # Prevent scaling by negative power of ten

    def transform(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        self._assert_fit()
        return inputs / (10 ** self.scale)

    def undo(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        self._assert_fit()
        return inputs * (10 ** self.scale)


class MinMax(Normalizer):
    min: NDArray
    max: NDArray

    def fit(self, inputs: NDArray) -> None:
        self._assert_shape(inputs)
        self._adapted = True
        self.min = np.min(inputs, axis=0)
        self.max = np.max(inputs, axis=0)

    def transform(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        self._assert_fit()
        diff: NDArray = self.max - self.min
        return (inputs - self.min) / np.where(diff == 0, 1, diff)  # Prevent division by zero

    def undo(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        self._assert_fit()
        return inputs * (self.max - self.min) + self.min


class ZScore(Normalizer):
    mean: NDArray
    std_dev: NDArray

    def fit(self, inputs: NDArray) -> None:
        self._assert_shape(inputs)
        self._adapted = True
        self.mean = np.mean(inputs, axis=0)
        self.std_dev = np.std(inputs, axis=0)
        self.std_dev = np.where(self.std_dev == 0, 1, self.std_dev)  # Prevent division by zero

    def transform(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        self._assert_fit()
        return (inputs - self.mean) / self.std_dev

    def undo(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        self._assert_fit()
        return inputs * self.std_dev + self.mean
