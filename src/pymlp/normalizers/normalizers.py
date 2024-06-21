
from pymlp.typing import *


class Normalizer(ABC):
    @abstractmethod
    def adapt(self, inputs: NDArray) -> None:
        pass

    @abstractmethod
    def transform(self, inputs: NDArray) -> NDArray:
        pass

    @abstractmethod
    def undo(self, norm_inputs: NDArray) -> NDArray:
        pass


class ZScore(Normalizer):
    mean: NDArray
    std_dev: NDArray

    def adapt(self, inputs: NDArray) -> None:
        self.mean = np.mean(inputs, axis=0)
        self.std_dev = np.std(inputs, axis=0)

    def transform(self, inputs: NDArray) -> NDArray:
        return (inputs - self.mean) / self.std_dev

    def undo(self, norm_inputs: NDArray) -> NDArray:
        return norm_inputs * self.std_dev + self.mean


class MinMax(Normalizer):
    min: NDArray
    max: NDArray

    def adapt(self, inputs: NDArray) -> None:
        self.min = np.min(inputs, axis=0)
        self.max = np.max(inputs, axis=0)

    def transform(self, inputs: NDArray) -> NDArray:
        return (inputs - self.min) / (self.max - self.min)

    def undo(self, norm_inputs: NDArray) -> NDArray:
        return norm_inputs * (self.max - self.min) + self.min


class DecimalScaling(Normalizer):
    scale: NDArray

    def adapt(self, inputs: NDArray) -> None:
        self.scale = np.ceil(np.log10(np.max(np.abs(inputs))))

    def transform(self, inputs: NDArray) -> NDArray:
        return inputs / (10 ** self.scale)

    def undo(self, norm_inputs: NDArray) -> NDArray:
        return norm_inputs * (10 ** self.scale)
